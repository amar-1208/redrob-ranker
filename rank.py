#!/usr/bin/env python3
"""
Redrob Hackathon — Hybrid Candidate Ranking Engine
====================================================
Architecture: 5-component weighted scoring with honeypot detection.

Components & Weights:
  1. Skill Match Score         35%  — TF-IDF cosine over JD vs candidate's skill corpus
  2. Career Trajectory Score   30%  — Title alignment + production deployment evidence + company quality
  3. Experience Fit Score      15%  — Non-linear curve targeting 5-9yr sweet spot
  4. Behavioral Availability   15%  — Composite of 6 Redrob signals (recency, responsiveness, openness)
  5. Education / Extras         5%  — Tier, field relevance, GitHub activity, certifications

Disqualifiers (hard filters before scoring):
  - Career that is entirely consulting / IT-services with zero product company experience
  - Pure non-technical roles (no ML/software history at all)
  - Honeypot flag (impossible experience timeline)

Constraints satisfied:
  - No network calls during ranking
  - CPU only, no GPU
  - ≤ 5 minutes on 16 GB RAM for 100 K candidates
  - All 100 rows are real candidates from the pool

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TODAY = date(2026, 6, 12)

# JD-derived signal: the exact text of the job description distilled into
# the semantic corpus our TF-IDF model should rank against.
JD_CORPUS = """
Senior AI Engineer Founding Team Redrob AI talent intelligence platform Pune Noida India hybrid
5 9 years experience applied machine learning NLP retrieval ranking embeddings LLMs fine-tuning
production deployment real users product company startup Series A
embeddings-based retrieval systems sentence-transformers OpenAI embeddings BGE E5 all-MiniLM
embedding drift index refresh retrieval quality regression production
vector databases hybrid search Pinecone Weaviate Qdrant Milvus OpenSearch Elasticsearch FAISS
Python code quality strong python
evaluation frameworks ranking systems NDCG MRR MAP A/B test offline benchmark recruiter feedback loop
LLM fine-tuning LoRA QLoRA PEFT learning-to-rank XGBoost neural ranking
HR tech recruiting tech marketplace products
hybrid retrieval dense sparse BM25 re-ranking cross-encoder
candidate JD matching intelligence layer ranking retrieval search
ship v2 ranking system demonstrably improves recruiter engagement metrics
own intelligence layer product ranking retrieval matching
recommendation system search engine information retrieval
shipped end-to-end ranking search recommendation system real users meaningful scale
strong opinions retrieval hybrid dense evaluation offline online LLM integration fine-tune prompt
scrappy product engineering attitude ship working ranker week
Hyderabad Pune Mumbai Delhi NCR Bangalore India relocation
active Redrob platform job market signal
open source contributions AI ML space
distributed systems large scale inference optimization
""".strip()

# ---------------------------------------------------------------------------
# Skill taxonomies for structured matching
# ---------------------------------------------------------------------------

# Skills the JD explicitly requires (hard skills)
REQUIRED_SKILLS = {
    "sentence transformers", "sentence-transformers", "embeddings", "word embeddings",
    "faiss", "pinecone", "weaviate", "qdrant", "milvus", "opensearch", "elasticsearch",
    "vector database", "vector search", "hybrid search", "dense retrieval",
    "bm25", "information retrieval", "retrieval", "ranking",
    "nlp", "natural language processing", "transformers", "bert", "roberta",
    "pytorch", "tensorflow", "hugging face transformers", "huggingface",
    "python", "machine learning", "deep learning",
    "ndcg", "mrr", "map", "evaluation", "a/b testing",
    "recommendation systems", "search",
    "lora", "qlora", "peft", "fine-tuning llms", "llm", "llms",
    "xgboost", "learning to rank",
}

# Nice-to-have skills
BONUS_SKILLS = {
    "mlflow", "weights & biases", "wandb", "ray", "triton",
    "onnx", "trt", "tensorrt", "model serving", "bentoml", "torchserve",
    "spark", "kafka", "distributed systems",
    "docker", "kubernetes", "aws", "gcp", "azure",
    "rust", "go", "c++",
    "open source", "github",
    "langchain", "llamaindex", "openai", "anthropic",
}

# Titles that signal strong ML/AI alignment
STRONG_TITLES = {
    "ml engineer", "machine learning engineer", "ai engineer", "senior ai engineer",
    "principal ml", "staff ml", "applied ml", "applied scientist",
    "data scientist", "senior data scientist", "nlp engineer",
    "research engineer", "ai research", "search engineer", "ranking engineer",
    "recommendation systems engineer", "recommendation engineer",
    "senior software engineer ml", "software engineer ml",
    "retrieval engineer", "information retrieval",
}

# Partial title keywords that provide partial credit
PARTIAL_TITLE_KEYWORDS = {
    "machine learning", "ml", "ai", "deep learning", "nlp", "data science",
    "recommendation", "search", "ranking", "retrieval", "applied scientist",
    "computer vision",  # partial credit — adjacent but not core
}

# Career title keywords that indicate production ML work in career history
CAREER_ML_KEYWORDS = {
    "ml engineer", "machine learning", "ai engineer", "data scientist",
    "nlp", "deep learning", "recommendation", "search engineer", "ranking",
    "applied scientist", "research engineer", "ai research",
    "retrieval", "information retrieval",
}

# Consulting / IT-services disqualifiers (pure services = hard penalty)
SERVICES_COMPANIES = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
    "mindtree", "l&t infotech", "ltimindtree", "ltts",
}

# Company size buckets that suggest product company (not services)
PRODUCT_COMPANY_SIZES = {"1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000"}

# Pure research environments — production deployment likely absent
RESEARCH_KEYWORDS = {"research lab", "iit", "iim", "university", "academia", "phd", "professor"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalise(s: str) -> str:
    return s.lower().strip()


def days_ago(d_str: str) -> int:
    """Return how many days ago a date string (YYYY-MM-DD) was from TODAY."""
    try:
        d = date.fromisoformat(d_str)
        return max(0, (TODAY - d).days)
    except Exception:
        return 365


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


# ---------------------------------------------------------------------------
# Component 1: Skill Match Score  (weight 0.35)
# ---------------------------------------------------------------------------

def build_jd_vectorizer(candidates_corpus: list[str]) -> TfidfVectorizer:
    """Fit TF-IDF on JD + candidate corpus for vocabulary coverage."""
    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
    )
    vec.fit([JD_CORPUS] + candidates_corpus)
    return vec


def candidate_skill_corpus(c: dict) -> str:
    """Build the text representation of a candidate for TF-IDF."""
    parts = []
    p = c["profile"]
    parts.append(p.get("headline", ""))
    parts.append(p.get("summary", ""))
    parts.append(p.get("current_title", ""))

    # Weight skills by proficiency — repeat advanced/expert skills
    prof_weight = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
    for s in c.get("skills", []):
        w = prof_weight.get(s.get("proficiency", "beginner"), 1)
        parts.extend([s["name"]] * w)

    # Add career descriptions (most recent 3 roles)
    for role in c.get("career_history", [])[:3]:
        parts.append(role.get("title", ""))
        parts.append(role.get("description", ""))

    # Add certifications and assessment skill names
    for cert in c.get("certifications", []):
        parts.append(cert.get("name", ""))
    for skill_name in c.get("redrob_signals", {}).get("skill_assessment_scores", {}).keys():
        parts.append(skill_name)

    return " ".join(parts)


def compute_skill_match_score(c: dict, jd_vec: np.ndarray, vectorizer: TfidfVectorizer) -> float:
    """
    TF-IDF cosine similarity PLUS structured keyword bonus.
    Pure cosine only rewards shared vocabulary; the structured bonus rewards
    having the specific skills the JD requires regardless of phrasing.
    """
    corpus = candidate_skill_corpus(c)
    cand_vec = vectorizer.transform([corpus])
    cosine = float(cosine_similarity(jd_vec, cand_vec)[0][0])

    # Structured keyword matching
    skill_names_lower = {normalise(s["name"]) for s in c.get("skills", [])}
    # Also extract from assessment scores
    skill_names_lower |= {normalise(k) for k in c.get("redrob_signals", {}).get("skill_assessment_scores", {}).keys()}

    required_hits = len(skill_names_lower & REQUIRED_SKILLS)
    bonus_hits = len(skill_names_lower & BONUS_SKILLS)

    # Structured bonus: diminishing returns
    structured_bonus = math.log1p(required_hits) * 0.08 + math.log1p(bonus_hits) * 0.02

    # Proficiency weighting for required skills — advanced/expert skills count more
    prof_weight = {"beginner": 0.5, "intermediate": 0.75, "advanced": 1.0, "expert": 1.2}
    prof_bonus = 0.0
    for s in c.get("skills", []):
        if normalise(s["name"]) in REQUIRED_SKILLS:
            w = prof_weight.get(s.get("proficiency", "beginner"), 0.5)
            prof_bonus += (w - 0.5) * 0.015

    raw = cosine + structured_bonus + prof_bonus
    return clamp(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Component 2: Career Trajectory Score  (weight 0.30)
# ---------------------------------------------------------------------------

def compute_career_score(c: dict) -> float:
    """
    Evaluates the quality of a candidate's career trajectory:
    - Are their titles in the ML/AI space?
    - Have they worked at product companies (not exclusively services)?
    - Do their role descriptions mention production deployment?
    - Title progression toward seniority?
    - Penalise pure consulting backgrounds explicitly.
    """
    history = c.get("career_history", [])
    if not history:
        return 0.0

    p = c["profile"]
    current_title_lower = normalise(p.get("current_title", ""))

    # --- Current title score ---
    current_title_score = 0.0
    if any(t in current_title_lower for t in STRONG_TITLES):
        current_title_score = 1.0
    elif any(k in current_title_lower for k in PARTIAL_TITLE_KEYWORDS):
        current_title_score = 0.55
    elif any(k in current_title_lower for k in {"software engineer", "backend engineer", "engineer", "developer"}):
        current_title_score = 0.3  # technical but not ML
    else:
        current_title_score = 0.05  # non-technical

    # --- Career ML depth ---
    ml_role_months = 0
    total_months = 0
    product_company_months = 0
    services_only_months = 0
    production_evidence = 0  # count of roles with production deployment keywords
    career_ml_title_count = 0

    production_keywords = {
        "production", "deployed", "shipped", "real users", "scale", "latency",
        "a/b test", "a/b testing", "serving", "inference", "recsys", "retrieval",
        "ranking system", "search system", "recommendation system",
        "embedding", "vector", "index", "pipeline at scale",
    }

    for role in history:
        dm = role.get("duration_months", 0)
        total_months += dm
        title_l = normalise(role.get("title", ""))
        company_l = normalise(role.get("company", ""))
        desc_l = normalise(role.get("description", ""))
        industry_l = normalise(role.get("industry", ""))
        size = role.get("company_size", "")

        is_services = any(s in company_l for s in SERVICES_COMPANIES)
        is_product_size = size in PRODUCT_COMPANY_SIZES

        if is_services:
            services_only_months += dm
        elif is_product_size:
            product_company_months += dm

        if any(k in title_l for k in CAREER_ML_KEYWORDS):
            ml_role_months += dm
            career_ml_title_count += 1

        # Check description for production deployment evidence
        desc_hits = sum(1 for kw in production_keywords if kw in desc_l)
        if desc_hits >= 2:
            production_evidence += 1

    if total_months == 0:
        return 0.0

    ml_fraction = ml_role_months / max(total_months, 1)
    product_fraction = product_company_months / max(total_months, 1)
    services_fraction = services_only_months / max(total_months, 1)

    # Build component scores
    # (a) Current title quality
    title_component = current_title_score * 0.35

    # (b) ML career depth — what fraction of career has been in ML roles
    ml_depth_component = clamp(ml_fraction * 1.5, 0.0, 1.0) * 0.30

    # (c) Product company experience — startup/product > consulting
    product_component = clamp(product_fraction, 0.0, 1.0) * 0.20

    # (d) Production deployment evidence in descriptions
    prod_evidence_component = clamp(production_evidence / max(len(history), 1), 0.0, 1.0) * 0.15

    raw = title_component + ml_depth_component + product_component + prod_evidence_component

    # Hard penalty for pure consulting/services background (no product company experience)
    if services_fraction > 0.85 and product_company_months < 12:
        raw *= 0.40  # hard penalty per JD explicitly warning against pure-services backgrounds

    # Penalty for non-technical current title regardless of skills
    if current_title_score < 0.2:
        raw *= 0.50  # keyword-stuffer trap: great skills list, non-ML title

    return clamp(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Component 3: Experience Fit Score  (weight 0.15)
# ---------------------------------------------------------------------------

def compute_experience_score(yoe: float) -> float:
    """
    Non-linear curve targeting the JD's explicit 5-9 year sweet spot.
    Peak at 7 years. Soft penalties for under/over-qualification.
    The JD says 'some people hit senior judgment at 4 years' — so we don't
    hard-cut below 4, but we do penalise heavily below 3 and above 13.
    """
    if yoe < 2:
        return 0.15
    elif yoe < 4:
        # Ramp up from junior — partial credit
        return 0.15 + (yoe - 2) / 2.0 * 0.45
    elif yoe <= 5:
        # Approaching sweet spot
        return 0.60 + (yoe - 4) * 0.20
    elif yoe <= 9:
        # Full sweet spot — peak at 7
        # Gaussian-like peak
        peak = 7.0
        sigma = 2.0
        return 0.80 + 0.20 * math.exp(-0.5 * ((yoe - peak) / sigma) ** 2)
    elif yoe <= 12:
        # Over-qualified soft penalty
        return 0.80 - (yoe - 9) * 0.08
    elif yoe <= 15:
        # Significant over-qualification
        return 0.56 - (yoe - 12) * 0.06
    else:
        # Very senior — likely wrong fit per JD
        return max(0.20, 0.38 - (yoe - 15) * 0.03)


# ---------------------------------------------------------------------------
# Component 4: Behavioral Availability Score  (weight 0.15)
# ---------------------------------------------------------------------------

def compute_availability_score(c: dict) -> float:
    """
    Composite of 6 Redrob behavioral signals that indicate whether a
    candidate is genuinely hireable right now, not just theoretically qualified.
    Key insight from JD: 'a perfect-on-paper candidate who hasn't logged in
    for 6 months and has a 5% response rate is not actually available.'
    """
    sig = c.get("redrob_signals", {})
    scores = []

    # (1) Recency — how recently were they active?
    last_active_ago = days_ago(sig.get("last_active_date", "2020-01-01"))
    if last_active_ago <= 14:
        recency = 1.0
    elif last_active_ago <= 30:
        recency = 0.90
    elif last_active_ago <= 60:
        recency = 0.75
    elif last_active_ago <= 90:
        recency = 0.55
    elif last_active_ago <= 180:
        recency = 0.30
    else:
        recency = 0.05
    scores.append(("recency", recency, 0.30))

    # (2) Open to work flag
    open_flag = 1.0 if sig.get("open_to_work_flag", False) else 0.35
    scores.append(("open_flag", open_flag, 0.20))

    # (3) Recruiter response rate
    resp_rate = float(sig.get("recruiter_response_rate", 0.0))
    scores.append(("resp_rate", clamp(resp_rate, 0.0, 1.0), 0.20))

    # (4) Notice period — JD explicitly says sub-30 preferred, can buy out up to 30
    notice = int(sig.get("notice_period_days", 90))
    if notice <= 15:
        notice_score = 1.0
    elif notice <= 30:
        notice_score = 0.90
    elif notice <= 60:
        notice_score = 0.65
    elif notice <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.25
    scores.append(("notice", notice_score, 0.15))

    # (5) Interview completion rate — will they actually show up?
    icr = float(sig.get("interview_completion_rate", 0.5))
    scores.append(("interview_rate", clamp(icr, 0.0, 1.0), 0.10))

    # (6) Profile completeness — proxy for engagement seriousness
    completeness = float(sig.get("profile_completeness_score", 50)) / 100.0
    scores.append(("completeness", clamp(completeness, 0.0, 1.0), 0.05))

    weighted = sum(s * w for _, s, w in scores)
    return clamp(weighted, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Component 5: Education & Extras Score  (weight 0.05)
# ---------------------------------------------------------------------------

def compute_education_score(c: dict) -> float:
    """
    Light signal: institution tier, field relevance, GitHub activity,
    and assessment scores. Intentionally low weight — the JD does not
    make education a strong filter.
    """
    edu = c.get("education", [])
    sig = c.get("redrob_signals", {})

    # Education tier
    tier_scores = {"tier_1": 1.0, "tier_2": 0.75, "tier_3": 0.55, "tier_4": 0.40, "unknown": 0.50}
    tier_score = max(
        (tier_scores.get(e.get("tier", "unknown"), 0.50) for e in edu),
        default=0.50,
    )

    # Field relevance
    relevant_fields = {
        "computer science", "cs", "information technology", "it",
        "electrical engineering", "electronics", "mathematics", "statistics",
        "data science", "ai", "machine learning", "computational",
    }
    field_score = 0.5
    for e in edu:
        fos = normalise(e.get("field_of_study", ""))
        if any(f in fos for f in relevant_fields):
            field_score = 1.0
            break
        elif "engineer" in fos or "science" in fos:
            field_score = max(field_score, 0.70)

    # GitHub activity
    gh = float(sig.get("github_activity_score", -1))
    if gh == -1:
        gh_score = 0.35  # no GitHub linked — mild negative signal
    else:
        gh_score = clamp(gh / 100.0, 0.0, 1.0)

    # Skill assessment scores (if any)
    assessments = sig.get("skill_assessment_scores", {})
    if assessments:
        avg_assessment = sum(assessments.values()) / len(assessments)
        assessment_score = clamp(avg_assessment / 100.0, 0.0, 1.0)
    else:
        assessment_score = 0.50

    raw = (tier_score * 0.25 + field_score * 0.25 + gh_score * 0.30 + assessment_score * 0.20)
    return clamp(raw, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Honeypot Detection
# ---------------------------------------------------------------------------

def is_honeypot(c: dict) -> bool:
    """
    Detect candidates with subtly impossible profiles.
    Rules:
    1. Total career months significantly exceeds stated years of experience.
    2. Future dates in career history.
    3. Current role end_date is not null.
    4. Multiple concurrent current roles.
    5. Years of experience ≫ oldest career start.
    """
    history = c.get("career_history", [])
    yoe = c["profile"].get("years_of_experience", 0)

    # Rule 1: career duration vs stated YoE
    total_months = sum(r.get("duration_months", 0) for r in history)
    if total_months > (yoe * 12 + 30):  # >2.5yr gap tolerance
        return True

    # Rule 2: multiple current roles
    current_count = sum(1 for r in history if r.get("is_current", False))
    if current_count > 1:
        return True

    # Rule 3: current role has an end_date
    for r in history:
        if r.get("is_current", False) and r.get("end_date") is not None:
            return True

    # Rule 4: start_date after today
    for r in history:
        sd = r.get("start_date", "")
        try:
            if date.fromisoformat(sd) > TODAY:
                return True
        except Exception:
            pass

    # Rule 5: YoE > 0 but no career history
    if yoe > 2 and len(history) == 0:
        return True

    return False


# ---------------------------------------------------------------------------
# Location Score helper (soft bonus, not a separate component)
# ---------------------------------------------------------------------------

def location_bonus(c: dict) -> float:
    """Small bonus for India-based candidates or willing-to-relocate."""
    sig = c.get("redrob_signals", {})
    country = normalise(c["profile"].get("country", ""))
    location = normalise(c["profile"].get("location", ""))
    relocate = sig.get("willing_to_relocate", False)

    india_cities = {"pune", "noida", "delhi", "ncr", "bangalore", "bengaluru",
                    "hyderabad", "mumbai", "gurgaon", "gurugram", "chennai"}

    if country == "india":
        if any(city in location for city in india_cities):
            return 0.04  # preferred cities
        return 0.02  # India but other city
    elif relocate:
        return 0.01
    return 0.0


# ---------------------------------------------------------------------------
# Reasoning Generator
# ---------------------------------------------------------------------------

def generate_reasoning(c: dict, scores: dict, rank: int) -> str:
    """
    Generate specific, fact-grounded reasoning per Stage 4 rubric:
    - Reference specific facts from the profile
    - Connect to JD requirements
    - Acknowledge honest concerns where they exist
    - Match rank tone (rank 1-10 glowing, rank 80-100 measured)
    """
    p = c["profile"]
    sig = c.get("redrob_signals", {})
    history = c.get("career_history", [])
    skills = c.get("skills", [])

    yoe = p.get("years_of_experience", 0)
    title = p.get("current_title", "")
    company = p.get("current_company", "")
    country = p.get("country", "")
    location = p.get("location", "")

    # Gather key facts
    top_skills = [s["name"] for s in sorted(skills, key=lambda x: x.get("endorsements", 0), reverse=True)[:4]]
    skill_str = ", ".join(top_skills) if top_skills else "general engineering skills"

    # Active status
    last_active_ago = days_ago(sig.get("last_active_date", "2020-01-01"))
    open_flag = sig.get("open_to_work_flag", False)
    notice = sig.get("notice_period_days", 90)
    resp_rate = sig.get("recruiter_response_rate", 0.0)
    gh_score = sig.get("github_activity_score", -1)

    # Career company context
    current_company_size = p.get("current_company_size", "")
    current_industry = p.get("current_industry", "")

    concerns = []
    positives = []

    # Experience fit
    if 5 <= yoe <= 9:
        positives.append(f"{yoe:.1f} years experience in the JD's target band")
    elif yoe < 5:
        concerns.append(f"{yoe:.1f} years experience is below the 5-9yr target")
    else:
        concerns.append(f"{yoe:.1f} years may be over-qualified for the founding-team dynamic")

    # Location
    if country == "India":
        positives.append(f"{location}-based (India preferred)")
    elif sig.get("willing_to_relocate", False):
        positives.append(f"willing to relocate from {country}")
    else:
        concerns.append(f"based in {country}, not willing to relocate")

    # Notice period
    if notice <= 30:
        positives.append(f"notice period {notice}d (within buyout window)")
    elif notice > 60:
        concerns.append(f"notice period {notice}d (above 30-day preference)")

    # Responsiveness
    if resp_rate >= 0.7:
        positives.append(f"strong recruiter responsiveness ({resp_rate:.0%})")
    elif resp_rate < 0.3:
        concerns.append(f"low recruiter response rate ({resp_rate:.0%})")

    # Recency
    if last_active_ago <= 14:
        positives.append("active within the last 2 weeks")
    elif last_active_ago > 120:
        concerns.append(f"last active {last_active_ago} days ago")

    # GitHub
    if gh_score >= 60:
        positives.append(f"GitHub activity score {gh_score:.0f}/100")

    # Services penalty
    company_lower = normalise(company)
    if any(s in company_lower for s in SERVICES_COMPANIES):
        concerns.append(f"currently at {company} (IT-services context)")

    # Build reasoning string
    # Lead with the most important fact: title + YoE + company context
    lead = f"{yoe:.1f}yr {title} at {company} ({current_industry})"
    skills_note = f"top skills: {skill_str}"

    if rank <= 20:
        pos_str = "; ".join(positives[:2]) if positives else ""
        con_str = f"Minor concern: {concerns[0]}" if concerns else ""
        parts = [p for p in [lead, skills_note, pos_str, con_str] if p]
        return ". ".join(parts) + "."

    elif rank <= 60:
        pos_str = positives[0] if positives else ""
        con_str = f"Concern: {concerns[0]}" if concerns else ""
        parts = [p for p in [lead, skills_note, pos_str, con_str] if p]
        return ". ".join(parts) + "."

    else:
        con_str = "; ".join(concerns[:2]) if concerns else "limited ML alignment"
        return f"{lead}. {skills_note}. Below cutoff due to: {con_str}."


# ---------------------------------------------------------------------------
# Master scoring function
# ---------------------------------------------------------------------------

WEIGHTS = {
    "skill": 0.35,
    "career": 0.30,
    "experience": 0.15,
    "availability": 0.15,
    "education": 0.05,
}


def score_candidate(c: dict, jd_vec: np.ndarray, vectorizer: TfidfVectorizer) -> dict:
    """Compute all component scores and return a scoring dict."""
    yoe = c["profile"].get("years_of_experience", 0)

    skill_score = compute_skill_match_score(c, jd_vec, vectorizer)
    career_score = compute_career_score(c)
    exp_score = compute_experience_score(yoe)
    avail_score = compute_availability_score(c)
    edu_score = compute_education_score(c)

    weighted = (
        skill_score * WEIGHTS["skill"]
        + career_score * WEIGHTS["career"]
        + exp_score * WEIGHTS["experience"]
        + avail_score * WEIGHTS["availability"]
        + edu_score * WEIGHTS["education"]
    )

    # Location soft bonus (not part of main weights, small top-up)
    loc_bonus = location_bonus(c)

    final = clamp(weighted + loc_bonus, 0.0, 1.0)

    return {
        "candidate_id": c["candidate_id"],
        "final_score": final,
        "skill": skill_score,
        "career": career_score,
        "experience": exp_score,
        "availability": avail_score,
        "education": edu_score,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> list[dict]:
    print(f"Loading candidates from {path}...")
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates):,} candidates.")
    return candidates


def run_ranking(candidates: list[dict]) -> list[dict]:
    """Full pipeline: preprocess → vectorise → score → filter → rank."""

    print("Step 1: Building candidate text corpora...")
    corpora = [candidate_skill_corpus(c) for c in candidates]

    print("Step 2: Fitting TF-IDF vectorizer...")
    vectorizer = build_jd_vectorizer(corpora)
    jd_vec = vectorizer.transform([JD_CORPUS])

    print("Step 3: Pre-computing candidate vectors (batch)...")
    candidate_vecs = vectorizer.transform(corpora)  # sparse matrix, all at once

    print("Step 4: Computing cosine similarities in batch...")
    cosine_scores = cosine_similarity(jd_vec, candidate_vecs).flatten()  # shape (N,)

    print("Step 5: Scoring all candidates...")
    results = []
    honeypot_count = 0

    for i, c in enumerate(candidates):
        if i % 10000 == 0:
            print(f"  ... {i:,}/{len(candidates):,}")

        # Honeypot check
        if is_honeypot(c):
            honeypot_count += 1
            continue

        yoe = c["profile"].get("years_of_experience", 0)

        # Hard disqualifier: pure consulting background with zero ML alignment
        career_score = compute_career_score(c)
        if career_score < 0.05:
            continue  # skip completely non-technical candidates to save compute

        # Use pre-computed cosine score
        raw_cosine = float(cosine_scores[i])

        # Structured skill bonus (fast path using pre-computed tfidf)
        skill_names_lower = {normalise(s["name"]) for s in c.get("skills", [])}
        skill_names_lower |= {normalise(k) for k in c.get("redrob_signals", {}).get("skill_assessment_scores", {}).keys()}
        required_hits = len(skill_names_lower & REQUIRED_SKILLS)
        bonus_hits = len(skill_names_lower & BONUS_SKILLS)
        structured_bonus = math.log1p(required_hits) * 0.08 + math.log1p(bonus_hits) * 0.02

        prof_weight_map = {"beginner": 0.5, "intermediate": 0.75, "advanced": 1.0, "expert": 1.2}
        prof_bonus = sum(
            (prof_weight_map.get(s.get("proficiency", "beginner"), 0.5) - 0.5) * 0.015
            for s in c.get("skills", [])
            if normalise(s["name"]) in REQUIRED_SKILLS
        )

        skill_score = clamp(raw_cosine + structured_bonus + prof_bonus, 0.0, 1.0)

        exp_score = compute_experience_score(yoe)
        avail_score = compute_availability_score(c)
        edu_score = compute_education_score(c)

        weighted = (
            skill_score * WEIGHTS["skill"]
            + career_score * WEIGHTS["career"]
            + exp_score * WEIGHTS["experience"]
            + avail_score * WEIGHTS["availability"]
            + edu_score * WEIGHTS["education"]
        )
        final = clamp(weighted + location_bonus(c), 0.0, 1.0)

        results.append({
            "candidate_id": c["candidate_id"],
            "final_score": final,
            "skill": skill_score,
            "career": career_score,
            "experience": exp_score,
            "availability": avail_score,
            "education": edu_score,
            "_candidate": c,
        })

    print(f"  Honeypots detected and excluded: {honeypot_count}")
    print(f"  Scoreable candidates: {len(results):,}")

    # Sort by score descending, then by candidate_id ascending for tie-breaking
    results.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))

    return results


def write_submission(ranked: list[dict], output_path: str, team_id: str = "PARTICIPANT_001"):
    """Write the final submission CSV."""
    top100 = ranked[:100]

    # Normalise scores to be strictly non-increasing
    # (handle floating point noise)
    for i in range(1, len(top100)):
        if top100[i]["final_score"] > top100[i - 1]["final_score"]:
            top100[i]["final_score"] = top100[i - 1]["final_score"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])

        for rank, entry in enumerate(top100, start=1):
            c = entry["_candidate"]
            scores = {k: entry[k] for k in ["skill", "career", "experience", "availability", "education"]}
            reasoning = generate_reasoning(c, scores, rank)

            writer.writerow([
                entry["candidate_id"],
                rank,
                f"{entry['final_score']:.6f}",
                reasoning,
            ])

    print(f"\nSubmission written to: {output_path}")
    print(f"Top 5 candidates:")
    for i, entry in enumerate(top100[:5], 1):
        p = entry["_candidate"]["profile"]
        print(f"  #{i}: {entry['candidate_id']} | {p['current_title']} | "
              f"{p['years_of_experience']}yr | score={entry['final_score']:.4f} "
              f"(skill={entry['skill']:.3f} career={entry['career']:.3f} "
              f"exp={entry['experience']:.3f} avail={entry['availability']:.3f})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Redrob Hybrid Candidate Ranker")
    parser.add_argument("--candidates", default="./candidates.jsonl",
                        help="Path to candidates JSONL file")
    parser.add_argument("--out", default="./submission.csv",
                        help="Output CSV path")
    parser.add_argument("--team-id", default="PARTICIPANT_001",
                        help="Team ID for filename")
    args = parser.parse_args()

    import time
    t0 = time.time()

    candidates = load_candidates(args.candidates)
    ranked = run_ranking(candidates)
    write_submission(ranked, args.out, args.team_id)

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("WARNING: Exceeded 5-minute constraint!")
    else:
        print(f"Well within the 5-minute ({300}s) constraint.")


if __name__ == "__main__":
    main()
