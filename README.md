# Redrob Hackathon — Hybrid Candidate Ranking Engine

**Team:** Neural Network 
**Challenge:** Intelligent Candidate Discovery & Ranking  
**Runtime:** ~81s for 100K candidates on CPU | **Status:** ✅ Passes `validate_submission.py`

---

## The Core Insight

Our engine is designed around one principle: **the gap between what a JD says and what it means**. A candidate whose profile says "RAG, Pinecone, LangChain" but whose title is "Marketing Manager" is not a fit. A candidate who "built a candidate-JD retrieval pipeline at a product company" without using those exact words is. We engineered both of these distinctions explicitly.

---

## Architecture: 5-Component Weighted Scoring

```
Final Score = 0.35 × Skill Match
            + 0.30 × Career Trajectory
            + 0.15 × Experience Fit
            + 0.15 × Behavioral Availability
            + 0.05 × Education & Extras
            + location soft bonus (≤0.04)
```

### Component 1 — Skill Match Score (35%)

**Method:** TF-IDF cosine similarity + structured keyword bonus

We fit a bigram TF-IDF vectorizer over the full 100K candidate corpus plus the JD. This gives every term its document-frequency-adjusted weight. The cosine similarity between the JD vector and each candidate's "rich corpus" (headline + summary + current title + skills × proficiency weight + career descriptions + assessment scores) is computed in batch using a sparse matrix operation.

**Why not sentence-transformers?** We evaluated `all-MiniLM-L6-v2` but it is unavailable offline. More importantly, TF-IDF cosine over a bigram vocabulary captures the domain-specific co-occurrences (e.g., "vector search", "retrieval ranking", "hybrid dense") that are precisely what this JD's core requirements look like. For this narrow domain, it is competitive with dense embeddings.

**Structured keyword bonus:** A second pass against two curated vocabularies (38 required skills, 20 bonus skills) adds a log-scaled bonus for exact skill hits, weighted by proficiency level. This handles candidates who use plain-language descriptions rather than buzzwords.

### Component 2 — Career Trajectory Score (30%)

This is the anti-keyword-stuffer component. It answers: *has this person actually worked in ML/AI roles at product companies, and does their career history show production deployment?*

Four sub-signals:
- **Current title alignment** (35% of component): Strong titles (ML Engineer, AI Engineer, Applied Scientist, etc.) score 1.0. Non-technical titles score 0.05 regardless of skills. This directly penalises the "Marketing Manager with ML skills" trap.
- **ML career depth** (30%): What fraction of total career months were spent in ML-adjacent roles?
- **Product company experience** (20%): Months at product companies (size ≤ 5000) vs IT-services companies. Companies like TCS, Infosys, Wipro, Cognizant, Accenture, Capgemini receive a hard penalty if they represent >85% of the candidate's career with <12 months of product company experience.
- **Production deployment evidence** (15%): Career descriptions are scanned for production deployment signals ("shipped", "deployed", "real users", "at scale", "serving", "A/B test"). Roles with 2+ hits count toward production evidence.

### Component 3 — Experience Fit Score (15%)

Non-linear curve targeting the JD's explicit 5-9 year sweet spot:

| YoE | Score |
|-----|-------|
| <2  | 0.15  |
| 4   | 0.60  |
| 5-9 | 0.80-1.00 (peak at 7yr) |
| 12  | 0.56  |
| 15+ | ≤0.38 |

The JD says "some people hit senior judgment at 4 years" — so we do not hard-cut below 4, but we penalise. The over-qualification penalty is intentional: a 15-year Google veteran is explicitly not who this founding-team role is looking for.

### Component 4 — Behavioral Availability Score (15%)

The JD explicitly states: *"a perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."* We took this seriously.

Six signals, weighted:
| Signal | Weight | Logic |
|--------|--------|-------|
| Last active date | 30% | ≤14 days = 1.0; >180 days = 0.05 |
| Open to work flag | 20% | True = 1.0; False = 0.35 |
| Recruiter response rate | 20% | Direct 0-1 pass-through |
| Notice period | 15% | ≤30 days = 0.90; >90 days = 0.25 |
| Interview completion rate | 10% | Direct 0-1 pass-through |
| Profile completeness | 5% | Proxy for engagement seriousness |

### Component 5 — Education & Extras (5%)

Institution tier, field of study relevance (CS/EE/Math/Stats = 1.0), GitHub activity score (0-100 pass-through, penalises unlinked accounts), and average Redrob skill assessment scores.

---

## Honeypot Detection

22 candidates were flagged and excluded before scoring. Detection rules:

1. Total career months > (stated YoE × 12) + 30 months tolerance
2. Multiple roles marked `is_current: true`
3. Current role has a non-null `end_date`
4. Career `start_date` in the future
5. YoE > 2 with empty career history

This keeps our honeypot rate in the top 100 at approximately 0%, well below the 10% disqualification threshold.

---

## Key Findings: What Semantic + Career Scoring Discovers

**Hidden talent that keyword matching misses:**  
Candidate `CAND_0018499` (rank #3) is a *Senior Machine Learning Engineer at Zomato* with 7.2 years experience. Their skills list (`scikit-learn`, `Recommendation Systems`, `Learning to Rank`) does not contain buzzwords like "RAG" or "LangChain". A pure keyword-match system ranks them below candidates who list "LangChain tutorials" as a skill. Our career trajectory scoring correctly identifies that Zomato is a high-scale product company and "Learning to Rank" directly maps to what this JD requires.

**Keyword stuffers correctly penalised:**  
Several candidates in the pool have skills lists containing 10+ required ML terms (FAISS, Qdrant, Sentence Transformers, LLMs) but their current title is "Operations Manager" or "Business Analyst" and their career history shows exclusively consulting work. Our career component scores them ≤0.20, dropping them out of the top 100 regardless of skill match.

**Behaviorally inactive high-scorers down-ranked:**  
Multiple candidates with strong skill and career scores were pushed down or out of the top 100 by the availability component. A candidate who scores 0.85 on skill+career fit but hasn't logged in for 200 days and has a 5% recruiter response rate ends up at rank 70+.

**Top 10 profile:**  
Every candidate in our top 10 has: an ML/AI title, 5-9 years experience, India-based location, last-active within the past 3 months, and at least 3 required skills with advanced/expert proficiency. This is exactly the "narrow profile" the JD says it expects.

---

## Reasoning Quality

Every row in the submission includes a grounded, candidate-specific reasoning string that:
- Cites exact YoE, title, and employer
- Lists top 4 skills by endorsement count
- States the experience band verdict
- Notes the location
- Explicitly calls out concerns (notice period, low response rate, international location, services background) where they exist
- Tones to rank (rank 1-20 positive with minor concerns; rank 80-100 explicitly notes why the candidate is at the bottom)

---

## Reproduce

```bash
python rank.py --candidates ./candidates.jsonl --out ./PARTICIPANT_Neural_Network.csv
```

**Runtime:** ~80s on CPU (16 GB RAM). No GPU. No network. No external APIs.

**Dependencies:**
```
numpy
scikit-learn
```

---

## File Structure

```
├── rank.py                              # Main ranker (single file, self-contained)
├── PARTICIPANT_Neural_Network.csv       # Submission (100 rows, validated)
├── README.md                            # This file
└── submission_metadata.yaml             # Hackathon metadata
```
