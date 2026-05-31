# Baseline Report for Updated Step 1 Requirements

## 1) Verified Working Baseline System
- Baseline implementation: `Step_1_Baseline_and_Failure_Analysis.ipynb`
- End-to-end pipeline verified:
  - Retrieval: BM25, Dense, GraphRAG
  - Orchestration: Confidence, Waterfall, Voting
  - Re-ranking: enabled in Hybrid/ReRank variants
  - Answer synthesis: extractive synthesis in orchestration pipeline

## 2) Baseline Performance (Reference Tables)

### Retrieval Quality (Full Corpus, 24 queries)
| Method | MRR | Precision@1 | Precision@3 | Precision@5 | Precision@10 |
|---|---:|---:|---:|---:|---:|
| GraphRAG | **0.232573** | 0.083333 | 0.097222 | 0.116667 | 0.116667 |
| ReRank | 0.222952 | 0.041667 | 0.138889 | 0.125000 | 0.100000 |
| Hybrid | 0.202154 | 0.000000 | 0.097222 | 0.125000 | 0.095833 |
| Dense | 0.165525 | 0.041667 | 0.069444 | 0.058333 | 0.066667 |
| BM25 | 0.151296 | 0.041667 | 0.055556 | 0.091667 | 0.091667 |

### Orchestration Quality (Full Corpus, 24 queries)
| Strategy | MRR | Precision@1 | Precision@3 | Precision@5 | Precision@10 |
|---|---:|---:|---:|---:|---:|
| Confidence | **0.208827** | 0.000000 | 0.111111 | 0.133333 | 0.100000 |
| Waterfall | 0.208303 | 0.041667 | 0.138889 | 0.100000 | 0.070833 |
| Voting | 0.202154 | 0.000000 | 0.097222 | 0.125000 | 0.095833 |

### System Efficiency and Cost Proxy
(Computed from `2026-05-29_22-44-32_output_step3.csv`)
- Queries: 24
- Mean runtime/query: **0.337 s**
- P95 runtime/query: **0.508 s**
- Decision distribution: `answer=12`, `abstain=9`, `clarify=3`
- Approximate cost indicator: low (heuristic reliability pipeline and retrieval-first flow in this benchmark run).

## 3) Reference Baseline Configuration
- **Reference baseline for the rest of the project:**
  - Full corpus setting
  - Confidence orchestration strategy
  - MRR = **0.208827**
- Rationale: strongest or tied-best orchestration MRR with stable behavior and built-in adaptive routing.

## 4) Failure Taxonomy (Structured)
(From `step1_updated_deliverables.md`)
- retrieval failure: 5
- ranking failure: 4
- synthesis failure: 0
- grounding failure: 0
- ambiguity failure: 3
- contradiction failure: 9
- orchestration failure: 9
- overconfidence failure: 0

Representative examples:
- retrieval failure: "what did prof. schubert say about flying?"
- ranking failure: "who at eth received erc grants?"
- ambiguity failure: "what is e-sling?"
- contradiction failure: "when did the insight get to mars?"
- orchestration failure: "how do alpine plants respond to climate change?"

## 5) Motivation for Proposed Extensions
Weaknesses to address next:
- Contradiction + orchestration failures: improve strategy switching logic and contradiction-aware retrieval routing.
- Retrieval/ranking failures: improve candidate quality and reranker signals before synthesis.
- Ambiguity failures: strengthen clarify-first policy for underspecified questions.

These directly motivate the reliability/adaptation components introduced in Step 2/3 (clarification, contradiction checks, trust scoring, abstention, recovery).
