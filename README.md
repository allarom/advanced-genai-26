# Reliable Adaptive Agentic RAG System

**Course:** Advanced Generative AI Capstone  
**Institution:** HSLU Applied Information and Data Science

Dongyuan Gao, Julia Studer, Alla Roman

---

## What this project does

This repository implements a **multi-agent Retrieval-Augmented Generation (RAG) system** with an added **reliability layer**. The system does not just retrieve and answer -- it checks whether the answer is trustworthy before returning it.

```
User Query --> Clarify? --> Retrieve --> Check Signals --> Trust Score --> [Recover if low] --> Answer / Abstain
```

---

## Open in Google Colab

| Step | Notebook |
|------|----------|
| Step 1 — Baseline and Failure Analysis | [Open in Colab](https://colab.research.google.com/github/allarom/advanced-genai-26/blob/main/Step_1_Baseline_and_Failure_Analysis.ipynb) |
| Step 2 — Reliability-Aware Design | [Open in Colab](https://colab.research.google.com/github/allarom/advanced-genai-26/blob/main/Step_2_Reliability_Aware_Design.ipynb) |
| Step 3 — Reliable Adaptive Agentic RAG | [Open in Colab](https://colab.research.google.com/github/allarom/advanced-genai-26/blob/main/Step_3_Reliable_Adaptive_Agentic_RAG.ipynb) |
| Step 4 — Evaluation | [Open in Colab](https://colab.research.google.com/github/allarom/advanced-genai-26/blob/main/Step_4_Evaluation.ipynb) |

---

## Repository Structure

| Path | Purpose |
|------|---------|
| `Step_1_Baseline_and_Failure_Analysis.ipynb` | Baseline reproduction, full-corpus retrieval evaluation, answer synthesis, and failure analysis |
| `legacy_retrieval_engine.py` | Clean reusable retrieval engine used by Step 3; exposes Confidence, Waterfall, and Voting orchestration |
| `Step_2_Reliability_Aware_Design.ipynb` | Reliability-aware design: architecture, signals, decision policy, and trace schema |
| `Step_3_Reliable_Adaptive_Agentic_RAG.ipynb` | Implementation of reliability agents and `ReliableAdaptiveRAG` |
| `Step_4_Evaluation.ipynb` | Evaluation framework: baseline, cold/warm memory, HITL feedback, ablation, reliability metrics, and challenge queries |
| `Step_4_1_extra_challenges.ipynb` | Bonus memory-based adaptation and human-in-the-loop feedback interface |
| `report.md` | Source markdown for the final project report |
| `baseline_repro_report.md` | Source markdown for the baseline reproduction report |
| `reports/` | Generated report PDFs, including `report.pdf` and `baseline_repro_report.pdf` |
| `scripts/build_reports.R` | R Markdown build script that renders report PDFs into `reports/` |
| `scripts/` | Supporting utilities for notebook/report maintenance |
| `memory/` | Persisted learned weights, verified answers, and evaluation CSV outputs |
| `baseline/` | Reference materials and inherited baseline work from the previous project |
| `archived_documents/` | Archived supporting documents, screenshots, and audit notes from the project process |

---

## Quick Start

### 1. Setup
If you want to run it locally

```bash
# Install dependencies
pip install pandas numpy transformers accelerate
# Note: pytrec_eval is only needed for baseline TREC evaluation
```

### 2. Run baseline (Step 1 / Step 2 legacy)

Open `Step_1_Baseline_and_Failure_Analysis.ipynb` in Google Colab or Jupyter.
- Mount Google Drive with `benchmark/` and `storage/` data.
- Set `HF_TOKEN` in Colab secrets (optional, for Hugging Face downloads).
- Run all cells to reproduce baseline retrieval, orchestration, answer synthesis, and failure analysis results.

### 3. Run reliability-augmented system (Step 3)

Open `Step_3_Reliable_Adaptive_Agentic_RAG.ipynb`.
- The setup cells make `legacy_retrieval_engine.py` available.
- The retrieval wrappers are imported from `legacy_retrieval_engine.py`;
- Run agent definition cells (5.1--5.8).
- Run the main `ReliableAdaptiveRAG` class cell.
- Test with:

```python
rag = ReliableAdaptiveRAG()
result = rag.run("Who was president of ETH in 2003?", strategy="confidence")
print(result["decision"])   # "answer", "abstain", or "clarify"
print(result["reason"])     # human-readable explanation
print(result["trace_log"])  # step-by-step log
```

### 4. Ablation testing

Disable individual agents to measure their contribution:

```python
rag = ReliableAdaptiveRAG(ablate=["contradiction", "critic"])
result = rag.run("Did ETH funding increase?", strategy="confidence")
```

### 5. Run memory-augmented system with human feedback (Step 4.1)

Open `Step_4_1_extra_challenges.ipynb`.
- Cell 1 loads Step 3 via `%run Step_3_Reliable_Adaptive_Agentic_RAG.ipynb`.
- Run the `MemoryStore`, `classify_query_type`, and `MemoryAugmentedRAG` cells.
- Use the feedback UI to rate answers; memory learns per query type.

```python
# After running the Step 4.1 notebook cells, feedback_ui is available:
feedback_ui("Who received ERC grants at ETH?")
# Click Good / Bad / Fix -- feedback is saved to memory/step4_memory.json
```

- **Evaluation:** `Step_4_Evaluation.ipynb` runs a 9-phase framework (baseline → cold → HITL feedback → warm re-eval → comparison → ablation → reliability metrics → challenge queries → qualitative analysis). Results are saved to `memory/csv_outputs/`.

### 6. Build reports

Generated report PDFs are written to `reports/`:

```bash
Rscript scripts/build_reports.R
```

---

## Architecture Overview

The system wraps a proven retrieval engine with 8 reliability agents, plus memory and human feedback in Step 4.1:

| Layer | Agents | Role |
|-------|--------|------|
| **Pre-retrieval** | `ClarificationAgent` | Detect ambiguous queries |
| **Retrieval** | `ConfidenceOrchestrator` (legacy) | Retrieve, fuse, re-rank, synthesize |
| **Signals** | `EvidenceSufficiencyAgent`, `GroundednessAgent`, `ContradictionAgent` | Measure evidence quality |
| **Scoring** | `TrustAgent` | Combine signals into [0,1] score |
| **Decision** | `AbstentionAgent`, `RecoveryAgent` | Decide: abstain, retry, or answer |
| **Feedback** | `CriticAgent` | Human-readable quality notes |
| **Memory** | `MemoryStore` | Verified-answer cache + learned weights per query type |
| **HITL** | `feedback_ui()` | Human feedback buttons (Good/Bad/Fix) |

---

## Key Results

| Metric | Value |
|--------|-------|
| Best baseline (full corpus) | GraphRAG MRR = 0.233 |
| Best orchestration (full corpus) | Confidence MRR = 0.209 |
| Trust threshold | 0.4 (abstain if below) |
| Recovery actions | switch_strategy, rewrite_query, none |
| Memory strategies learned | confidence, waterfall, voting per query_type |
| Weight learning | confidence-only, clamped [0.3, 1.6] |

---

## Design Principles

1. **Reuse, do not rewrite** -- Legacy retrieval pipeline is loaded as a library.
2. **One agent, one job** -- Each reliability check is a separate class.
3. **Everything is traceable** -- Every run returns a unified trace dict.
4. **Fail safely** -- When in doubt, abstain or ask for clarification.
5. **Ablatable** -- Any agent can be disabled for debugging and evaluation.

---

## Known Limitations

- All reliability checks use lightweight heuristics (token overlap, keyword matching).
- Answer generation is extractive, not generative.
- Trust score weights (`0.6*sufficiency + 0.3*groundedness - 0.4*contradiction`) are hand-tuned, not learned.
- Memory weight learning is confidence-only; waterfall/voting are not weight-tunable.
- Weight credit assignment is coarse (all retrievers nudged equally on feedback).
- See `report.md` section 7 for full upgrade paths.
