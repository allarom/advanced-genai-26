# What We Built Today — Step 4 Evaluation

## 1. The Two Notebooks

```
Step_4_1_extra_challenges.ipynb          Step_4_Evaluation.ipynb
├─ MemoryStore (M1 cache)                ├─ Setup (mount Drive, clone repo)
├─ MemoryAugmentedRAG (wraps rag_system) ├─ Load (%run Step_4_1) ← imports everything
├─ HITL feedback demo                     ├─ Helpers (gold_map, run_one, run_batch)
├─ Strategy learning (M2/M2.5)          ├─ Part 1: Step 3 core evaluation
└─ Creates memory/*.json, memory/*.csv   ├─ Part 2: Extra Challenge evaluation
                                         ├─ Part 3: Qualitative + tradeoffs
                                         └─ 33 cells total
```

**Relationship:** Step 4 Evaluation calls `%run Step_4_1` in Cell 3 to import `rag_system`, `MemoryStore`, `MemoryAugmentedRAG`, etc. You only upload Step 4 Evaluation to Colab — Step 4.1 runs automatically.

---

## 2. What Step 4.1 Does

```
┌─────────────────────────────────────────────┐
│  Step 4.1 — Extra Challenges                │
│  (memory adaptation + simulated HITL)       │
├─────────────────────────────────────────────┤
│  M1  → Verified-answer cache                │
│  M2  → Per-query-type strategy learning     │
│  M2.5→ Per-type weight memory             │
│  HITL→ Human feedback simulation          │
└─────────────────────────────────────────────┘
     ↓
  Creates in memory/ folder:
    feedback_qa_matched.csv   (10 questions, matched to benchmark types)
    feedback_qa_random.csv    (10 questions, random)
    step4_memory.json         (warm memory snapshot)
    memory_matched.json       (after feedback loop)
    memory_random.json        (after feedback loop)
```

---

## 3. What Step 4 Evaluation Does

### Structure (33 cells, 3 parts)

```
Cell 1  Intro                "What this notebook evaluates"
Cell 2  Setup                Mount Drive, clone repo, install packages
Cell 3  Load                 %run Step_4.1 → imports rag_system, eval_qa_data, etc.
Cell 4  Helpers              gold_map, run_one, run_batch, answer_matches_gold
Cell 5-6  Context            Step 1→3 lineage table (MRR scores)

┌──────────── Part 1: Step 3 Reliability Core ────────────┐
│ Cell 7   Divider         "Step 3 Reliability System"     │
│ Cell 8   Mechanism checklist (A-H: 7/8 implemented)       │
│ Cell 9-10  Baseline      Run 24 benchmark queries         │
│ Cell 11-12 Agent ablation  Full / No-Contra / No-Recovery │
│ Cell 13  Documentation   CHALLENGE set explanation      │
│ Cell 14-15 Reliability   8 metrics + CHALLENGE + conf   │
└───────────────────────────────────────────────────────────┘

┌─────── Part 2: Extra Challenge (Memory + HITL) ─────────┐
│ Cell 16  Divider         "Extra Challenge"                │
│ Cell 17-18  Cold memory  Empty MemoryStore                │
│ Cell 19-20  Load feedback  CSVs + no-overlap check      │
│ Cell 21-22  Feedback loop  Simulate HITL on 20 queries  │
│ Cell 23-24  Warm re-eval   Benchmark on warm snapshots    │
│ Cell 25-26  Comparison     IR + reliability tables        │
│ Cell 27-28  Memory ablation  Disable cache / fix strategy │
└───────────────────────────────────────────────────────────┘

┌─────── Part 3: Integration & Tradeoffs ─────────────────┐
│ Cell 29  Divider         "Project-Wide Comparison"      │
│ Cell 30-31 Qualitative   5 required examples            │
│ Cell 32-33 Tradeoffs     MRR vs reliability vs latency  │
└───────────────────────────────────────────────────────────┘
```

---

## 4. Key Concepts We Implemented

### 4.1 Agent-Level Ablation (Cell 11-12)

**What:** Selectively disable Step 3 agents to measure their impact.

```
Full system          → rag_system.run(query, ablate=[])
No Contradiction     → rag_system.run(query, ablate=["contradiction"])
No Recovery          → rag_system.run(query, ablate=["recovery"])
```

**Why:** If removing an agent makes the system worse, that agent was actually useful. If removing it does nothing, it was dead weight.

**How:** `run_one` accepts `ablate` parameter → passes to `system.run(query, ablate=...)`. Each condition produces its own CSV.

---

### 4.2 CHALLENGE Set (Cell 14-15)

**What:** 12 hand-crafted queries to test reliability, not IR performance.

```
CHALLENGE = [
    {"cat": "ambiguous",    "exp": "clarify",  "query": "What is it?"},
    {"cat": "insufficient", "exp": "abstain",  "query": "When did ETH start accepting students?"},
    {"cat": "conflicting",  "exp": "abstain",  "query": "What is the current ETH budget?"},
    {"cat": "adversarial",  "exp": "abstain",  "query": "What did Elon Musk say about ETH Zurich?"},
    {"cat": "standard",     "exp": "answer",   "query": "How does ETH support innovation?"},
    # ... 12 total, 5 categories
]
```

**Why the requirement wants this:** To test if the system knows *when not to answer*.

**Metrics we compute from it:**
- **Correct abstention** — abstains on queries that should be abstained
- **False abstention** — abstains on queries that should be answered
- **Clarification usefulness** — does ambiguous trigger "clarify"?

---

### 4.3 Confidence-Correctness Alignment (Cell 14-15)

**What:** Does high trust score = actually correct answer?

```
Trust score bin → % of answers that are correct (token-overlap ≥ 0.3)

0-0.2   → 10% correct   (low trust = mostly wrong)
0.2-0.4 → 30% correct
0.4-0.7 → 60% correct
0.7-1.0 → 90% correct   (high trust = mostly right)
```

**Why:** A reliable system must be *calibrated* — it shouldn't be confident when wrong.

---

### 4.4 Resume-Safe Phases

**What:** Each phase checks if its CSV already exists. If yes → skip computation.

```python
df = load_version("baseline_step3")   # checks if CSV exists
if df is None:
    df = run_batch(...)              # only runs if no CSV
    save_version(df, "baseline_step3")
```

**Why:** A full run is ~20 min. If Colab disconnects or you fix a bug, you don't restart from zero.

---

## 5. What We Changed Today

| Change | Where | Why |
|--------|-------|-----|
| **3-part structure** | `build_step4_eval_notebook.py` | Requirement says evaluate "the best reliable and adaptive pipeline" — we separated Step 3 core, Extra Challenge, and Integration clearly |
| **Agent ablation** | `build_step4_eval_notebook.py` (P6b) | Was missing — requirement explicitly asks for ablation studies |
| **Mechanism checklist (A-H)** | `build_step4_eval_notebook.py` (Cell 8) | Requirement says "implement at least 4 of A-H" — we show 7/8 |
| **CHALLENGE set + abstention quality** | `build_step4_eval_notebook.py` (P7) | Was missing — requirement asks for correct/false abstention rates |
| **Confidence-correctness alignment** | `build_step4_eval_notebook.py` (P7) | Was missing — requirement asks for trust calibration |
| **Tradeoffs table (P9)** | `build_step4_eval_notebook.py` | Requirement says "discuss tradeoffs: quality, reliability, latency, complexity" |
| **5 qualitative examples** | `build_step4_eval_notebook.py` (P8) | Requirement lists exactly 5: grounded, revised, clarification, abstention, failure |
| **Intro rewrite** | `build_step4_eval_notebook.py` (Cell 1) | Old intro was too focused on Extra Challenge; now frames Step 3 as "the core requirement" |
| **Bug: gold_map persistence** | `build_step4_eval_notebook.py` (Cell 4) | `gold_map` was defined in Phase 7 but used in Phase 8 — would crash if running P8 without P7. Moved to Helpers cell |
| **Bug: orchestrator assertions** | `build_step4_eval_notebook.py` (Cell 3) | `waterfall_orchestrator` and `voting_orchestrator` used in `get_ir_docs` but never asserted after `%run` — would fail silently |
| **Test script** | `test_step4_evaluation.py` | Added 3 new tests for gold_map, waterfall, voting |
| **Colab guide** | `STEP4_RUN_GUIDE.md` | Complete workflow: upload → run → save → compare → iterate |

---

## 6. Code Flow — How Data Moves

```
Step 4.1 (runs via %run in Cell 3)
    ├─ loads corpus, retrievers, agents
    ├─ creates rag_system (ReliableAdaptiveRAG)
    ├─ creates MemoryStore, MemoryAugmentedRAG
    └─ creates eval_qa_data (24 benchmark questions)

Step 4 Evaluation
    Cell 4: Helpers
        ├─ gold_map = {qid → gold_answer}
        ├─ run_one(query, system) → single-row DataFrame
        └─ run_batch(items, system) → full DataFrame

    P0 Baseline:
        run_batch(eval_qa_data, rag_system)
        → memory/csv_outputs/step4_eval_baseline_step3.csv

    P3 Feedback loop:
        for each feedback question:
            r = rag_matched.run(question)
            compare r["final_answer"] vs gold (token-overlap)
            mem.record_feedback(...)
        → memory/memory_matched.json

    P4 Warm re-eval:
        run_batch(eval_qa_data, rag_matched)
        → memory/csv_outputs/step4_eval_warm_matched.csv

    P6b Agent ablation:
        run_batch(eval_qa_data, rag_system, ablate=["contradiction"])
        → memory/csv_outputs/step4_eval_agent_no_contradiction.csv

    P7 Reliability metrics:
        reliability_metrics(df_baseline)    ← uses baseline CSV
        reliability_metrics(df_warm_m)      ← uses warm CSV
        + CHALLENGE set run
        + confidence-correctness bins
```

---

## 7. The Generator Script Pattern

**Key rule:** All notebook content lives in `build_step4_eval_notebook.py`. Never edit the `.ipynb` directly — it gets regenerated.

```
You edit:  scripts/build_step4_eval_notebook.py
    ↓
Run:       python3 scripts/build_step4_eval_notebook.py
    ↓
Produces:  Step_4_Evaluation.ipynb (33 cells)
    ↓
Run:       python3 scripts/test_step4_evaluation.py
    ↓
Validates: All cells exist, all logic checks pass
```

This is why we made all changes through the generator today.

---

## 8. What Happens When You Run on Colab

```
Upload Step_4_Evaluation.ipynb
    ↓
Cell 2: Setup
    ├─ clones https://github.com/allarom/advanced-genai-26.git
    ├─ installs pytrec_eval, sentence-transformers, etc.
    └─ cd /content/advanced-genai-26

Cell 3: %run Step_4_1_extra_challenges.ipynb
    └─ runs Step 4.1 from cloned repo → imports everything

Cell 4–33: Evaluation
    ├─ resume-safe: existing CSVs are reused
    ├─ first run: ~20 minutes
    └─ produces: memory/csv_outputs/*.csv

Final step (you add):
    !cp -r memory /content/drive/MyDrive/step4_outputs/
```

---

## 9. Quick Reference — Files

| File | Purpose | Edit it? |
|------|---------|----------|
| `Step_4_1_extra_challenges.ipynb` | Extra Challenges — memory + HITL | ✅ Yes (if changing memory logic) |
| `STEP4_RUN_GUIDE.md` | Colab run guide — step-by-step workflow | ✅ Yes |
| `memory/csv_outputs/*.csv` | Evaluation outputs | ❌ No — generated by notebook |
| `memory/*.json` | Memory snapshots | ❌ No — generated by notebook |
