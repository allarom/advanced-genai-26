# Step 4 Colab Run Guide

How to run `Step_4_Evaluation.ipynb` on Google Colab, store outputs, run ablations, compare results, and iterate.

---

## 0. Verify Feedback Q&A (Optional)

Before running the evaluation, you can verify that the feedback questions and answers are factually grounded in the corpus.

### Where the source documents are

```
baseline/advanced_genAI-main/advanced_genAI/storage/general/document_level/
```

Each file is named: `{doc_id}_DOC_LEVEL.json`

### How to verify one item

1. Look at the `source_doc_id` column in `memory/feedback_qa_matched.csv` or `memory/feedback_qa_random.csv`.
2. Open the matching JSON file from the corpus folder above.
3. Check the `"text"` field for the person name or claim from the answer.

### Quick batch verification

Add this cell in a Colab notebook (or run locally):

```python
import csv, json, os

CORPUS_DIR = "baseline/advanced_genAI-main/advanced_genAI/storage/general/document_level"

def check_csv(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            doc_id = row["source_doc_id"]
            path = os.path.join(CORPUS_DIR, f"{doc_id}_DOC_LEVEL.json")
            if not os.path.exists(path):
                print(f"❌ {row['id']}: Missing file for {doc_id}")
                continue
            doc = json.load(open(path))
            text = doc["text"].lower()
            keywords = [w for w in row["answer"].lower().split() if len(w) > 3]
            hits = sum(1 for w in keywords if w in text)
            status = "✅" if hits >= 1 else "❌"
            print(f"{status} {row['id']}: {row['question'][:50]}...")

# Check both feedback files
check_csv("memory/feedback_qa_matched.csv")
check_csv("memory/feedback_qa_random.csv")
```

A green checkmark means the answer keywords appear in the source document. Red means they don't — that's a potential hallucination to fix before running the evaluation.

---

## 1. Before You Start

### What you need
- Google Colab (free tier is enough — only inference, no training)
- Your repo pushed to GitHub (the setup cell clones it automatically)

> ⚠️ **Important:** The setup cell clones your repo from GitHub. If you have local edits that aren't pushed, Colab will run the old GitHub version. **Push before running.**

> ⚠️ **Runtime timeout:** Colab disconnects after ~90 min of inactivity. A full run takes ~20 min, so stay active or save outputs immediately after.

---

## 2. Upload and Run

### Step A: Upload Step 4 Evaluation
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `Step_4_Evaluation.ipynb`

### Step B: Run top-to-bottom
Cell 2 (Setup) does everything:
- Mounts Google Drive
- Clones your repo to `/content/advanced-genai-26/`
- Installs packages (`pytrec_eval`, `sentence-transformers`, etc.)
- Changes working directory to the repo

Cell 3 (`%run Step_4_1_extra_challenges.ipynb`) automatically runs Step 4.1 from the cloned repo. **You do not need to upload Step 4.1 separately.**

**Wait for Cell 2 to finish before continuing.** It takes ~1–2 min (package install + clone).

### What each phase does (and how long)

| Phase | What | Time | Skip if CSV exists? |
|-------|------|------|---------------------|
| P0 Baseline | 24 queries through plain `rag_system` | ~2 min | ✅ |
| P1 Cold memory | Empty MemoryStore | ~2 min | ✅ |
| P2 Load feedback | Load CSVs, check no overlap | instant | — |
| P3 Feedback loop | 10 matched + 10 random HITL | ~3 min | ✅ |
| P4 Warm re-eval | Benchmark on warm snapshots | ~2 min | ✅ |
| P5 Comparison | Tables: IR + reliability + per-type | instant | — |
| P6 Memory ablation | Disable cache / fix strategy | ~4 min | ✅ |
| P6b Agent ablation | Full / No-Contra / No-Recovery | ~6 min | ✅ |
| P7 Reliability metrics | 8 metrics + CHALLENGE set | ~1 min | — |
| P8 Qualitative examples | 5 required cases | ~1 min | — |
| P9 Tradeoffs + discussion | Final comparison tables | instant | — |

**Total first run:** ~20 minutes.
**Resume run** (some CSVs already exist): only recomputes missing phases.

---

## 3. Run Ablations Separately (if needed)

Ablations are **resume-safe** — they check if their CSV already exists.

### To re-run just agent ablation (P6b)

Add a cell above P6b and run it:
```python
!rm -f memory/csv_outputs/step4_eval_agent_no_contradiction.csv
!rm -f memory/csv_outputs/step4_eval_agent_no_recovery.csv
```
Then re-run the P6b cell. The `agent_full` CSV will be skipped (already exists), but No-Contra and No-Recovery will recompute.

### Same for memory ablation (P6)
```python
!rm -f memory/csv_outputs/step4_eval_ablation_no_cache.csv
!rm -f memory/csv_outputs/step4_eval_ablation_fixed_strategy.csv
```

---

## 4. Save Outputs to Drive

After the run, the `memory/` folder is in `/content/advanced-genai-26/memory/`.

**Option A: Copy to Drive**
Add a cell at the very end:
```python
!cp -r /content/advanced-genai-26/memory /content/drive/MyDrive/step4_outputs/
```

**Option B: Download via Files panel**
Left sidebar → Files → `advanced-genai-26` → `memory/` → right-click → Download.

### Which files matter
| File | Why save it |
|------|-------------|
| `memory/csv_outputs/step4_eval_baseline_step3.csv` | Baseline for all comparisons |
| `memory/csv_outputs/step4_eval_warm_matched.csv` | Main result |
| `memory/csv_outputs/step4_eval_agent_*.csv` | Agent ablation evidence |
| `memory/csv_outputs/step4_eval_ablation_*.csv` | Memory ablation evidence |
| `memory/memory_matched.json` | Warm memory snapshot |

---

## 5. Compare Results

Add a cell at the end of Step 4 Evaluation (or in a new notebook):

```python
import pandas as pd

b = pd.read_csv("memory/csv_outputs/step4_eval_baseline_step3.csv")
w = pd.read_csv("memory/csv_outputs/step4_eval_warm_matched.csv")

print(f"Baseline MRR: {b['mrr'].mean():.4f}")
print(f"Warm MRR:     {w['mrr'].mean():.4f}")
print(f"Cache hits:   {w['cache_hit'].sum()}")
print(f"Decisions:\n{w['decision'].value_counts()}")
```

### Compare ablations
```python
full = pd.read_csv("memory/csv_outputs/step4_eval_agent_full.csv")
no_c = pd.read_csv("memory/csv_outputs/step4_eval_agent_no_contradiction.csv")

print("Answers — Full:", (full['decision'].str.contains('answer')).sum())
print("Answers — No-Contra:", (no_c['decision'].str.contains('answer')).sum())
```

---

## 6. Improve and Re-Evaluate

### What you can change
| Component | Where | Effect |
|-----------|-------|--------|
| Contradiction threshold | Step 3 `ContradictionAgent` | More/less abstentions |
| Trust threshold | Step 3 `ReliableAdaptiveRAG` | Stricter/looser answering |
| Recovery strategies | Step 3 `RecoveryAgent` | More retry options |
| CHALLENGE queries | `scripts/build_step4_eval_notebook.py` | More test cases |

### Iteration loop
1. Change code in Step 3 (or edit generator and rebuild).
2. **Delete affected CSVs** so they recompute:
   ```python
   !rm memory/csv_outputs/step4_eval_baseline_step3.csv
   !rm memory/csv_outputs/step4_eval_warm_matched.csv
   ```
3. Re-run Step 4 Evaluation from the relevant phase.
4. Compare new CSV against old.

### Keep a run log
In Colab, create the log in Drive so it persists:
```python
log = """| Date | Change | Baseline MRR | Warm MRR | Notes |
|------|--------|-------------|----------|-------|
| 2026-06-01 | Original | 0.208 | 0.209 | No gain, as expected |"""
with open("/content/drive/MyDrive/step4_run_log.md", "w") as f:
    f.write(log)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `rag_system not found` | Cell 3 (`%run Step_4_1`) failed. Check its output for errors, then re-run cells 2–3 |
| Phase skipped (says "loaded") | Delete its CSV in the Files panel or via `!rm memory/csv_outputs/...csv`, then re-run that cell |
| Slow first run | Normal — corpus + retrievers load. Wait 2–3 min |
| `gold_map undefined` | Re-run from cell 4 (Helpers) or earlier — Phase 8 needs `gold_map` from Helpers |
| Runtime disconnected mid-run | Re-run from start. Phases with existing CSVs will skip automatically |

---

## Summary — What to Do Right Now

1. **Push repo to GitHub** (Colab clones from there).
2. **Upload `Step_4_Evaluation.ipynb` to Colab.**
3. **Run top-to-bottom.** Cell 2 clones repo, Cell 3 auto-runs Step 4.1 via `%run`.
4. **Save `memory/` folder to Drive** (add the `!cp` cell at the end).
5. **Compare baseline vs warm CSVs** (add a comparison cell at the end).
6. **Iterate** — push changes to GitHub, delete affected CSVs, re-run.
