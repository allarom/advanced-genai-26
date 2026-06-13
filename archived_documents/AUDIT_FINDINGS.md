# Audit Findings — Reliable Adaptive Agentic RAG

Read-only audit of the project + reports for severe logic/faithfulness errors, requirement gaps, and report quality. Each item tagged **[grade impact]** and **[effort]**. This is an internal working note — do NOT submit it.

## TL;DR
- **MOST IMPORTANT (P0-0):** the §6.10 challenge-set "10/10 correct" claim is **false** vs the notebook output — actual is **3/6 correct abstention**, and **both conflicting-evidence queries were answered** (contradiction did not fire). This must be corrected; it currently contradicts the data and §7.
- **Good news:** The core Step 3 logic is correct and the bugs your report admits were fixed are *genuinely* fixed (trust formula, abstain threshold, recovery-before-abstain ordering, `query_type` threading, regex). Step 4.1 memory logic is sound. The §5.1/5.3/6.x CSV-backed numbers match (apart from 3 small §5.3 values).
- **Must-fix before submit:** correct §6.10 (P0-0); rebuild the stale PDFs; fix the `pytrec_eval` claim; **the two "baseline MRR" numbers (0.209 vs 0.3646) are NOT comparable — different code, depth, and rerank** (P1-2); correct 3 small numbers in §5.3.

## Source-of-truth discipline (per your instruction)
- **Ground truth = code-computed outputs** = the CSVs in `memory/csv_outputs/` (exported by the Step 4 eval code) and the saved notebook output cells. I treated these as authoritative.
- **NOT ground truth:** the report prose, the README, and even `verify_facts.py`'s hardcoded "expected" values — these can be AI/human-written. In fact `verify_facts.py`'s expectations match the *report* (0.062/0.020/0.099), but the *CSV* disagrees → the report+script are wrong, the CSV is right (see P1-3).
- **Caveat:** "ground truth" assumes the code that produced the CSV is logically sound. I verified the decision/trust/recovery code is sound, so the CSVs are trustworthy. The one place the *metric definition itself* is questionable is the MRR (P1-2/P1-5).

---

## P0 — Severe (fix before submitting)

### P0-0. Challenge-set results in §6.10 are UNFAITHFUL to the notebook output (most serious) — **FIXED**
- **STATUS: FIXED** (report-only) — `report.md` §6.10 table rewritten to the verified numbers (correct abstention **3/6**, false abstention **0/4**, clarification **2/2**, 9/12 overall); the dependent "100% challenge-set" claims in §7 (critical reflection + tradeoff table) and the §8 conclusion were also corrected for consistency. Re-verified against the notebook's `ch_results_plain` output via `scripts/extract_challenge.py`.
- **[impact: Step4=10, Step5=5, credibility]** **[effort: medium]**
- **Ground truth (saved output in `Step_4_Evaluation.ipynb`, the only challenge run `ch_results_plain`):**
  `Correct abstention: 3/6` and `False abstention: 0/4`. Three queries that should abstain were **ANSWERED**:
  - `"Who won the ETH robotics competition in 1999?"` (insufficient) → **answer** (wrong)
  - `"Did ETH's student numbers go up or down in 2015?"` (conflicting) → **answer** (wrong)
  - `"Is ETH bigger or smaller than EPFL in staff count?"` (conflicting) → **answer** (wrong)
- **The actual challenge set has 12 queries** (`CHALLENGE` list): ambiguous 2, insufficient 2, conflicting 2, adversarial 2, **standard 4 (expect answer)**.
- **Report §6.10 claims** "10 queries… 10/10 correct… **Conflicting 2/2 correct, contradiction detected, abstained**… no false answers, no false abstentions." This is **false**: both conflicting queries were answered (contradiction did NOT fire), only 3/6 abstained, and the 4 standard "should-answer" queries are omitted entirely.
- **Internal contradiction:** §7 admits "keyword contradiction misses semantic conflicts" — which is exactly why these 2 conflicting queries slipped through. So §6.10 contradicts §7 and the data.
- **Why it slipped past `verify_facts.py`:** that script only checks §5.1/5.3/6.6/6.9/8 — it never checks §6.10, and §6.10 has no CSV.
- **Falsification checks (to be sure this isn't my error):** (1) the strings `10/10`, `all 10`, `2/2 correct`, `3/3 correct` appear in **no** notebook's saved output; the only challenge result anywhere is `3/6`. (2) There is **no challenge set in `Step_4_1`** — the only `CHALLENGE` list is the 12-query one in `Step_4_Evaluation`. (3) The report says results are on **"warm memory"**, but the only run is `run_challenge(rag_system)` = **plain system, no memory**. So the §6.10 table matches no reproducible artifact.
- **Fix (honest + still strong):** replace the §6.10 table with the real numbers — clarify 2/2, adversarial 2/2, insufficient 1/2, conflicting **0/2**, standard 4/4, overall correct-abstention **3/6**, false-abstention **0/4**. Frame the conflicting-evidence misses as evidence for the §7 limitation (keyword contradiction is too shallow) — this is a *more credible* analysis than a fake 100%, and graders reward honest failure analysis (Step 5 explicitly asks "what did not work and why"). Also fix the §8 conclusion sentence that claims "abstains correctly 100% on the challenge set."

### P0-1. Stale PDFs — you submit the PDF, and both are out of date — **PENDING**
- **STATUS: PENDING** — deliberately deferred to the final step (rebuild both PDFs only after all content edits land, per the plan's Phase G).
- **[impact: all steps]** **[effort: quick]**
- `report.pdf` last built **Jun 4**, but `report.md` changed **Jun 8 + today** (diagram rewrites, content). `baseline_repro_report.pdf` is from **Apr 22** vs its `.md` today.
- **Risk:** every report improvement you made is missing from the submitted artifact.
- **Fix:** rebuild both before submitting, e.g.
  `pandoc report.md -o report.pdf` (you previously used a LaTeX/pandoc toolchain — use the same one so the `\footnotesize` ASCII diagrams render). Then visually confirm the §3.2 diagram and tables render.

---

## P1 — Important (real marks at stake)

### P1-1. `pytrec_eval` claim is false (faithfulness)
- **[impact: Step1=15, Step5=5]** **[effort: quick]**
- `report.md` §1.1: *"metrics computed with `pytrec_eval`."* The Step 1 notebook actually computes MRR/P@k/Recall@k with **custom functions** (`reciprocal_rank()`, `precision_at_k()`, `recall_at_k()` ~line 464). **Falsification check:** `pytrec_eval` appears **only in the `pip install` line** — `import pytrec` / `pytrec_eval.` / `RelevanceEvaluator` have **zero** hits in the notebook, so it is installed but never used. `baseline_repro_report.md` correctly says "recompute ourselves with one shared evaluator."
- **Fix:** change §1.1 to "computed with a shared custom IR evaluator" (matches the baseline report and the code).

### P1-2. The two "baseline MRR" numbers are NOT comparable (invalid comparison) — VERIFIED against code
- **[impact: Step4=10, Step1=15]** **[effort: medium]**
- **What each number actually is (from the code, not the prose):**
  - **Step 1 "Confidence MRR = 0.209"**: computed by Step 1's *own reproduction* `orchestration_methods['Confidence'].search(q, top_k=TOP_N)` with **`TOP_N = 100`** (ranks 100 docs), MRR via `reciprocal_rank` over depth-100.
  - **Step 4 "MRR = 0.3646"**: computed in `run_one()` via `get_ir_docs()` → the **real legacy `orchestrator.run(query, retrieve_k=50, top_k=max(K_VALUES)=10)`** with the full pipeline (incl. X-encoder rerank), MRR over depth-**10**.
- **Confirmed root cause of the gap:** Step 1's reproduction reranks with **lexical `_overlap_rerank`** (and `HybridRetriever(rerank=False/True)` uses overlap, no neural model), while the legacy orchestrator used in Step 3/4 reranks with a **neural CrossEncoder (`ms-marco-MiniLM-L-6-v2`)**. Neural rerank + depth 100→10 fully accounts for 0.209 vs 0.3646.
- **So they differ in implementation (reproduction vs real orchestrator), reranker (lexical overlap vs neural cross-encoder), and depth (100 vs 10).** They are **not logically comparable**, yet they are the two things a reader will treat as "the baseline." The requirement (Step 4) explicitly asks to "compare directly against the baseline from Step 1" — that valid comparison is currently **not made**.
- Note: the report's *internal* claim "MRR unchanged 0.3646 → 0.3438 (memory doesn't change retrievers)" **is valid** (same harness, same code) — keep that. The problem is only the cross-notebook 0.209 vs 0.3646.
- **Fix (pick one):** (a) recompute the final system's IR with the **same evaluator + depth as Step 1** (`reciprocal_rank`, depth-100) and present one apples-to-apples Step-1↔final table; or (b) explicitly state in §5/§6 that the Step-4 MRR uses a different harness/depth than Step-1 and is therefore not directly comparable, and rename "baseline" → "Step-3 (no memory)" to remove the ambiguity.

### P1-5. The reported MRR is decision-independent and includes abstain/clarify — clarify the metric (faithfulness/logic)
- **[impact: Step4=10]** **[effort: medium]**
- `get_ir_docs()` runs the orchestrator **regardless of the reliability decision**, so the `mrr` column is pure retrieval IR, *decoupled* from answer/abstain/clarify. Evidence: **clarified queries have nonzero MRR** (qid 14 = 1.0, qid 12 = 0.10) even though clarification short-circuits before retrieval in the decision path.
- Consequence 1: the headline **0.3646 averages over all 24 incl. 9 abstain + 3 clarify**; the **answer-only MRR is 0.2986** (n=12). Presenting 0.3646 near the reliability results can read as "answer quality," which it is not.
- Consequence 2 (notable): **abstained queries have HIGHER retrieval MRR (0.452) than answered ones (0.299)** — qid 4/11/25 abstained despite the relevant doc at **rank 1 (MRR=1.0)**. So abstention is driven by the trust heuristics, not by retrieval failure. The report's "abstentions are correct / threshold cleanly separates" framing is about trust scores, but on this 24-query set several abstentions sit on well-retrieved queries — i.e. possible **over-conservative / false abstentions** worth acknowledging (MRR≠groundedness, so not necessarily wrong, but the narrative should be nuanced).
- **Fix:** (a) state explicitly that MRR is the underlying orchestrator IR, decision-independent; (b) report answer-only MRR (0.299) alongside the all-query value; (c) soften the "abstentions are all correct" claim for the benchmark set, or add the groundedness rationale for the high-MRR abstentions.

### P1-3. Three wrong numbers in §5.3 trust-distribution table (faithfulness)
- **[impact: Step4=10]** **[effort: quick]**
- `verify_facts.py` FAILs vs the CSVs:
  - answer **Std Dev**: report `0.062` → actual **0.057**
  - abstain **Min trust**: report `0.020` → actual **0.000**
  - abstain **Std Dev**: report `0.099` → actual **0.116**
- The prose at §5.3 (line ~397) repeats `0.062`, `0.099`, and "far below (0.020)" — update those too.
- **Fix:** correct the four cells + the sentence. (All other §5.1/5.3/6.6/6.9/8 numbers verified correct.)

### P1-4. Title/framing undersells the required Step 4 evaluation
- **[impact: Step4=10, Step5=5]** **[effort: quick]**
- Title: *"Steps 1–3 + Step 4.1 Bonus."* The required **Step 4 Evaluation (10 pts)** content exists (it's in §5 and §6) but isn't labelled as Step 4, so a grader may not map it to the rubric.
- **Fix:** rename §5 (and the Step-4 parts of §6) to explicitly say "Step 4: Evaluation," or add a one-line rubric map in the executive summary ("Step 4 evaluation = §5 + §6.4–6.10").

---

## P2 — Minor (polish / clarity)

### P2-1. Failure taxonomy is multi-label but report never says so
- **[impact: Step1=15]** **[effort: quick]**
- Counts sum to **29 across 24 queries** because the Step 1 code assigns *multiple* labels per query (verified; counts 10/8/5/4/1/1 match the saved notebook output exactly). A grader may read 29>24 as an error.
- **Fix:** add one sentence to §1.4: "labels are non-exclusive; a query may exhibit multiple failure modes, so counts sum to more than 24." (The code can also emit `grounding_failure`/`ambiguity_failure`; both scored 0 here — fine to omit.)

### P2-2. Step 3 notebook has invalid JSON (missing `execution_count`)
- **[impact: Step5=5]** **[effort: quick]**
- `jupyter nbconvert` refuses `Step_3_...ipynb` ("'execution_count' is a required property"), likely from a Colab/manual edit. Jupyter/Colab usually still open it, but a grader running tooling could hit this.
- **Fix:** open and re-save the notebook in Jupyter/Colab (or run "Restart & Run All") to normalize the JSON.

### P2-3. Challenge-set naming + missing multilingual category
- **[impact: Step4=10]** **[effort: quick]**
- Step 4 code labels the robustness category `adversarial`; report §6.10 calls it `off-topic`. No multilingual category (requirement lists it as "if relevant" — optional, but worth a one-line note that it was out of scope).
- **Fix:** align naming; add a sentence on multilingual being out of scope.

### P2-4. Required qualitative cases only partially explicit
- **[impact: Step4=10]** **[effort: medium]**
- Requirement wants 5 named cases: grounded answer, **revised-after-critique**, clarification, abstention, hard failure. §5.2 shows clarification/abstention/contradiction but doesn't explicitly label a "successful grounded answer" or a "revised-after-critique/recovery" example.
- **Fix:** label one grounded-answer example and one recovery-success example (e.g., a keyword query where recovery switched strategy and then answered) as the "revised after critique" case.

### P2-5. Only 3 of 9 screenshots used
- **[impact: Step1/Step4 presentation]** **[effort: quick]**
- `01_benchmark_table.png`, `02_grouped_means.png`, `03_decision_counts.png`, `memory_sys_working.png` exist but aren't embedded. Adding the benchmark table / decision-counts figures would strengthen §1–§5.

---

## Verified CORRECT (no action — reassurance)
- **Step 3 trust** = `0.6*sufficiency + 0.3*groundedness − 0.4*contradiction`, clamped `max(0,min(1,·))`; abstain threshold `0.4`. (verified)
- **Recovery-before-abstain**: `run()` attempts recovery when `result["abstain"]` is true, recomputes signals, *then* the abstention branch runs. The admitted "recovery at wrong time" bug is fixed. (verified)
- **`query_type` threading** into `_compute_signals` / critic. (verified)
- **Regex** (contradiction word-boundary, year `\b(1\d{3}|20\d{2})\b`) — correct single-escaped raw strings, not over-escaped. (verified)
- **Step 4.1**: weight clamp `[0.3, 1.6]`; M2.5 reflection substrings aligned with `failure_reason_from_signals()` (substring-mismatch bug fixed); `MemoryAugmentedRAG` composition + `try/finally` weight restore. (verified)
- **Faithfulness (CSV-backed)**: per-type decision splits, recovery 3/12 (3/5 keyword, 0/7 others), abstention 37.5%, trust gap 0.45, clarification 12.5%, MRR 0.3646/0.3438, ablation counts (21/0, 9/12, 14/7) — all match. (verified)
- **QID 1 exclusion** confirmed (CSVs start at QID 2; 24 queries). Worth a one-line justification in §5.1 but not an error.
- **Patch-revert risk**: the current Step 3 / Step 4.1 notebooks contain the *fixed* code, not pre-fix versions. (verified)
- **Legacy Step 2 retriever internals**: weighted **RRF** `w * 1/(k_rrf+rank)` (k=60) is the standard formula; gating drops sub-threshold retrievers with a "never gate all out" guard; **CrossEncoder rerank** (top-50 cands) with lexical fallback; critic-driven single retry with broadened weights. No logic bug. (verified)
- **NOT verified-correct (correction to earlier pass):** the §6.10 challenge-set results are **wrong** (see P0-0). My first pass had trusted the report prose here; re-checking against the saved `ch_results_plain` output exposed the 3/6 reality. This is exactly your point about using code output, not prose, as truth.

---

## Suggested 1-day fix order (impact × effort)
0. **P0-0 rewrite §6.10 challenge-set table + §8 sentence** with the real 3/6 numbers (medium, highest credibility risk). Frame conflicting-evidence misses as support for the §7 keyword-contradiction limitation.
1. **P1-3 fix §5.3 numbers** (quick, faithfulness). Update `verify_facts.py`'s expected values to the CSV truth too, then re-run until "NO ERRORS."
2. **P1-1 pytrec_eval wording** (quick).
3. **P1-4 Step-4 labelling** (quick).
4. **P1-2 + P1-5 MRR clarity** (medium, highest Step-4 value): state the Step-4 MRR is the decision-independent orchestrator IR at depth-10; do NOT compare it to Step-1's 0.209 (different impl/depth/rerank); add answer-only MRR (0.299); keep the valid baseline-vs-warm claim; rename "baseline" → "Step-3 (no memory)".
5. **P2-1 taxonomy multi-label sentence**, **P2-3 naming**, **P2-2 re-save Step 3** (quick).
6. **P2-4 qualitative case labels**, **P2-5 add figures** (if time).
7. **P0-1 LAST: rebuild + eyeball both PDFs** (must be the final step, after all edits land).

## Audit depth note
Deep-verified: Step 3 agent logic, Step 4.1 memory, all CSV-backed numbers, Step 1 taxonomy + metrics, regex, **legacy Step 2 retriever internals (RRF/gating/rerank/retry)**, and the challenge-set output. Lighter pass: full paragraph-by-paragraph prose readability — flag if you want a dedicated copy-edit pass.
