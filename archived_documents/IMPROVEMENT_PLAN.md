# Tomorrow's Fix & Improvement Plan — Reliable Adaptive Agentic RAG

A time-boxed, severity-ordered plan to fix the audit's faithfulness/logic errors first (incl. a guarded code-fix attempt for the conflicting-query miss), then raise report presentation and analytical substance, then rebuild the PDFs last. Internal working note — do NOT submit it.

## Guiding principles (per your ask)
- **Faithfulness first:** every number in the report must match a saved notebook/CSV output. No prose-only claims.
- **Substance over filler:** each section leads with a *project-specific* insight + interpretation of what the number means; cut generic "RAG is important" padding.
- **Minimal, revertible code changes:** the one code fix is guarded — keep only if it improves the target *without* regressing the benchmark; otherwise revert and report honestly.
- **One golden rule on sequencing:** if we re-run any notebook, ALL dependent report numbers + `verify_facts.py` must be re-synced before the PDF build. Never half-sync.

---

## Rubric alignment & grading leverage (checked against `requirement_new_2026.pdf`)
**Where the plan already wins points** (these fixes map onto *named* rubric items):
- Step 4 explicitly grades **correct-abstention rate** and **false-abstention rate** → fixing §6.10 to the honest 3/6 (P0-0) lands exactly here. A fabricated 100% risks a credibility/integrity hit; the honest version scores the same metric *and* feeds Step 5.
- Step 4 requires **"compare to the baseline"** → Phase D makes that comparison valid.
- Step 4 requires **≥1 ablation** → already present (§6.7/6.8); E2 sharpens its interpretation.
- Step 5 requires **"what did not work and why"** → the §6.10 conflicting-query miss (E4/C2) is ideal evidence.
- Step 3 mechanisms A–H all implemented; the **Adaptation Requirement** is met by recovery switching strategy/rewriting.

**Easy points the report is currently MISSING (add in Phase E):**
- **Individual member contributions** — the rubric *requires* this in the submission; not found in `report.md`. (E9)
- **Citations of external resources** — zero references found, but BM25/RRF/E5/ms-marco cross-encoder/GraphRAG are used. (E10)
- **Metric justification** — rubric: "explain which metrics you use and why." (folded into D1/E11)
*(Already satisfied: efficiency/latency — runtime columns in §5.1/§6.5–6.7; AI-tool declaration — Appendix B; benchmark extension — challenge set; interpretable traces — §3.4.)*

---

## Language naturalness guardrails (apply to EVERY text edit in C/D/E)
Verified baseline: the report body currently has **no emojis, no unicode arrows (→), no checkmarks**, and only one mild AI-phrase — so the job is to **keep it that way** and not regress when adding/rewriting prose.
- **No emojis, no unicode arrows/symbols in prose.** Arrows belong ONLY inside the ASCII diagram code blocks (`-->`, `|`, `v` are fine there).
- **Short, plain sentences.** One idea per sentence; break run-ons. Write like a capable university student explaining their own work — professional and approachable, not marketing copy.
- **Kill AI-filler.** Avoid "it's worth noting", "importantly", "moreover", "furthermore", "comprehensive", "leverage", "seamless", "robust", "a testament to", and 3-item parallel triads. Cut any sentence that would be true of *any* RAG project.
- **Trim em-dashes.** There are ~35 `---` em-dashes; prefer periods/commas and roughly halve them.
- **Avoid long comma-lists** (e.g. the 7-item agent list in §1.4) — name the 2–3 that matter, not all of them.
- **Every sentence earns its place with specifics:** a number, an agent name, a query, or a behavior from *this* system. If it has no project-specific anchor, delete it.
- **Voice consistency:** match the existing report's tone so edits don't read as pasted-in.

---

## CRITICAL sequencing gate (decide at the start)
**Verified data flow:** `Step_4_Evaluation.ipynb` `%run`s → `Step_4_1` → `Step_3` → `Step_2`, and it writes **all 8 `step4_eval_*.csv`** via `save_version()`. `verify_facts.py` and report §5.1/§5.3/§6.6/§6.9/§8 all read those same CSVs. So the `ContradictionAgent` edit in Phase B propagates through the whole chain and **regenerates all 8 CSVs at once** — baseline, cold/warm memory, AND the `no_contradiction` / `no_recovery` / `fixed_strategy` ablations.
- **Cascade is broad:** a re-run changes **essentially every number in §5–§6** (trust dist, abstention/grounded rates, ablation counts, challenge set) AND every hardcoded `expected` value in `verify_facts.py`.
- **If Phase B re-run succeeds** → re-sync ALL of §5–§6 + `verify_facts.py` from the new CSVs (not just §6.10/§5.3).
- **If Phase B is skipped / fails / not finished** → REVERT the edit; use the EXISTING CSVs/outputs (challenge = real **3/6**); no other number moves.

Either path is fully faithful. What we must NOT do is mix old and new numbers. Note: the stale `2026-05-29_output_step3.csv` is unused by `verify_facts.py`/report — ignore it. **If Phase B re-runs, also sync the README's "Key Results" numbers** so the repo isn't self-contradictory.

---

## Phase A — Setup & feasibility (15 min) — do first
- A1. Confirm the Colab/Jupyter env + ETH corpus + LLM/API access are available to re-run `Step_4_Evaluation.ipynb` (it chains 4.1→3→2). If not available → **skip Phase B, go report-only**.
- A1b. **Recommendation (time-aware):** because Phase B regenerates all 8 CSVs and forces a full §5–§6 + `verify_facts.py` re-sync (~1–2 h compute + ~1 h re-sync, plus regression risk), it is only worth it if the env is ready early and the validation gate passes cleanly. If there's any doubt about finishing before the deadline, **go report-only** — the honest 3/6 is fully faithful and strong for Step 5.
- A2. Make a git checkpoint/branch so the code edit is trivially revertible.

---

## Phase B — SEVERE code-fix attempt (guarded, ~2–3 h incl. re-run + re-sync, hard cutoff) [P0-0]
Goal: make the 2 conflicting-evidence challenge queries abstain *without* breaking the benchmark.

- B1. **Edit one cell** — `ContradictionAgent.CONTRADICTIONS` (Step 3 cell 14, line ~3510). The **notebook is the source of truth** (verified: `scripts/patch_improvements.py` is a one-time migration script that writes the notebook once and is not called by any build pipeline). So edit the notebook cell directly. **Also update the same antonym list in `scripts/patch_improvements.py`** — not because it's needed for the change to take effect, but so that if anyone re-runs that script it won't silently *revert* your fix. Add comparison antonyms:
  `("bigger","smaller")`, `("larger","smaller")`, `("grew","shrank")`, `("more","fewer")`, `("expanded","contracted")`.
  *(Keep word-boundary matching; do NOT add `up/down` — too common, high false-trigger risk.)*
- B2. **Re-run only `Step_4_Evaluation.ipynb`** end-to-end (it `%run`s 4.1 → 3 → 2 automatically) → this regenerates all 8 `step4_eval_*.csv` + the `ch_results_plain` challenge output in one pass. NOTE: this is ~8 variants × 24 queries of LLM-driven runs — **realistically 1–2 h of compute alone**, before any re-sync.
- B3. **VALIDATE (mandatory gate):**
  - Challenge correct-abstention improved (target: conflicting 0/2 → 2/2; overall ≥ 5/6).
  - **No regression on the 24-benchmark:** abstention rate, grounded rate, answer/abstain/clarify counts stay essentially as before (the new keywords must not create *false* contradictions).
- B4. **GATE:** improved AND no regression → keep + record new numbers. Otherwise → **revert** B1 and proceed report-only with the honest **3/6**.
- B5. **Conceptual honesty (RAG/AI correctness):** this keyword expansion is a *band-aid*, not the conceptually-right fix. `ContradictionAgent` scans the **retrieved docs**, not the query, so it only fires if both antonyms literally appear in the top-k passages — for "bigger or smaller than EPFL" the corpus may simply not surface conflicting text, in which case no keyword list helps. The conceptually-correct fix is **semantic/NLI-based contradiction detection** (already named in §7 and the rubric), which is too large for one day. So frame B in the report honestly: either "a small lexical extension recovered the comparison cases, but semantic conflicts still need NLI," or "even expanded keywords miss these, confirming keyword matching is the wrong tool and motivating NLI." Both are strong Step-5 reflections; neither overclaims.

---

## Phase C — SEVERE faithfulness corrections (must-fix) [P0-0, P0-1 prep, P1-1, P1-3]
Pull every number from the regenerated CSVs (if Phase B ran) or the existing ones.
- C1. **§6.10 challenge table** → real numbers. **[FIXED]** report-only: rewritten to clarify 2/2, adversarial 2/2, insufficient 1/2, **conflicting 0/2**, standard 4/4, overall **correct-abstention 3/6, false-abstention 0/4** (9/12 correct); removed the false "warm memory / 10 queries / contradiction detected" framing; now states the plain-system 12-query run. Verified vs `ch_results_plain`.
- C2. **§8 conclusion** → delete/replace the "abstains correctly 100% on the challenge set" sentence with the honest result + the limitation it exposes. **[FIXED]** §8 corrected; also fixed the dependent "100% challenge-set accuracy" claims in §7 (critical reflection + tradeoff table) for consistency.
- C3. **§5.3 numbers** → correct the 3 wrong values (answer-std, abstain-min, abstain-std) to the CSV truth; also update `verify_facts.py`'s hardcoded "expected" values to the CSV, then re-run until it prints NO ERRORS.
- C4. **§1.1 `pytrec_eval`** → change to "computed with a shared custom IR evaluator" (it's installed but never imported/used).

---

## Phase D — SEVERE: make the MRR comparison logically valid [P1-2, P1-5]
- D1. In §5/§6: state explicitly that the Step-4 MRR (0.3646) is **decision-independent orchestrator IR at depth-10 with a neural cross-encoder reranker**, and is **not** comparable to Step-1's 0.209 (which uses lexical overlap rerank at depth-100). Either present one apples-to-apples table OR rename "baseline" → **"Step-3 (no memory)"** to kill the ambiguity.
- D2. Add **answer-only MRR (0.299)** alongside the all-query 0.3646, and add one sentence on the notable finding that **abstained queries have higher retrieval MRR (0.452) than answered (0.299)** → abstention is trust-driven, not retrieval-driven (nuances the "abstentions all correct" claim). Keep the valid internal "0.3646 → 0.3438 (memory doesn't change retrievers)" comparison.

---

## Phase E — Presentation, structure & analytical substance (the grade-driver) [P1-4, P2-1/4/5]
This is where "good analysis, not blah blah" lives. Target the biggest marks.
- E1. **Title/scope** → include the Step-4 evaluation (currently undersold as "Steps 1–3 + 4.1 bonus").
- E2. **Insight-first sections:** open each results subsection with the *finding*, then the table, then 1–2 sentences of interpretation. Pull the exact numbers from the relevant CSV/§6.8 at edit time (do NOT trust the round figures below). Lead examples:
  - Contradiction is the highest-impact agent — quantify with the `step4_eval_agent_no_contradiction.csv` answer/abstain counts vs baseline.
  - Recovery only fires/helps on keyword queries (≈3/5), never on others (0/7) — say why.
  - Trust threshold 0.4 separates answered vs abstained — but on this set some high-MRR queries still abstain (interpret why).
- E3. **Cut filler:** delete generic background paragraphs and any sentence that doesn't reference *this* system's numbers/behavior.
- E4. **Five required qualitative cases** (§5.2) — explicitly label: (1) grounded answer, (2) revised-after-critique = a recovery-success keyword query, (3) clarification, (4) abstention, (5) **hard failure = a conflicting query that was answered** (use the real failure from P0-0 — strong, honest analysis).
- E5. **Figures:** embed the unused screenshots that add evidence (`01_benchmark_table.png`, `03_decision_counts.png`, `memory_sys_working.png` — all exist) near the matching tables. They auto-number as "Figure N" via the configured `caption` package, so just add `![caption](path)` on its own line.
- E6. **One-liners:** note the failure taxonomy is multi-label (counts overlap); note QID-1 exclusion (CSVs start at QID 2; 24 queries) with a one-sentence reason.
- E7. **Heading-numbering consistency (render):** **[DONE]** numbered §7 subsections `7.1`–`7.9` to match §1–§6's manual numbering; verified consistent in the rendered PDF. (Build keeps `number_sections: false`, so no doubling.)
- E8. **Requirement-coverage pass:** quickly confirm each graded requirement is *visibly* answered — Step 1 (baseline reproduce + taxonomy + efficiency), Step 2 (multi-agent design + strategies), Step 3 (mechanisms A–H + decision policy + adaptation), Step 4 (evaluation vs baseline + ablation + reliability metrics + benchmark extension), Step 5 (limitations / what didn't work + why), and the Step 4.1 bonus. The honest §6.10 failure now strengthens Step 5.
- E9. **Individual contributions (rubric-required):** add a short "Contributions" subsection (or appendix line) stating what each group member did — design, implementation, experiments, analysis, writing. If solo, state that explicitly. Do not skip; it's an explicit submission requirement.
- E10. **References (Step-5 professionalism):** add a short References section citing the external methods actually used — BM25, Reciprocal Rank Fusion, `multilingual-e5-large-instruct`, the `ms-marco-MiniLM` cross-encoder, GraphRAG, and any papers/repos. Do NOT cite `pytrec_eval` as used (it isn't — see C4).
- E11. **Metric justification (rubric: explain why):** one or two sentences stating MRR/P@k/Recall@k measure *retrieval*, grounded-rate / unsupported-rate / abstention rates measure *reliability/answer support*, and that BLEU/ROUGE are not used because there are no long-form gold answers — grounding is the answer-quality proxy.

---

## Phase F — Minor / robustness (only if time) [P2-2, P2-3]
- F1. Re-save `Step_3` notebook with valid JSON (`execution_count`) to prevent patch-revert / render issues.
- F2. Fix challenge-category naming consistency ("adversarial" vs "off-topic") so report = code.

---

## PDF render state — already verified GOOD (so we don't over-work it)
- **Frontmatter is solid:** YAML sets `geometry: margin=2.5cm`, `fontsize: 10pt`, `onehalfspacing`, `times`, `booktabs`, `titlesec`, and a `caption` setup. ASCII diagrams are correctly scoped `\footnotesize … \normalsize` (6 paired, **no font leak**). The widest source lines are all **prose** (they wrap automatically) — no horizontal overflow from text. `\newpage` already precedes Appendix A.
- **So the format is largely clean already** — the remaining render work is small and targeted (below).

## Phase G — LAST: render fixes + rebuild + eyeball PDFs [P0-1] — **DONE (rebuilt this round; re-run after further edits)**
- G1. **Two small pre-build fixes:** **[DONE]** added `\newpage` before `## Appendix B`. **[N/A]** the `%20` image (`manual and auto feedback compare.png`) was *not* renamed — it already renders correctly as Figure 3 in both pandoc/xelatex and rmarkdown/pdflatex (verified visually), so no rename was needed.
- G2. **Build:** **[DONE]** built BOTH PDFs via `Rscript scripts/build_reports.R` = the RStudio `rmarkdown::render` path (pdflatex, `number_sections` off, `fig_caption` on). No errors, only a harmless `--highlight-style` deprecation warning.
- G3. **Visual QA checklist (eyeball the rendered PDF):** **[DONE]** verified on rendered pages.
  - Section numbers read cleanly (no "1 1." doubling); §7 numbering matches the E7 decision.
  - Each ASCII diagram (lines 51/130/173/235/298/420) fits within the right margin and isn't clipped; text after each returns to normal size.
  - Every table (esp. the 24-query benchmark, ablation, challenge) fits page width and doesn't split awkwardly — if any overflows, wrap it in `\small{}` or add a `{tabular}` column spec / `\resizebox`.
  - All 6 figures render with a "Figure N" caption and sit near their reference.
  - Appendices A and B each start on a new page; no orphaned headings at page bottoms.
- G4. **Pre-submit naturalness scan (read-only grep):** confirm no emojis / unicode arrows / checkmarks crept in, and AI-filler phrases stay near-zero, e.g.:
  `rg -n "→|⇒|✓|✗|⚠|❌|✅" report.md` (expect none) and `grep -niE "worth noting|importantly,|moreover|furthermore|comprehensive|leverage|seamless|a testament to" report.md` (expect ~0). Spot-read any rewritten paragraph aloud for run-on sentences.

---

## If time is very short (minimal faithful path)
Do **C1, C2, C3, C4, D1** (severe faithfulness/logic) → **E1, E4, E9, E10, E11** (title + honest failure case + the rubric-required contributions/references/metric justification) → **G** (rebuild PDFs). Skip Phase B (report the honest 3/6), skip F. This removes every false claim, adds the high-value honest failure analysis, and closes the easy-points deliverables the rubric explicitly requires.

## Rough time budget
- **With Phase B (code fix + full re-sync):** A 0.25h · **B ≈ 2–3h** (1–2h compute + ~1h re-sync of all §5–§6 + verify_facts) · C→folded into B's re-sync · D 0.75h · E 2h · F 0.5h · G 0.75h → **~6.5–8h, tight.**
- **Report-only (recommended if time is uncertain):** A 0.1h · C 1h · D 0.75h · E 2h · F 0.5h · G 0.75h → **~5h, comfortable**, zero re-run risk.
Front-load C/D/E either way; PDF build (G) is always last.
