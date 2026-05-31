# Cross-Step Alignment Note (Step 1 -> Step 4)

## Purpose
Make Step 2, Step 3, and Step 4 narrative consistent with the updated Step 1 baseline and failure analysis.

## Canonical Step 1 Reference (must stay fixed)
Use this baseline everywhere as the anchor unless explicitly stated otherwise:
- Scope: **full corpus**
- Orchestration reference baseline: **Confidence**
- Baseline orchestration MRR: **0.208827**
- Best single retriever MRR: **GraphRAG 0.232573**
- Failure taxonomy counts:
  - retrieval: 5
  - ranking: 4
  - ambiguity: 3
  - contradiction: 9
  - orchestration: 9

Source files:
- `baseline_updated_requirements_report.md`
- `step1_updated_deliverables.md`

## Why mismatches currently appear
Step 3 and Step 4 include additional runs with different settings/artifacts and report larger MRR values (e.g., ~0.3646). These are valid experiment results, but they are **not the Step 1 baseline** and must be labeled as such.

## Alignment Rules
1. Always keep two labels in text/tables:
- `Baseline (Step 1 canonical)`
- `Extended experiment (Step 3/4 run YYYY-MM-DD)`

2. Never compare numbers across runs without a config note.
Required note fields:
- corpus scope
- artifact/version source
- retriever/top-k/re-ranker settings
- whether reliability layer was active

3. In Step 2 design motivation, map mechanisms to canonical failures:
- ambiguity -> ClarificationAgent
- retrieval/ranking -> EvidenceSufficiencyAgent + RecoveryAgent
- contradiction/orchestration -> ContradictionAgent + strategy switch

4. In Step 3 results, split metrics into:
- Retrieval/orchestration quality metrics (MRR, P@k, R@k)
- Reliability decision metrics (answer/abstain/clarify, retry rate, trust distribution)

5. In Step 4, explicitly state:
- memory layer primarily targets decision quality and adaptation,
- not guaranteed to improve baseline retrieval MRR.

## Exact Text Snippets You Can Reuse

### Snippet A (for Step 3/4 result sections)
"Unless otherwise stated, Step 1 canonical baseline is full-corpus Confidence orchestration (MRR 0.208827). Any higher/lower MRR shown below comes from a different run configuration and is reported as an extended experiment, not as baseline replacement."

### Snippet B (for Step 2 motivation)
"The Step 1 failure taxonomy showed highest concentration in contradiction and orchestration failures, followed by retrieval/ranking failures. Therefore, Step 2 prioritizes contradiction-aware routing, trust-based abstention/recovery, and clarify-first handling for ambiguous queries."

### Snippet C (for Step 4 memory section)
"Step 4 memory mechanisms optimize strategy selection and recovery behavior under repeated query types. They are evaluated primarily via decision and adaptation metrics; retrieval MRR remains anchored to the Step 1/2 retrieval pipeline configuration."

## Minimal Editing Checklist for Team
- Step 2 notebook/report: add one explicit paragraph referencing canonical baseline and failure counts.
- Step 3 notebook/report: relabel non-canonical MRR tables as `extended experiment`.
- Step 4 notebook/report: add one sentence separating memory/adaptation metrics from retrieval baseline metrics.
- Final report: include one short "Metric comparability" note under evaluation methodology.

## Suggested single source of truth
Treat these as authoritative for baseline references:
- `baseline_updated_requirements_report.md`
- `step1_updated_deliverables.md`
- this file: `cross_step_alignment.md`
