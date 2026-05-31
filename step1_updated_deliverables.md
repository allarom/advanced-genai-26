# Step 1 Deliverables (Updated Requirements)

## Verified Baseline
- Baseline notebook: `Step_1_Baseline_and_Failure_Analysis.ipynb`
- Reference baseline configuration: Full corpus + Confidence orchestration (MRR 0.208827) as reference baseline
- Baseline components verified in notebook: retrieval (BM25/Dense/GraphRAG), orchestration (Confidence/Waterfall/Voting), re-ranking, answer synthesis.

## Baseline Efficiency Indicators
- Runs analyzed: 24
- Avg runtime per query: 0.337s
- P95 runtime per query: 0.508s
- Decision split: {'answer': 12, 'abstain': 9, 'clarify': 3}
- Strategy usage: {'confidence': 12, 'voting': 12}
- Approximate cost proxy: low (heuristic pipeline + no per-query generative model expansion in this output).

## Structured Failure Taxonomy
- retrieval_failure: 5
  - example: what did prof. schubert say about flying?
  - example: how would you make ferzlizer without carbon emissions?
- ranking_failure: 4
  - example: who at eth received erc grants?
  - example: when did the insight get to mars?
- synthesis_failure: 0
- grounding_failure: 0
- ambiguity_failure: 3
  - example: what is e-sling?
  - example: how much of eth’s electricity consumpzon is due to compuzng? how did that develop over the years?
- contradiction_failure: 9
  - example: who at eth received erc grants?
  - example: when did the insight get to mars?
- orchestration_failure: 9
  - example: who at eth received erc grants?
  - example: when did the insight get to mars?
- overconfidence_failure: 0

## Motivation for Proposed Extensions
- Ambiguity + contradiction + orchestration failures indicate need for stronger query clarification, contradiction-aware retrieval routing, and recovery policy improvements.
- Retrieval/ranking failures motivate adaptive weighting, better reranking features, and stricter evidence sufficiency thresholds before synthesis.
- These findings justify reliability agents (clarification, contradiction detection, trust scoring, abstention) as primary extension targets.
