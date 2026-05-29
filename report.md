---
output:
  pdf_document: default
  html_document: default
---

# Reliable Adaptive Agentic RAG System
## Steps 1-3 Report | Advanced Generative AI Capstone

**Date:** May 2026

---

## Executive Summary

This report documents the design, implementation, and evaluation of a **Reliable Adaptive Agentic RAG system** built on top of a high-performing baseline retrieval pipeline. The project progresses through four logical stages:

1. **Baseline Reproduction** — Reproduce and verify BM25, Dense, GraphRAG, Hybrid, and ReRank methods.
2. **Multi-Agent Retrieval (Legacy Step 2)** — Implement and compare orchestration strategies (Confidence, Waterfall, Voting).
3. **Reliability-Aware Design (New Step 2)** — Design architecture for evidence sufficiency, groundedness, contradiction detection, trust scoring, and adaptive recovery.
4. **Reliable Adaptive Implementation (Step 3)** — Code 8 reliability agents that wrap the legacy retrieval engine.

The core insight: **retrieval quality is necessary but not sufficient for trustworthy answers.** We wrap the best-performing orchestration strategy (Confidence, MRR ~0.209) with a reliability layer that decides when to answer, abstain, recover, or clarify.

---

## 1. Baseline Reproduction (Step 1)

### 1.1 What the baseline does

```
User Query
    |
    v
+-------------------------+
|  BM25 Retriever         |--> Keyword-based exact match
|  (lexical)              |
+-------------------------+
    |
    v
+-------------------------+
|  Dense Retriever        |--> Semantic vector search (E5 embeddings)
|  (semantic)             |
+-------------------------+
    |
    v
+-------------------------+
|  GraphRAG Retriever     |--> Community-based graph retrieval
|  (relational)           |
+-------------------------+
```

The system uses three different search agents:

- **BM25 (Keyword Agent):** Counts rare word frequencies for ranking.
- **Dense Retriever (Meaning Agent):** Uses `multilingual-e5-large-instruct` embeddings to capture semantic similarity.
- **GraphRAG (Detective Agent):** Uses community detection over a knowledge graph to find connected evidence.

All retrievers expose a unified `search(query, top_k)` interface via adapter classes. Evaluation uses shared `Precision@k`, `Recall@k`, and `MRR` metrics computed with `pytrec_eval`.

### 1.2 Baseline Results (Full Corpus, 24 queries)

| Method | MRR | Precision@1 | Precision@5 | Recall@10 |
|--------|-----|-------------|-------------|-----------|
| GraphRAG | **0.233** | 0.083 | 0.117 | 0.053 |
| ReRank | 0.223 | 0.042 | 0.125 | 0.011 |
| Hybrid | 0.202 | 0.000 | 0.125 | 0.031 |
| Dense | 0.166 | 0.042 | 0.058 | 0.034 |
| BM25 | 0.151 | 0.042 | 0.092 | 0.011 |

**Key finding:** GraphRAG achieved the best overall ranking quality on the full corpus. However, all methods drop significantly from subsample (817 chunks) to full corpus (7,531 chunks), confirming that full-corpus evaluation is essential for realistic assessment.

### 1.3 Reproducibility Verification

| Aspect | Original Report | Reproduced | Status |
|--------|----------------|------------|--------|
| Subsample size | 817 chunks | 817 chunks | Match |
| Full-corpus size | ~7,544 chunks | 7,531 chunks | Near match |
| Full-corpus Dense MRR | 0.166 | 0.166 | Exact match |
| Subsample to full trend | Metrics decrease | Metrics decrease | Match |

---

## 2. Multi-Agent Retrieval System (Legacy Step 2)

### 2.1 Why move from single retriever to multi-agent?

Single retrievers are brittle. BM25 fails on semantic questions; Dense misses exact entity matches; GraphRAG is expensive. A multi-agent system combines them adaptively.

### 2.2 Agent architecture

```
User Query
    |
    v
+-------------------------+
| QueryUnderstandingAgent |--> Classifies: entity_temporal / semantic / keyword / graph / mixed
|                         |    Sets dynamic weights for retrievers
+-------------------------+
    |
    v
+-------------------------+
| Retriever Agents        |
| - BM25RetrieverAgent    |
| - DenseRetrieverAgent   |
| - GraphRetrieverAgent   |
+-------------------------+
    |
    v
+-------------------------+
| FusionAgent             |--> Weighted RRF merge + deduplication
+-------------------------+
    |
    v
+-------------------------+
| ReRankerAgent           |--> CrossEncoder re-ranking
+-------------------------+
    |
    v
+-------------------------+
| AnswerSynthesizerAgent  |--> Extractive sentence scoring
|                         |    (overlap + temporal bonus), top-3 pick
+-------------------------+
    |
    v
+-------------------------+
| CriticAgent             |--> Grounding check (45% global overlap)
|                         |    Temporal coherence (+-10 years)
|                         |    Triggers re-retrieval if failed
+-------------------------+
    |
    v
+-------------------------+
| Fallback: honest "not found" if still ungrounded after retry
+-------------------------+
```

### 2.3 Orchestration strategies compared

Three strategies were implemented and evaluated:

| Strategy | How it works | Full-corpus MRR |
|----------|-------------|-----------------|
| **Confidence** | Query classification, dynamic weights, gated retrievers, critic retry | **0.209** |
| **Waterfall** | Escalate BM25 → +Dense → +Graph on critic failure | 0.208 |
| **Voting** | Equal-weight parallel retrievers, merged with RRF | 0.202 |

**Confidence was selected as the default** because:
1. Highest MRR on full corpus.
2. Built-in query gating reduces unnecessary retriever calls.
3. Existing Critic retry loop naturally extends to the new reliability layer.
4. Most adaptive -- weights change per query type.

---

## 3. Reliability-Aware Design (New Step 2)

### 3.1 Why a reliability layer?

The old system produces an answer, but **does not ask: "Can I trust this answer?"** The new design adds a wrapper that judges answer quality before returning it to the user.

### 3.2 Architecture: The reliability layer wraps the old engine

```
+-------------------------------------------------------------+
|                     RELIABILITY LAYER                        |
|  (New Step 2 / Step 3 -- signals, trust, decision, recovery)|
|                                                              |
|  +---------+  +---------+  +---------+  +---------+          |
|  |Evidence |  |Grounded-|  |Contra-  |  |Clarifi- |          |
|  |Suffici-|  |ness     |  |diction  |  |cation   |          |
|  |ency     |  |Agent    |  |Agent    |  |Agent    |          |
|  +----+----+  +----+----+  +----+----+  +----+----+          |
|       |            |            |            |               |
|       +------------+------+-----+------------+               |
|                           |                                  |
|                    +-------------+                            |
|                    |  TrustAgent |--> trust score [0, 1]      |
|                    +------+------+                            |
|                           |                                  |
|              +------------+------------+                     |
|              |            |            |                     |
|       +----------+ +----------+ +----------+                |
|       |Abstention| | Critic   | | Recovery |                |
|       |Agent     | | Agent    | | Agent    |                |
|       +----+-----+ +----+-----+ +----+-----+                |
|            |            |            |                        |
|            +------------+----+-----+                        |
|                               |                               |
|                    +-----------------+                        |
|                    |  Decision Policy |                        |
|                    |  clarify? -> compute trust -> [recover] |
|                    |  -> answer or abstain                   |
|                    +-----------------+                        |
+-------------------------------------------------------------+
                              |
                              v
              +-------------------------+
              |  Legacy Step 2 Engine   |
              |  (Confidence Orchestrator)
              |  --> retrieves, fuses, re-ranks, synthesizes
              +-------------------------+
```

**Each agent has one responsibility:**

| Agent | Signal / Output | Decision criterion |
|-------|----------------|--------------------|
| `EvidenceSufficiencyAgent` | `{"sufficient": bool, "score": float}` | Enough relevant docs? |
| `GroundednessAgent` | `True/False` | Answer supported by docs? |
| `ContradictionAgent` | `{"contradiction": bool, "reason": str}` | Docs disagree? |
| `ClarificationAgent` | `{"needs_clarification": bool, "question": str}` | Query ambiguous? |
| `CriticAgent` | List of feedback strings | Quality issues (temporal, etc.) |
| `TrustAgent` | `{"score": float}` | Combined confidence [0, 1] |
| `AbstentionAgent` | `True/False` | Trust below 0.4? |
| `RecoveryAgent` | `{"action": "switch_strategy"/"rewrite_query"/"none"}` | How to fix low trust? |

### 3.3 Decision policy

The system makes decisions in this order:

```
1. CLARIFY   --> if query is ambiguous (short / pronouns / vague)
                Return: {"decision": "clarify", "question": "..."}
                (No retrieval happens for unclear queries.)

2. RETRIEVE + CHECK SIGNALS  --> run the legacy engine, then compute
                sufficiency, groundedness, contradiction, and trust score.

3. RECOVER   --> if trust < 0.4 AND recovery is NOT ablated
                - contradiction detected  --> switch strategy to "voting"
                - not grounded             --> switch strategy to "waterfall"
                - not sufficient           --> rewrite query with context
                Then re-retrieve once and re-evaluate.

4. ANSWER    --> if trust >= 0.4 after first check OR after successful recovery
                Return: {"decision": "answer", "final_answer": "..."}

5. ABSTAIN   --> if trust < 0.4 AND recovery failed or was disabled
                Return: {"decision": "abstain", "reason": "..."}
```

**Important:** Recovery is attempted *before* giving up. The system tries to fix the problem once. Only if the retry still fails does it abstain.

### 3.4 Unified trace schema

Every run returns the same dict structure for reproducibility and debugging:

```python
{
    "decision": "answer" | "abstain" | "clarify",
    "reason": "human-readable explanation",
    "signals": {
        "sufficiency": {"sufficient": bool, "score": float},
        "grounded": bool,
        "contradiction": {"contradiction": bool, "reason": str},
        "trust": {"score": float},
    },
    "intermediate": {
        "strategy_used": str,
        "recovery_action": str | None,
        "retry_count": int,
    },
    "final_answer": str | None,
    "trace_log": ["step-by-step log"],
}
```

---

## 4. Implementation Details (Step 3)

### 4.1 How the legacy engine is reused

The reliability layer does **not** rewrite retrieval. It loads the legacy Step 2 notebook via `%run` and wraps its orchestrator classes:

```python
%run multi-agent-step-2_strategy-A.ipynb

def confidence_orchestrate(query, top_k=5):
    answer, docs, trace = orchestrator.run(query, top_k=top_k)
    return answer, docs, trace
```

This ensures:
- **No regression**: Proven retrieval pipeline stays intact.
- **Modular testing**: Baseline and reliability-augmented runs are side-by-side comparable.
- **Clear ablation**: `ReliableAdaptiveRAG(ablate=["groundedness"])` disables checks individually.

### 4.2 Signal implementations

All signals use **lightweight heuristics** for interpretability and debuggability:

| Signal | Implementation | Threshold |
|--------|---------------|-----------|
| Sufficiency | Token overlap between query and top-5 docs | score >= 0.3 |
| Groundedness | 20% token overlap between answer and any single doc | overlap >= 0.20 |
| Contradiction | Keyword-pair matching + year conflict detection | any match |
| Trust | `0.6*sufficiency + 0.3*groundedness - 0.4*contradiction` | clip to [0, 1] |
| Abstention | `trust_score < 0.4` | threshold = 0.4 |

**Note:** These are intentionally simple. Each has a documented upgrade path to LLM-based or learned alternatives (see Limitations).

### 4.3 Recovery logic

```python
RecoveryAgent.recover(query, strategy, sufficiency, contradiction, grounded):
    if contradiction["contradiction"]:
        return {"action": "switch_strategy", "new_strategy": "voting"}
    if not grounded:
        fallback = "waterfall" if strategy != "waterfall" else "voting"
        return {"action": "switch_strategy", "new_strategy": fallback}
    if not sufficiency["sufficient"]:
        return {"action": "rewrite_query", "query": query + " ETH Zurich"}
    return {"action": "none"}
```

This mirrors the old CriticAgent's "broaden retrieval" retry, but at the orchestration level rather than the retriever-weight level.

---

## 5. Evaluation

### 5.1 Quantitative: Baseline vs. Orchestration vs. Reliability-Augmented

| Scope | Method | MRR | Best Baseline |
|-------|--------|-----|---------------|
| full_corpus | GraphRAG | 0.233 | Yes |
| full_corpus | Confidence | 0.209 | -- |
| full_corpus | ReliableAdaptiveRAG (confidence) | pending full eval | -- |

**Note:** The reliability layer adds abstention and recovery overhead. Primary metric shifts from retrieval MRR to **trust score distribution**, **abstention rate**, and **recovery success rate**.

### 5.2 Qualitative: Failure modes caught by reliability layer

| Query Type | Failure | Agent that catches it | Action |
|------------|---------|----------------------|--------|
| "President in 2003?" | Answer mentions 2015 | CriticAgent (temporal) | Switch strategy |
| "Did funding increase?" | Doc A: yes, Doc B: no | ContradictionAgent | Switch to voting |
| "What is it?" | Query too vague | ClarificationAgent | Ask to clarify |
| Off-topic query | No relevant docs | EvidenceSufficiencyAgent | Abstain |

### 5.3 Ablation study

Using `ReliableAdaptiveRAG(ablate=[...])`, we can measure the contribution of each agent:

```python
rag = ReliableAdaptiveRAG(ablate=["contradiction", "critic"])
# Runs with neutral defaults: contradiction=False, critique=[]
```

This isolates the true impact of each check on the final decision.

---

## 6. Limitations & Future Work

### Evidence Sufficiency
- **Current:** Token overlap between query and top-5 docs.
- **Upgrade:** Semantic coverage scoring or LLM-based assessment.

### Groundedness
- **Current:** 20% single-document token overlap.
- **Upgrade:** NLI (Natural Language Inference) model for entailment.

### Contradiction Detection
- **Current:** Keyword-pair matching (`yes/no`, `increase/decrease`) and year conflict.
- **Upgrade:** LLM-based semantic contradiction detection.

### Trust Scoring
- **Current:** Hand-tuned linear formula `0.6*s + 0.3*g - 0.4*c`.
- **Upgrade:** Calibration on labeled data or learned weights.

### Clarification
- **Current:** Pronoun and length heuristic.
- **Upgrade:** LLM for entity disambiguation and intent clarification.

### Answer Generation
- **Current:** Orchestrator's extractive synthesizer or first-doc truncation.
- **Upgrade:** Full generative LLM with grounding constraints.

---

## 7. Conclusion

We successfully:
1. **Reproduced** the baseline and confirmed GraphRAG as the best single retriever (MRR 0.233).
2. **Implemented** three orchestration strategies, selecting Confidence as the default (MRR 0.209).
3. **Designed** a reliability layer with 8 specialized agents, a deterministic decision policy, and a unified trace schema.
4. **Integrated** the reliability layer with the legacy engine via `%run` and wrapper functions.
5. **Added** ablation support, retry-aware reasoning, and temporal coherence restoration.

The architecture separates **retrieval** (produces candidates) from **reliability judgment** (decides whether to answer). This matches production patterns at major AI labs and provides a clean upgrade path from heuristics to LLM-based verification.

---

## Appendix A: Repository Structure

```
advanced-genai-26/
├── baseline_repro_report.md          # Step 1 baseline results
├── Step_1_Baseline_and_Failure_Analysis.ipynb
├── multi-agent-step-2_strategy-A.ipynb   # Legacy Step 2 (retrieval engine)
├── Step_2_Reliability_Aware_Design.ipynb   # New Step 2 (design document)
├── Step_3_Reliable_Adaptive_Agentic_RAG.ipynb  # Step 3 (implementation)
├── scripts/                          # Patch and test utilities
│   ├── extract_pdf.py
│   ├── patch_step3_run.py
│   ├── fix_recovery_flow.py
│   └── test_recovery_flow.py
└── report.md                         # This report
```

---

## Appendix B: Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Why Confidence as default? | Highest MRR, built-in gating, existing retry loop |
| Why 8 separate agents? | Single responsibility, clean trace logs, easy ablation |
| Why heuristics over LLMs? | Deterministic, debuggable, fast; LLMs reserved for future upgrade |
| Why abstain at trust < 0.4? | Configurable threshold from Step 2 design doc |
| Why only one recovery retry? | Prevents infinite loops; matches old CriticAgent behavior |

---

## Appendix C: Generative AI Usage Declaration

This report and the accompanying code documentation were developed with assistance from a generative AI coding assistant (Cascade, an agentic AI pair-programmer). The AI was used for:

- **Explaining existing code**: Clarifying how the legacy Step 2 orchestration and CriticAgent work.
- **Architecture visualization**: Creating ASCII flowcharts to explain agent interactions and the decision policy.
- **Code review and debugging**: Identifying bugs (e.g., RecoveryAgent running at wrong time, missing `query_type` threading, over-escaped regex) and suggesting fixes.
- **Documentation drafting**: Structuring and writing this report and the README.md based on the actual codebase.
- **Design reasoning**: Discussing trade-offs (heuristics vs. LLMs, agent separation vs. merging, ablation defaults).

All code changes were reviewed and accepted by the authors. The AI did not have access to private data or external APIs beyond the project's own files. The core algorithms, design decisions, and evaluation results are the authors' own work.
