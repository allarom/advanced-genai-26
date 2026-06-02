# Implementation Plan: Multi-Agent RAG System
## Steps 2, 3, 4 | 5 Weeks | 3 Team Members

---

### What We Have So Far (from Step 1)

The `multi-agent-step-1.ipynb` notebook our colleague put together gives us a great head start:
- Working BM25, Dense, and GraphRAG retrievers that all use the exact same `search(query, top_k)` method.
- Hybrid retrieval (where it combines all 3 agents using RRF fusion).
- Re-ranking (a simple overlap-based method applied after fusion).
- Basic examples of orchestration (Voting, Waterfall, Confidence) along with their evaluation scores.
- A shared pipeline for testing using `pytrec_eval`.
- Baseline test results for both the subsample and the full corpus.

The original provided notebooks (`Step_2_*.ipynb`, `Step_3_*.ipynb`) also have some skeleton code for orchestrating strategies, generating answers, and doing bonus tasks. We can use those for ideas, but we really need to **write our own clean code** where each agent is separated properly.

---

### Timeline Overview

Person A and Person B work **in parallel** from Week 1. This is possible because:
- The Step 1 notebook (`multi-agent-step-1.ipynb`) already has working `bm25_retriever.search()`, `dense_retriever.search()`, and `graph_retriever.search()` functions
- Person B uses these existing functions as temporary stand-ins while Person A builds the proper agent classes around them
- Both sides agree on the interface signatures on Day 1 (30 min kickoff)

| Week | Person A | Person B | Person C |
|------|----------|----------|----------|
| **Day 1** | All three: agree on agent interfaces (see below) | | |
| **Week 1-2** | Build Agent Classes (Query Understanding, Fusion, Re-Ranker, Answer Synth, Critic) | **Dongyuan:** Branch A orchestration strategy implementation (primary) + optional evaluation support this week | **Julia:** Branch B orchestration strategy implementation (primary), then evaluation setup/visualization |
| **Week 2 end** | Integration checkpoint: swap real agents into orchestration code | | |
| **Week 3** | Support + bug fixes | Support + bug fixes | Run full quantitative evaluation (P@k, Recall, MRR) |
| **Week 4** | Support qual eval + Stretch Goal A if time | Support qual eval + Stretch Goal B if time | Lead qualitative eval (complementarity, failure analysis) + start report |
| **Week 5** | Code cleanup + docs | Code cleanup + docs | Final report writing (all three contribute) |

### Day 1 Kickoff: Agree on Interfaces

Before anyone writes a single line of code, the team spends 30 minutes agreeing on these exact function signatures. This is what allows parallel work:

```python
# Every retriever agent exposes this (simplified view for planning):
class RetrieverAgent:
    def retrieve(self, query: str, top_k: int = 100) -> list: ...

# Fusion agent exposes this:
class FusionAgent:
    def fuse(self, runs: dict, weights: dict) -> list: ...

# Re-ranker exposes this:
class ReRankerAgent:
    def rerank(self, query: str, docs: list, top_k: int = 10) -> list: ...
```

> **Note:** The actual Step 2 code uses `run(self, state)` instead of `retrieve()`/`fuse()` directly. See section 1.2 for the real pattern. The signatures above are the *conceptual* interface — the orchestrator calls them through `AgentState` in practice.

Person B writes orchestration code using these conceptual signatures first (easier to test). Person A ensures the real `run(state)` methods behave the same way. At Week 2 integration, the logic transfers cleanly.

---

### Team Role Assignment

| Role | Scope | Why |
|------|-------|-----|
| **Person A** | Agent Classes (Query Understanding, Fusion, Re-Ranker, Answer Synth, Critic) | Builds the modular building blocks |
| **Person B (Dongyuan)** | Orchestration Strategy in Branch A + optional evaluation support this week | Wires agents into one strategy workflow and can help unblock early evaluation |
| **Person C (Julia)** | Orchestration Strategy in Branch B + Evaluation Pipeline + Visualizations + Report Writing | Implements second strategy in parallel, then leads formal evaluation and reporting |

All three collaborate during integration (end of Week 2) and report writing (Week 5).

Working mode note: Dongyuan and Julia implement different orchestration strategies on separate branches, then compare and merge the best parts at integration.

Branch/notebook workflow (recommended):
- Use **two separate branches** (one per person/strategy), not one shared branch.
- Use **two separate notebooks** (one per strategy), both copied from the same Step 2 base.
- Keep setup/retriever/agent sections aligned in both notebooks, and change only orchestration/evaluation blocks.
- Merge strategy results and the best logic at the Week 2 integration checkpoint.

---

## Weeks 1-2: Build Agents (Person A) + Orchestration (Person B) in Parallel

**Person A's Goal:** Take the old basic retrieval functions and turn them into clean "Agent" classes. Every agent should just have one clear job to do and use the interface we agreed on.

**Person B's Goal:** Implement **one primary orchestration strategy** (e.g., Confidence-Based Routing) and optionally start a **second distinct strategy** (e.g., Waterfall/Sequential) if time allows. Use the Step 1 retriever functions (`bm25_retriever.search()`, etc.) as placeholders for quick testing. At the end of Week 2, swap in Person A's real agent classes.

### 1.1 Query Understanding Agent

**What it does:** This agent takes the user's raw question and gets it ready. It figures out the language (EN/DE), decides what type of query it is (like factoid vs. semantic), and maybe expands short queries by throwing in some synonyms.

**Why it matters:** Different questions need different strategies. If someone asks a fact-heavy question like "When was ETH founded?", we want to lean on the BM25 keyword search. But if they ask a big-picture question like "How does ETH support sustainability?", the Dense embedding search is way better.

```python
class QueryUnderstandingAgent:
    def process(self, query: str) -> dict:
        lang = detect(query)  # 'en' or 'de'
        # Note: Step 2 notebook uses more granular types:
        #   'entity_temporal', 'entity', 'keyword', 'semantic', 'graph', 'mixed'
        # The pseudocode below is simplified; real integration should match Step 2.
        has_year = any(c.isdigit() for c in query)
        tokens = query.split()
        if query.lower().startswith('who') and has_year:
            query_type = 'entity_temporal'
        elif len(tokens) <= 6 or has_year or query.lower().startswith(('when', 'where', 'what')):
            query_type = 'keyword'
        else:
            query_type = 'semantic'
        return {
            'original': query,
            'language': lang,
            'query_type': query_type,
            'expanded': self._expand(query)  # optional synonym expansion
        }
```

### 1.2 Retriever Agents (BM25, Dense, GraphRAG)

**Status:** Already implemented in `multi-agent-step-2.ipynb`. They inherit from a shared `BaseAgent` and operate on an `AgentState` object rather than returning raw lists.

**What they do:** Each agent wraps its underlying retriever (BM25 index, dense vector store, or graph store) and writes results into the shared state under a fixed name.

```python
class BM25RetrieverAgent(BaseAgent):
    name = 'bm25_retriever'
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state, top_k=100):
        state.retrieval_by_agent['bm25'] = self.retriever.search(
            state.normalized_query, top_k=top_k
        )
        return state

class DenseRetrieverAgent(BaseAgent):
    name = 'dense_retriever'
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state, top_k=100):
        state.retrieval_by_agent['dense'] = self.retriever.search(
            state.normalized_query, top_k=top_k
        )
        return state

class GraphRetrieverAgent(BaseAgent):
    name = 'graph_retriever'
    def __init__(self, retriever):
        self.retriever = retriever

    def run(self, state, top_k=100):
        state.retrieval_by_agent['graph'] = self.retriever.search(
            state.normalized_query, top_k=top_k
        )
        return state
```

> **Learning point:** Notice all three agents share the same `run(state, top_k)` signature and write into `state.retrieval_by_agent`. This is what makes them swappable. The `AgentState` acts like a shared notebook that each agent writes into. The orchestrator reads from that notebook to fuse results.

### 1.3 Fusion Agent

**What it does:** This agent takes the lists of documents found by the different retrievers and merges them together using a math trick called Reciprocal Rank Fusion (RRF). This is basically how our teamwork happens under the hood.

**RRF Formula:** Every document gets a combined score. It's basically the sum of `weight / (k + rank)` for every agent that found it. So if a document was found by multiple agents, it gets a massive boost.

```python
class FusionAgent:
    def _uid(self, doc):
        return doc.metadata.get('chunk_id') or doc.metadata.get('record_id')

    def fuse(self, runs: dict, weights: dict, k_rrf: int = 60) -> list:
        from collections import defaultdict
        scores = defaultdict(float)
        store = {}
        for name, docs in runs.items():
            w = weights.get(name, 1.0)
            for rank, doc in enumerate(docs, 1):
                uid = self._uid(doc)
                if uid is None:
                    continue
                store[uid] = doc
                scores[uid] += w / (k_rrf + rank)
        return sorted(store.values(), key=lambda d: scores[self._uid(d)], reverse=True)
```

> **Pseudocode vs Reality:** The actual `FusionAgent` in Step 2 uses `run(self, state, top_k=30, **kwargs)` and reads from `state.retrieval_by_agent`. The `fuse()` logic above is the same math, just shown as a plain function so you can see the RRF formula clearly.

### 1.4 Re-Ranker Agent

**What it does:** Takes the fused list and re-scores the top candidates using a more expensive but accurate method. Two options:

- **Option A (Simple):** Overlap-based reranking, counting how many query words appear in each document. Already exists in Step 1 code.
- **Option B (Advanced):** Use a Cross-Encoder model from Hugging Face (`cross-encoder/ms-marco-MiniLM-L-6-v2`). This reads the query AND document together and outputs a single relevance score. Much more accurate but slower.

```python
# Option B: Cross-Encoder reranking
from sentence_transformers import CrossEncoder
class ReRankerAgent:
    def __init__(self):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    def rerank(self, query: str, docs: list, top_k: int = 10) -> list:
        pairs = [(query, doc.page_content) for doc in docs[:50]]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(docs[:50], scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]
```

> **Pseudocode vs Reality:** The actual Step 2 `ReRankerAgent` uses `run(self, state, top_k)` and has a fallback to overlap-based scoring if the Cross-Encoder is not available. It writes back into `state.reranked_docs`.

### 1.5 Answer Synthesizer Agent

**What it does:** After we find the right documents, this agent reads through them and writes out a normal, human-sounding answer using an LLM. This is basically the "Generation" part of RAG.

**Tool:** Use OpenAI API (GPT-4o-mini for cost efficiency) or a local model via `transformers`.

```python
class AnswerSynthesizerAgent:
    def synthesize(self, query: str, docs: list) -> str:
        context = "\n\n".join([d.page_content for d in docs[:5]])
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
```

> **Pseudocode vs Reality:** The actual Step 2 `AnswerSynthesizerAgent` uses `run(self, state)` and reads from `state.reranked_docs`. It writes the generated answer into `state.generated_answer`.

### 1.6 Critic Agent

**What it does:** Checks if the generated answer is actually supported by the retrieved documents. If the answer contains claims not found in the context, it flags the response as potentially hallucinated and can trigger re-retrieval.

```python
class CriticAgent:
    def verify(self, answer: str, docs: list, query: str) -> dict:
        context = "\n".join([d.page_content for d in docs[:5]])
        prompt = f"Is this answer fully supported by the context? Answer YES or NO with explanation.\n\nContext:\n{context}\n\nAnswer: {answer}"
        verdict = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        supported = 'yes' in verdict.choices[0].message.content.lower()
        return {'supported': supported, 'explanation': verdict.choices[0].message.content}
```

> **Pseudocode vs Reality:** The actual Step 2 `CriticAgent` uses `run(self, state)` and reads `state.generated_answer`. It can trigger a retry by setting a flag on the state object.

---

## Orchestration Mechanisms (Person B builds these in parallel during Weeks 1-2)

**Goal:** Implement at least 2 orchestration strategies as **swappable orchestrator classes**. Person B codes these using Step 1's existing retriever functions as stand-ins, then swaps in Person A's real agent classes at the end of Week 2.

> **Practical tip:** Always test new strategies on the `subsample` first (fast, ~30s). Only run `full_corpus` evaluation after subsample results look reasonable. This prevents wasting hours on broken code.

**Current State of Step 2:** The `multi-agent-step-2.ipynb` `MultiAgentPipeline` currently runs **all three retrievers in parallel** (BM25 + Dense + GraphRAG), then fuses, reranks, and applies a basic Critic with **one retry**. This is essentially Strategy 1 below, but it is **hardcoded** — it cannot be swapped with other strategies without rewriting the pipeline. What is **missing** as reusable, swappable classes are: Confidence-Based Routing, Waterfall/Sequential, and a full multi-retry Critic Loop.

> **Clarification:** The Step 2 notebook mentions "sequential orchestration" in the `MultiAgentPipeline` explanation. This refers to the *order of pipeline stages* (Query → Retrieve → Fusion → …), not the "Sequential/Waterfall" routing strategy where retrievers are run conditionally one after another.

### 2.1 Strategy 1: Parallel + Fusion (Voting/Ensemble) — Already in Step 2, Needs Refactoring

**What it does:** It just blasts the query out to all the retriever agents at the same time, collects everything, fuses it using RRF, and maybe re-ranks it.

**When to use it:** It's a great all-around strategy if you aren't sure what kind of query you're dealing with.

```python
def parallel_orchestrate(query, agents, fusion, reranker, top_k=5):
    # Step 1: Query understanding
    q_info = query_agent.process(query)

    # Step 2: All agents retrieve in parallel
    # Map agent names to weight keys (e.g., 'bm25_retriever' -> 'bm25')
    name_map = {'bm25_retriever': 'bm25', 'dense_retriever': 'dense', 'graph_retriever': 'graph'}
    runs = {name_map.get(agent.name, agent.name): agent.retrieve(q_info['expanded'], top_k=100) for agent in agents}

    # Step 3: Fuse
    weights = {'bm25': 1.2, 'dense': 1.0, 'graph': 0.6}
    fused = fusion.fuse(runs, weights)

    # Step 4: Re-rank top candidates
    final = reranker.rerank(query, fused, top_k=top_k)
    return final
```

> **Pseudocode vs Reality:** In the actual Step 2 notebook, this logic lives inside `MultiAgentPipeline.run()`. The pipeline creates an `AgentState`, calls `query_agent.run(state)`, then loops through retriever agents calling `agent.run(state)`, then calls `fusion.run(state)`, then `reranker.run(state)`, etc. The pseudocode above shows the same concept but with direct function calls so you can see the data flow clearly.

### 2.2 Strategy 2: Confidence-Based Routing — **Recommended for Dongyuan**

**Status:** Not yet implemented. This is a good primary strategy for Branch A because it uses the `QueryUnderstandingAgent` (already in Step 2) to dynamically adjust weights, making it clearly different from the hardcoded Parallel approach.

**What it does:** It thinks about the query first and then changes how much we trust each agent on the fly. If it's a fact question, it turns up BM25. If it's about meaning, it turns up Dense. If it's complicated, it turns to GraphRAG.

**When to use it:** Perfect for when we expect really mixed types of questions. It improves result quality by emphasizing the retrievers most likely to help for each query type. Note: the pseudocode below still runs all three retrievers — to truly skip one, set its weight to 0.0 in `fusion.fuse(...)` or don't include it in the `runs` dict.

> **Performance tip:** GraphRAG is usually the slowest retriever. If your evaluation is taking too long, modify the Confidence strategy to **truly skip** GraphRAG on keyword/entity queries by removing it from the `runs` dict instead of just lowering its weight. This is a good way to learn the difference between "weighting down" and "gating out" an agent.

```python
def confidence_orchestrate(query, agents, fusion, reranker, top_k=5):
    q_info = query_agent.process(query)

    # Dynamic weights based on query classification
    # Use the actual types from QueryUnderstandingAgent (keyword, semantic, entity_temporal, ...)
    if q_info['query_type'] in ('keyword', 'entity_temporal'):
        weights = {'bm25': 1.5, 'dense': 0.8, 'graph': 0.4}
    elif q_info['query_type'] == 'semantic':
        weights = {'bm25': 0.8, 'dense': 1.5, 'graph': 0.8}
    else:
        weights = {'bm25': 1.0, 'dense': 1.0, 'graph': 1.0}

    # Use expanded query consistently (same as parallel strategy)
    # Map agent names to weight keys (e.g., 'bm25_retriever' -> 'bm25')
    name_map = {'bm25_retriever': 'bm25', 'dense_retriever': 'dense', 'graph_retriever': 'graph'}
    runs = {name_map.get(agent.name, agent.name): agent.retrieve(q_info['expanded'], top_k=100) for agent in agents}
    fused = fusion.fuse(runs, weights)
    final = reranker.rerank(query, fused, top_k=top_k)
    return final, {'weights': weights, 'query_type': q_info['query_type']}
```

> **Pseudocode vs Reality:** Same note as Strategy 1 — in real Step 2 code you would build an `Orchestrator` class that creates an `AgentState`, calls `query_agent.run(state)`, decides weights, then runs retrievers through `state`. The data flow is identical; only the syntax differs.

**Alternative Strategy for Branch A:** If you want a second strategy that is even more distinct from Parallel, consider **Waterfall / Sequential Routing** — run BM25 first, check result quality, and only run Dense or GraphRAG if the early results look weak. This is especially good for latency comparison in the evaluation.

### 2.3 Strategy 3 (Optional): Critic Loop (Self-Verification)

**Dependency:** This strategy requires a working LLM (for Answer Synthesizer + Critic). Person B should build strategies 2.1 and 2.2 first, then add this one after the team decides on the LLM provider (OpenAI vs local) in Week 1.

**What it does:** After the Answer Synthesizer generates an answer, the Critic Agent checks it. If the Critic says "not supported", the system re-retrieves with an expanded query and tries again (max 2 loops).

```python
def critic_loop_orchestrate(query, agents, fusion, reranker, synthesizer, critic, max_retries=2):
    for attempt in range(max_retries):
        docs = parallel_orchestrate(query, agents, fusion, reranker)
        answer = synthesizer.synthesize(query, docs)
        verdict = critic.verify(answer, docs, query)
        if verdict['supported']:
            return answer, docs, {'attempts': attempt + 1, 'verified': True}
        query = query + " " + verdict['explanation'][:50]  # expand query with critic feedback
    return answer, docs, {'attempts': max_retries, 'verified': False}
```

> **Pseudocode vs Reality:** In Step 2, the `MultiAgentPipeline` already does one retry when the Critic flags an answer. A full multi-retry Critic Loop would wrap the entire pipeline in a `for` loop and re-run from the Query Understanding stage with a modified query, just like the pseudocode above.

---

## Week 3: Quantitative Evaluation (Step 3a)

**Goal:** Measure retrieval accuracy using the same metrics as Step 1, but now comparing our new orchestration strategies against each other and against the baseline.

### 3.1 Core Metrics

Use the same `pytrec_eval` pipeline from Step 1. For each strategy, loop through all 24 benchmark questions and collect:

| Metric | What It Measures |
|--------|-----------------|
| P@1, P@3, P@5, P@10 | Precision at different cutoffs |
| R@1, R@3, R@5, R@10 | Recall at different cutoffs |
| MRR | Position of first correct document |
| nDCG@5, nDCG@10 | Quality of ranking order |

### 3.2 Efficiency Metrics

Wrap each orchestration call in a timer:

```python
import time
latencies = []
for q in qa_data:
    start = time.time()
    docs = orchestrator(q['question'], top_k=10)
    latencies.append(time.time() - start)

print(f"Avg latency: {np.mean(latencies):.3f}s")
print(f"P95 latency: {np.percentile(latencies, 95):.3f}s")
```

### 3.3 Statistical Significance

Use a paired t-test to prove one strategy is genuinely better than another (not just random luck):

```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(scores_strategy_A, scores_strategy_B)
print(f"p-value: {p_value:.4f}, Significant: {p_value < 0.05}")
```

### 3.4 Visualization

Use `matplotlib` and `seaborn` to create:
- Bar chart comparing MRR across all strategies (baseline + new)
- Box plots showing per-query score distributions
- Latency comparison bar chart (avg vs P95)
- Heatmap of P@k and R@k across methods

---

## Week 4: Qualitative Evaluation + Stretch Goals (Step 3b)

**Goal:** Understand *why* strategies win or lose, not just *which* one wins. This is often the most educational part of the project because you get to inspect real mistakes.

### 4.1 Core: Agent Complementarity Analysis (Must do)

**What it is:** Measure how much overlap exists between the three retrievers. If BM25 and Dense return completely different documents, they are highly complementary (good). If they return the same documents, one of them is redundant.

**Why it matters for learning:** This teaches you whether "adding more retrievers" is actually useful or just overhead. It also helps you debug your Confidence strategy — if two retrievers always return the same top-3 docs, your weighting won't matter.

### 4.2 Core: Failure Analysis (Must do)

Identify queries where the system scores 0 on MRR. Categorize them:
- Was the query in German? (language issue)
- Was the query too vague? (query understanding issue)
- Was the correct document missing from the corpus? (data issue)
- Did the Confidence strategy assign a bad weight? (orchestration issue)

**Why it matters for learning:** This turns your evaluation from numbers into actionable fixes. If 80% of failures are German queries, you know Person A should improve language detection.

### 4.3 Stretch Goal A: Explainability (Easy bonus, ~4 hrs)

Make the orchestrator print a human-readable rationale for every decision. This teaches prompt engineering and human-AI communication.

```python
# Example output:
# "Query classified as KEYWORD (starts with 'When').
#  BM25 weight boosted to 1.5. Dense reduced to 0.8.
#  GraphRAG skipped (slow, low relevance for temporal fact).
#  Top document found by BM25 at rank 1."
```

> **Pick this if:** You finished core evaluation early and want easy bonus points. It does not require new models — just logging.

### 4.4 Stretch Goal B: Adaptive Orchestration with RL (Medium bonus, ~8 hrs)

Use Q-learning to let the orchestrator learn which strategy works best for which query type. After each query, update the Q-table based on whether the retrieval was successful.

```python
from collections import defaultdict
import random

class AdaptiveOrchestrator:
    def __init__(self):
        self.q_table = defaultdict(lambda: {'parallel': 0, 'confidence': 0, 'critic': 0})
        self.epsilon = 0.2  # exploration rate
        self.lr = 0.3

    def choose_strategy(self, query_type):
        if random.random() < self.epsilon:
            return random.choice(['parallel', 'confidence', 'critic'])
        return max(self.q_table[query_type], key=self.q_table[query_type].get)

    def update(self, query_type, strategy, reward):
        old = self.q_table[query_type][strategy]
        self.q_table[query_type][strategy] = old + self.lr * (reward - old)
```

> **Pick this if:** You want to learn RL basics and have at least 3 full days left. It is modular — it does not depend on 4.3.

**How to choose stretch goals:**
1. Finish 4.1 and 4.2 first. They are the highest-value core tasks.
2. If you have >2 days left, do 4.3 (Explainability).
3. If you have >3 days left and enjoy math, try 4.4 (RL).
4. If core evaluation is still broken by Week 3, skip both. A working core system scores higher than a broken system with half-finished bonuses.

---

## Week 5: Final Report + Polish (Step 4)

### 5.1 Report Structure (15 Points)

| Section | Content | Points |
|---------|---------|--------|
| **Clarity & Structure** | Clear sections, logical flow, proper tables | 5 |
| **Critical Reflection** | What worked, what didn't, lessons learned, limitations | 5 |
| **Professionalism** | Clean code, documentation, reproducibility | 5 |

### 5.2 Suggested Report Outline

1. **Introduction:** Problem statement, dataset description (ETH Zurich corpus, bilingual)
2. **Step 1 Summary:** Baseline reproduction results (reference our existing report)
3. **Step 2: Multi-Agent Design:** Agent descriptions, architecture diagram, orchestration strategy explanations
4. **Step 3: Evaluation:** All metric tables, charts, statistical tests, failure analysis
5. **Bonus Features:** Explainability demo, adaptive RL results (if implemented)
6. **Critical Reflection:** Strengths, weaknesses, what we would do differently, scalability concerns
7. **Appendix:** Code snippets, full metric tables

### 5.3 Code Cleanup Checklist

- [ ] All notebooks run top-to-bottom without errors
- [ ] Comments explain the "why", not just the "what"
- [ ] Remove debug prints and unused cells
- [ ] Consistent variable naming
- [ ] README updated with our additions

---

## Key Decisions to Make as a Team

1. **LLM Choice for Answer Synthesis:** OpenAI API (costs money but easy) vs. local model via Hugging Face (free but slower). Decide before Week 1 ends.
2. **Cross-Encoder for Re-Ranking:** Using one improves accuracy significantly but adds ~2s per query. Worth it for the grade.
3. **Stretch Goal Priority:** Week 4 explains how to pick between Explainability (easy, teaches logging/prompt design) and Adaptive RL (medium, teaches Q-learning). Both are optional but modular.
4. **Subsample vs Full Corpus:** Develop on subsample. Final evaluation on full corpus. Always report both.
