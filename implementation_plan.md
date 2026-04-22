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
- The Step 1 notebook already has working `bm25_retriever.search()`, `dense_retriever.search()`, and `graph_retriever.search()` functions
- Person B uses these existing functions as temporary stand-ins while Person A builds the proper agent classes around them
- Both sides agree on the interface signatures on Day 1 (30 min kickoff)

| Week | Person A | Person B | Person C |
|------|----------|----------|----------|
| **Day 1** | All three: agree on agent interfaces (see below) | | |
| **Week 1-2** | Build Agent Classes (Query Understanding, Fusion, Re-Ranker, Answer Synth, Critic) | Build 2-3 Orchestration Strategies using Step 1 retrievers as stand-ins | Set up evaluation pipeline, start visualizations |
| **Week 2 end** | Integration checkpoint: swap real agents into orchestration code | | |
| **Week 3** | Support + bug fixes | Support + bug fixes | Run full quantitative evaluation (P@k, Recall, MRR) |
| **Week 4** | Bonus: Explainability | Bonus: Adaptive RL | Qualitative eval (failure analysis, complementarity) |
| **Week 5** | Code cleanup + docs | Code cleanup + docs | Final report writing (all three contribute) |

### Day 1 Kickoff: Agree on Interfaces

Before anyone writes a single line of code, the team spends 30 minutes agreeing on these exact function signatures. This is what allows parallel work:

```python
# Every retriever agent exposes this:
class RetrieverAgent:
    def retrieve(self, query: str, top_k: int = 100) -> list: ...

# Fusion agent exposes this:
class FusionAgent:
    def fuse(self, runs: dict, weights: dict) -> list: ...

# Re-ranker exposes this:
class ReRankerAgent:
    def rerank(self, query: str, docs: list, top_k: int = 10) -> list: ...
```

Person B writes orchestration code calling these exact signatures. Person A builds the real implementations behind them. At Week 2 integration, swapping in the real agents is a one-line change.

---

### Team Role Assignment

| Role | Scope | Why |
|------|-------|-----|
| **Person A** | Agent Classes (Query Understanding, Fusion, Re-Ranker, Answer Synth, Critic) | Builds the modular building blocks |
| **Person B** | Orchestration Strategies (Parallel, Confidence, Critic Loop) | Wires agents into workflows using Step 1 stand-ins first |
| **Person C** | Evaluation Pipeline + Visualizations + Report Writing | Measures everything and writes it up |

All three collaborate during integration (end of Week 2) and report writing (Week 5).

---

## Weeks 1-2: Build Agents (Person A) + Orchestration (Person B) in Parallel

**Person A's Goal:** Take the old basic retrieval functions and turn them into clean "Agent" classes. Every agent should just have one clear job to do and use the interface we agreed on.

**Person B's Goal:** Start coding 2-3 orchestration strategies right away. You can use the old Step 1 retriever functions (`bm25_retriever.search()`, etc.) as placeholders. Then, at the end of Week 2, we just swap those out for Person A's real agent classes.

### 1.1 Query Understanding Agent

**What it does:** This agent takes the user's raw question and gets it ready. It figures out the language (EN/DE), decides what type of query it is (like factoid vs. semantic), and maybe expands short queries by throwing in some synonyms.

**Why it matters:** Different questions need different strategies. If someone asks a fact-heavy question like "When was ETH founded?", we want to lean on the BM25 keyword search. But if they ask a big-picture question like "How does ETH support sustainability?", the Dense embedding search is way better.

```python
class QueryUnderstandingAgent:
    def process(self, query: str) -> dict:
        lang = detect(query)  # 'en' or 'de'
        tokens = query.split()
        query_type = 'factoid' if len(tokens) <= 6 or any(c.isdigit() for c in query) else 'semantic'
        return {
            'original': query,
            'language': lang,
            'query_type': query_type,
            'expanded': self._expand(query)  # optional synonym expansion
        }
```

### 1.2 Retriever Agents (BM25, Dense, GraphRAG)

**What they do:** These already exist from Step 1. We just wrap them into a consistent `RetrieverAgent` interface so the orchestrator can treat them identically.

```python
class RetrieverAgent:
    def __init__(self, name: str, retriever):
        self.name = name
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 100) -> list:
        return self.retriever.search(query, top_k=top_k)
```

### 1.3 Fusion Agent

**What it does:** This agent takes the lists of documents found by the different retrievers and merges them together using a math trick called Reciprocal Rank Fusion (RRF). This is basically how our teamwork happens under the hood.

**RRF Formula:** Every document gets a combined score. It's basically the sum of `weight / (k + rank)` for every agent that found it. So if a document was found by multiple agents, it gets a massive boost.

```python
class FusionAgent:
    def _uid(self, doc):
        return doc.metadata.get('chunk_id') or doc.metadata.get('record_id')

    def fuse(self, runs: dict, weights: dict, k_rrf: int = 60) -> list:
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

---

## Orchestration Mechanisms (Person B builds these in parallel during Weeks 1-2)

**Goal:** Implement at least 2 orchestration strategies. Person B codes these using Step 1's existing retriever functions as stand-ins, then swaps in Person A's real agent classes at the end of Week 2.

### 2.1 Strategy 1: Parallel + Fusion (Voting/Ensemble)

**What it does:** It just blasts the query out to all the retriever agents at the same time, collects everything, fuses it using RRF, and maybe re-ranks it.

**When to use it:** It's a great all-around strategy if you aren't sure what kind of query you're dealing with.

```python
def parallel_orchestrate(query, agents, fusion, reranker, top_k=5):
    # Step 1: Query understanding
    q_info = query_agent.process(query)

    # Step 2: All agents retrieve in parallel
    runs = {agent.name: agent.retrieve(q_info['expanded'], top_k=100) for agent in agents}

    # Step 3: Fuse
    weights = {'bm25': 1.2, 'dense': 1.0, 'graph': 0.6}
    fused = fusion.fuse(runs, weights)

    # Step 4: Re-rank top candidates
    final = reranker.rerank(query, fused, top_k=top_k)
    return final
```

### 2.2 Strategy 2: Confidence-Based Routing

**What it does:** It thinks about the query first and then changes how much we trust each agent on the fly. If it's a fact question, it turns up BM25. If it's about meaning, it turns up Dense. If it's complicated, it turns to GraphRAG.

**When to use it:** Perfect for when we expect really mixed types of questions. It also saves processing time by basically ignoring agents that probably won't help.

```python
def confidence_orchestrate(query, agents, fusion, reranker, top_k=5):
    q_info = query_agent.process(query)

    # Dynamic weights based on query classification
    if q_info['query_type'] == 'factoid':
        weights = {'bm25': 1.5, 'dense': 0.8, 'graph': 0.4}
    elif q_info['query_type'] == 'semantic':
        weights = {'bm25': 0.8, 'dense': 1.5, 'graph': 0.8}
    else:
        weights = {'bm25': 1.0, 'dense': 1.0, 'graph': 1.0}

    # Use expanded query consistently (same as parallel strategy)
    runs = {agent.name: agent.retrieve(q_info['expanded'], top_k=100) for agent in agents}
    fused = fusion.fuse(runs, weights)
    final = reranker.rerank(query, fused, top_k=top_k)
    return final, {'weights': weights, 'query_type': q_info['query_type']}
```

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

## Week 4: Qualitative Evaluation + Bonus (Step 3b)

### 4.1 Agent Complementarity Analysis

**What it is:** Measure how much overlap exists between the three retrievers. If BM25 and Dense return completely different documents, they are highly complementary (good). If they return the same documents, one of them is redundant.

### 4.2 Failure Analysis

Identify queries where the system scores 0 on MRR. Categorize them:
- Was the query in German? (language issue)
- Was the query too vague? (query understanding issue)
- Was the correct document missing from the corpus? (data issue)

### 4.3 Explainability (Bonus - 5 pts)

Make the orchestrator print a human-readable rationale for every decision:

```python
# Example output:
# "Query classified as FACTOID (contains year '2003').
#  BM25 weight boosted to 1.5. Dense reduced to 0.8.
#  Top document found by BM25 at rank 1."
```

### 4.4 Adaptive Orchestration with RL (Bonus - 5 pts)

Use Q-learning to let the orchestrator learn which strategy works best for which query type. After each query, update the Q-table based on whether the retrieval was successful.

```python
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
3. **Bonus Feature Priority:** Explainability (easiest, ~5 hrs) and Adaptive RL (moderate, ~8 hrs) are the best return on effort. Human-in-the-loop and Adversarial Queries are more time-consuming.
4. **Subsample vs Full Corpus:** Develop on subsample. Final evaluation on full corpus. Always report both.
