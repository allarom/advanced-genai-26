---
output:
  pdf_document: default
  html_document: default
---
# Multi-Agent Orchestration for RAG Systems
## Step 1. RAG Baseline Reproduction & Learning Report

*Goal: Reproduce baseline results with BM25, Dense Retrieval, GraphRAG, Hybrid Retrieval, and re-ranking, while reporting key metrics (Precision@k, Recall@k, MRR).*

---

### Part 1: Project Flow & Concept Learnings

Before looking at the results, we wanted to really understand how the original system was built. Overall, the original RAG (Retrieval-Augmented Generation) setup essentially works like an "open-book" search engine. It reads through the project's ETH Zurich document corpus to find the best paragraphs to answer specific questions.

The system uses three different search agents for this:

- **BM25 (The Keyword Agent):** This focuses on exact text matches. It simply counts how often rare words appear to rank the documents.

- **Dense Retriever (The Meaning Agent):** This uses Hugging Face models (`multilingual-e5-large-instruct`) to turn text into vectors. This helps the system understand synonyms and the overall meaning instead of just exact words.

- **GraphRAG (The Detective Agent):** This uses a Knowledge Graph to find hidden connections across different documents.

**The Orchestrator** acts as the manager. It uses strategies like Voting, Waterfall, or Confidence to decide which agents to rely on. Then, it combines their results mathematically using Reciprocal Rank Fusion (RRF).

#### 1.1 Reproducibility Architecture

Instead of dealing with a bunch of scattered Python files from the original project, our colleague built custom "Adapter" classes. This gives all three agents the exact same `search(query, top_k)` interface. Because of this, it doesn't matter how the original data was saved (like via LangChain or plain Python). Everything loads properly now.

A random seed was also locked globally to ensure that metrics do not fluctuate across different runs:

```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
```

---

### Part 2: Results

*(The sections below keep the original comments and tables directly from the reproduction notebook.)*

> *This notebook reproduces baseline retrieval results for https://github.com/Trista1208/advanced_genAI.git (23.12.2025). We load BM25, Dense, and GraphRAG resources for a selected scope (subsample or full_corpus), then construct Hybrid and Re-ranking on top of the same retrieved candidates using consistent fusion and reranking logic. We recompute all metrics ourselves from the benchmark QA set and qrels with one shared evaluator, reporting Precision@k, Recall@k, and MRR identically for every method. This setup ensures reproducibility in our environment, keeps comparisons fair across methods, and allows transparent side-by-side analysis between subsample and full-corpus performance.*

#### Evaluation Metrics

Here is a quick refresher on what we are measuring:
- **Precision@K:** Out of the top K documents the system gave us, how many were actually correct?
- **Recall@K:** Out of all the correct documents out there, how many did the system actually find in its top K results?
- **MRR (Mean Reciprocal Rank):** This looks at the position of the first correct answer. If the first relevant document is at rank 1, the score is 1.0. If it's further down, the score drops.

---

#### 2.1 Baseline Evaluation (Full Corpus)

> *This cell runs the shared evaluation pipeline for all baseline methods using the same QA set and qrels. For each query, it collects ranked document IDs, computes Precision@k, Recall@k, and reciprocal rank, and then aggregates results into per-query and per-method summary tables. Using one evaluation function for all methods ensures the reported baseline comparison is consistent and directly comparable.*

\tiny

| method     | queries_evaluated | MRR        | Precision@1 | Recall@1   | Precision@3 | Recall@3   | Precision@5 | Recall@5   | Precision@10 | Recall@10  |
|------------|-------------------|------------|-------------|------------|-------------|------------|-------------|------------|--------------|------------|
| GraphRAG | 24 | **0.232573** | 0.083333 | 0.000687 | 0.097222 | 0.003166 | 0.116667 | 0.005892 | 0.116667 | 0.052857 |
| ReRank | 24 | 0.222952 | 0.041667 | 0.000196 | 0.138889 | 0.004480 | 0.125000 | 0.006323 | 0.100000 | 0.010619 |
| Hybrid | 24 | 0.202154 | 0.000000 | 0.000000 | 0.097222 | 0.003699 | 0.125000 | 0.028352 | 0.095833 | 0.031149 |
| Dense | 24 | 0.165525 | 0.041667 | 0.000147 | 0.069444 | 0.008953 | 0.058333 | 0.010095 | 0.066667 | 0.034097 |
| BM25 | 24 | 0.151296 | 0.041667 | 0.001016 | 0.055556 | 0.001681 | 0.091667 | 0.005506 | 0.091667 | 0.011165 |

\normalsize

> *On the full-corpus setup, GraphRAG achieved the best overall ranking quality with the highest MRR (0.233), indicating that graph-guided retrieval was most effective at placing relevant evidence early in the ranked list. ReRank and Hybrid followed closely, with ReRank slightly outperforming Hybrid on MRR, which suggests that post-fusion refinement can improve early precision in some cases. Dense and BM25 showed lower MRR, indicating weaker top-rank relevance when used alone in this setting. Overall, the results suggest that full-corpus retrieval benefits from multi-source or graph-aware methods, while single-retriever baselines remain useful reference points but are less competitive at early-rank retrieval quality.*

---

#### 2.2 Orchestration Evaluation (Full Corpus)

The three orchestration strategies are evaluated separately from the individual baselines to keep the comparison clean.

- **Confidence:** Analyzes the query first, then weights agents by question type (keyword-heavy vs. semantic).
- **Waterfall:** Starts with only BM25 and Dense. Adds GraphRAG only when the two disagree significantly.
- **Voting:** Runs all three agents in parallel and merges results using weighted Reciprocal Rank Fusion (RRF).

\tiny

| method     | queries_evaluated | MRR        | Precision@1 | Recall@1   | Precision@3 | Recall@3   | Precision@5 | Recall@5   | Precision@10 | Recall@10  |
|------------|-------------------|------------|-------------|------------|-------------|------------|-------------|------------|--------------|------------|
| Confidence | 24 | **0.208827** | 0.000000 | 0.000000 | 0.111111 | 0.003891 | 0.133333 | 0.028380 | 0.100000 | 0.031569 |
| Waterfall | 24 | 0.208303 | 0.041667 | 0.001344 | 0.138889 | 0.005348 | 0.100000 | 0.006052 | 0.070833 | 0.029732 |
| Voting | 24 | 0.202154 | 0.000000 | 0.000000 | 0.097222 | 0.003699 | 0.125000 | 0.028352 | 0.095833 | 0.031149 |

\normalsize

> *For orchestration on the full corpus, Confidence achieved the best MRR (0.209), with Waterfall very close (0.208) and Voting slightly lower (0.202), so overall early-rank performance is similar across all three strategies. Waterfall produced the strongest Precision@3 (0.139), indicating better short-list relevance in the top few results, while Confidence led at Precision@5 (0.133). At larger cutoffs, Confidence and Voting were nearly tied on Recall@10, with Waterfall slightly behind. In practice, these results suggest that orchestration variants are competitive but not dramatically separated, with Confidence showing the most balanced behavior and Waterfall favoring early precision at small k.*

---

#### 2.3 Subsample vs. Full Corpus Comparison

The **subsample** (817 chunks) was used during development to debug the pipeline quickly. The **full corpus** (7,531 chunks) is roughly nine times larger, creating a much harder retrieval environment with many more distracting passages.

\tiny

| Scope       | Method   | MRR        | P@1        | P@3        | P@5        | P@10       | R@1        | R@3        | R@5        | R@10       |
|-------------|----------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
| full_corpus | BM25 | 0.151296 | 0.041667 | 0.055556 | 0.091667 | 0.091667 | 0.001016 | 0.001681 | 0.005506 | 0.011165 |
| subsample | BM25 | 0.396272 | 0.166667 | 0.305556 | 0.316667 | 0.295833 | 0.002081 | 0.010785 | 0.017426 | 0.037034 |
| full_corpus | Dense | 0.165525 | 0.041667 | 0.069444 | 0.058333 | 0.066667 | 0.000147 | 0.008953 | 0.010095 | 0.034097 |
| subsample | Dense | 0.540476 | 0.375000 | 0.402778 | 0.350000 | 0.304167 | 0.031450 | 0.064326 | 0.073926 | 0.093394 |
| full_corpus | GraphRAG | 0.232573 | 0.083333 | 0.097222 | 0.116667 | 0.116667 | 0.000687 | 0.003166 | 0.005892 | 0.052857 |
| subsample | GraphRAG | 0.579613 | 0.458333 | 0.444444 | 0.366667 | 0.337500 | 0.029346 | 0.059344 | 0.063712 | 0.080849 |
| full_corpus | Hybrid | 0.202154 | 0.000000 | 0.097222 | 0.125000 | 0.095833 | 0.000000 | 0.003699 | 0.028352 | 0.031149 |
| subsample | Hybrid | 0.462108 | 0.250000 | 0.347222 | 0.316667 | 0.316667 | 0.004266 | 0.062629 | 0.076968 | 0.097059 |
| full_corpus | ReRank | 0.222952 | 0.041667 | 0.138889 | 0.125000 | 0.100000 | 0.000196 | 0.004480 | 0.006323 | 0.010619 |
| subsample | ReRank | 0.442001 | 0.291667 | 0.250000 | 0.233333 | 0.245833 | 0.002821 | 0.006986 | 0.012018 | 0.029840 |

\normalsize

> *Across all methods, performance is consistently higher on the subsample than on the full corpus, with large MRR drops when moving to full-corpus retrieval. The strongest relative degradation appears in Dense and GraphRAG, which perform very well on subsample but lose substantial early-rank quality at full scale, indicating that larger search space and higher distractor density make relevance ranking harder. BM25 also declines, but less dramatically in relative terms, remaining a stable lexical baseline with low absolute performance in both scopes. Hybrid and ReRank remain competitive in full-corpus settings and outperform single BM25/Dense in several top-k metrics, suggesting fusion and reranking help recover robustness under scale. Overall, the comparison confirms that subsample results are optimistic, while full-corpus evaluation is more realistic and should be treated as the primary indicator of deployment-level retrieval difficulty.*

---

#### 1.8 Reproducibility Comparison Table

| Aspect | Original Report (PDF) | Reproduced Notebook (Current) | Match Status |
|--------|----------------------|-------------------------------|--------------|
| Subsample size | 817 docs/chunks | 817 fixed-size chunks | Match |
| Full-corpus size | 7,544 docs/chunks | 7,531 fixed-size chunks (local artifacts) | Minor mismatch |
| Subsample best baseline MRR | Hybrid ~ 0.654 | GraphRAG ~ 0.580, Dense ~ 0.540, Hybrid ~ 0.462 | Partial mismatch |
| Full-corpus Dense MRR | 0.166 (reported best baseline) | 0.166 | Match (value) |
| Full-corpus baseline ranking | Dense reported strongest baseline | GraphRAG/ReRank/Hybrid above Dense in reproduced run | Mismatch |
| Full-corpus orchestration best | Confidence ~ 0.205 (Step 3) | Confidence ~ 0.209 | Close match |
| Full-corpus Voting MRR | ~ 0.190 (Step 3), ~ 0.189 (Step 2 section) | ~ 0.202 | Near but higher |
| Full-corpus Waterfall MRR | ~ 0.161 (Step 3) | ~ 0.208 | Mismatch |
| Subsample to full trend | Metrics decrease on full corpus | Metrics decrease on full corpus | Match |
| Evaluation protocol | Shared IR metrics (P@k, Recall, MRR, plus nDCG in report) | Shared IR metrics (P@1/3/5/10, Recall@1/3/5/10, MRR) | Largely aligned |

Our reproduction definitely confirms the main trend: things get much harder on the full corpus compared to the subsample. We also matched key numbers like the full-corpus Dense MRR (0.166). Some of the differences in specific scores or rankings are probably just because we're using different artifact versions or slightly different settings for our fusion and reranking.

---

### Conclusion

By putting all the original project's scripts into one clean Jupyter Notebook, we now have a really solid and reproducible baseline. Out of all the single agents, GraphRAG did the best on the full corpus (MRR 0.233). For the orchestrators, Confidence came out on top (MRR 0.209). 

The biggest takeaway is that performance dropped significantly for every method when we switched from the subsample to the full corpus. This shows that we really need to evaluate on the full corpus to know how the system will actually perform in the real world. Now that we have this baseline, we can use it as our benchmark when we design the full multi-agent system in Step 2.
