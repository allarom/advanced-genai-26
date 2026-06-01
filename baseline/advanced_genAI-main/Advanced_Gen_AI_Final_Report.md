Project Documentation:
Multi-Agent Orchestration for RAG
Systems
W.MSCIDS GEN03 Advanced
Generative AI
GitHub Repository:https://github.com/Trista1208/advanced_genAI.git
Google Drive:
https://drive.google.com/drive/folders/1Joe570BZnd4bWPcRF1sT8ERo4a3H7TBw?usp=sharing
Prepared by: Jiaqi Yu, Alina Yaroshchuk, Marco Leupi
Date: 23.12.2025
The assessment is submitted as part of the requirements for the MSc in Applied Information and Data
Science at the School of Business, Lucerne University of Applied Sciences and Arts

Introduction
Retrieval-Augmented Generation (RAG) systems combine information retrieval with large language
models to answer questions based on external knowledge sources [ 3]. While single-retriever RAG systems
are straightforward to implement, they often struggle with diverse query types: lexical methods miss
semantically similar content, while dense retrievers may fail on queries requiring exact term matching.
This project develops and evaluates a multi-agent RAG system that coordinates multiple retrieval
methods to improve answer quality. The system combines three complementary retrievers: BM25 for
lexical matching [ 5], Dense retrieval using multilingual E5 embeddings for semantic similarity [ 6], and
GraphRAG for entity-centric and relational queries [ 2]. An orchestration layer decides how to combine
these retrievers based on query characteristics.
The system was built and evaluated on a bilingual (English/German) corpus of ETH Zurich news
articles, consisting of 817 documents in the subsample and 7,544 documents in the full corpus. All
experiments were conducted on both corpus sizes to assess scalability.
The report is structured as follows. Step 1 establishes baseline retrieval performance for individual
methods. Step 2 designs and compares three orchestration strategies: Waterfall with conditional
fallback, Voting with fixed-weight fusion, and Confidence with query-adaptive routing. Step 3 provides
comprehensive evaluation including agent complementarity analysis, failure patterns, statistical significance
testing, and advanced features such as adaptive learning, explainable orchestration, adversarial robustness
testing, and human-in-the-loop simulation.
1 Step 1. Baseline Setup
1.1 The Setup
The baseline system establishes a full pipeline from raw text documents to retriever construction and
evaluation. It is run in a GPU-enabled Colab environment, where the project repository is cloned and
both data and outputs are stored in Google Drive.
The results
The baseline setup is used to construct the following types of retrievers and measure their performance:
BM25, Dense Retrieval, GraphRAG, Hybrid Retrieval, Re-Ranking (EcoRank, GTE, OpenAI, Cohere).
The performance of these retrievers are measured using the following metrics:
•Precision@k:the proportion of relevant documents among the top k retrieved. Higher is better.
•Recall@k:the proportion of all relevant documents found within the top 100 retrieved. Higher is
better.
•MRR (Mean Reciprocal Rank):measures the average of the reciprocal ranks of the first
relevant document. Higher is better.
The retriever's performance is measured on 25 bilingual benchmark questions with LLM-graded qrels
(relevance greater than 0.5). The performance is tested for fixed size chunks (Figure 1).
1.2 Results on Subsample (817 documents)
BM25 Retrieval (Multilingual Baseline)
Fast lexical baselines are enhanced with BM25 pre-retrieval strategies (routing, bilingual rewriting,
pseudo-relevance feedback, acronym expansion, temporal filtering). The pre-retrieval strategies are
evaluated. Pseudo-relevance feedback has proven to be the best pre-retrieval strategy (Figure 1). Even
though its performance on MRR is rather mediocre, it performs best for Precision@5, Precision@10 and
Recall@100 for both fixed size chunks and semantic chunks. And since performance is best with fixed size
chunks, BM25 with pseudo-relevance feedback with fixed chunks is chosen as the base model.
Dense Retrieval with Multilingual E5:Dense retrieval using intfloat/multilingual-E5-large-
instruct with Chroma, providing semantic search through multilingual embeddings. Dense retrieval
performs better with fixed size chunks than semantic chunks for all metrics except for Recall@100. It also
performs better than the BM25 variants especially MRR increases significantly (Figure 1).
1

GraphRAG:An entity/event graph is built, clustered into Leiden communities, and condensed LLM
summaries are generated for retrieval. Although GraphRAG-C2 achieves the best performance compared
to the other two GraphRAG models, it performs worse than dense retrieval in all metrics (Figure 1).
Hybrid Retrieval:For hybrid retrieval late fusion of BM25, dense retrieval and GraphRAG is used.
Three different fusion operators (weighted sum, z-score sum and Reciprocal Rank Fusion) are implemented
and evaluated. With a recip rank of 0.654, z-score sum achieves best performance across fusion operators.
It is configured best with dense retrieval (0.60) and GraphRAG (0.40). The hybrid retrieval approach
achieves best performance in all metrics, except for Precision@5 (Figure 1).
Re-Ranking:The fine-tuned hybrid retrieval model, which achieves the best performance, is used
as baseline for the re-ranking. For the re-ranking EcoRank, GTE, OpenAI and Cohere are tested and
evaluated (Figure 2). All of the re-ranking methods increased performance compared to traditional
retrieval methods. Cohere performs best, achieving highest scores in all metrics, except for MRR.
1.3 Results on Full Corpus (7,544 documents)
To validate scalability, the baseline retrievers (BM25, Dense, GraphRAG) are also evaluated on the full
corpus, which is approximately 9 times larger than the subsample.
As expected, all metrics decrease substantially when scaling from the subsample to the full corpus
(Figure 3). This is a natural consequence of the increased search space, making the retrieval task more
challenging. Dense retrieval maintains its lead in MRR (0.166) and Recall@100 (0.128), confirming that
the relative performance trends observed on the subsample generalize to the full dataset.
1.4 Conclusion
For maximizing retrieval quality, reranking provides substantial benefits over traditional retrieval methods.
The hybrid-tuned retriever itself is a strong baseline, demonstrating the power of late fusion. When
considering deployment, the trade-off between reranking quality and latency must be carefully evaluated.
Although Cohere scores best, it comes with high latency. The consistency of relative performance rankings
between subsample and full corpus validates that the subsample is representative for method selection
and hyperparameter tuning.
2 Step 2. Multi-Agent System Design
2.1 Overview
Step 2 implements a multi-agent orchestration layer that coordinates multiple retrieval systems to improve
retrieval quality and answer synthesis. We designed and evaluated three orchestration strategies: Waterfall,
Voting, and Confidence-based routing. Each strategy represents a different approach to combining retriever
outputs, balancing quality, efficiency, and adaptability to query types.
2.2 System Architecture
The multi-agent system combines three retriever agents. The BM25 Agent performs bilingual lexical
retrieval with separate English and German indices, using Facebook's M2M100 model for automatic query
translation. The Dense Agent provides semantic retrieval using multilingual E5 embeddings with the
Chroma vector store. The GraphRAG Agent performs graph-based retrieval using Leiden community
clustering and LLM-generated summaries.
Two supporting agents complete the architecture. The Query Classifier Agent uses a lightweight
‘google/flan-t5-base‘ model to categorize queries as FACTOID, SEMANTIC, or HYBRID, enabling
dynamic weight adjustment. The Answer Synthesizer Agent generates final answers using ‘Mistral-7B-
Instruct-v0.2‘ based on retrieved context.
All orchestration strategies combine retriever outputs using weighted Reciprocal Rank Fusion (RRF)
[1]. This fusion method assigns each document a score based on its rank position in each retriever's
output, weighted by retriever importance. Documents ranked higher receive proportionally more credit,
and the weights allow prioritising certain retrievers over others. A smoothing constant prevents excessive
emphasis on top-ranked documents.
2

2.3 Orchestration Strategies
Waterfall Strategy implements sequential routing with conditional fallback. It initially retrieves using only
BM25 and Dense with weights favoring BM25 (1.2) over Dense (1.0). A critic agent then evaluates the
top result's query-term overlap. If this overlap falls below 5%, indicating poor lexical match, GraphRAG
is activated as fallback with adjusted weights. This strategy optimizes for efficiency by avoiding expensive
GraphRAG calls when lexical and semantic retrieval prove sufficient.
Voting Strategy performs parallel retrieval with fixed-weight fusion. All three retrievers run simul-
taneously with predetermined weights: BM25 at 1.2, Dense at 1.0, and GraphRAG at 0.6. Results are
fused using weighted RRF without any conditional logic. This strategy maximizes coverage by always
utilising all available retrieval signals.
Confidence Strategy dynamically adjusts weights based on query classification. The Query Classifier
Agent first categorizes the query, then weights are set accordingly. For FACTOID queries involving names,
numbers, and dates, BM25 receives higher weight for precise term matching. For SEMANTIC queries
involving explanations and concepts, Dense retrieval is boosted for better conceptual understanding. For
HYBRID queries mixing both types, balanced weights are applied across all retrievers.
2.4 Architectural comparison: subsample vs full corpus
2.4.1 Retriever configuration differences
The most significant change concerns the BM25 retriever. On the subsample, BM25 uses Query Expansion
via Pseudo-Relevance Feedback (PRF) [ 4], which improves recall by expanding queries with frequent
terms from top initial results. On the full corpus, this expansion is removed in favor of baseline BM25.
This decision reflects the trade-off between recall improvement and computational cost, PRF requires two
retrieval passes and risks topic drift from noisy expansion terms when the corpus is larger.
The pre-retrieval depth also differs between configurations. The subsample uses dynamic pre-k
calculated as the maximum of 30 or ten times the requested top-k, adapting to query needs. The full
corpus uses a fixed pre-k of 120, which ensures consistent deep retrieval regardless of query characteristics.
This change simplifies system behavior and guarantees sufficient candidates for effective fusion in the
larger search space.
Additionally, BM25 scoring differs due to implementation constraints. The subsample uses native
BM25Okapi scores, while the full corpus uses rank-based scoring where each document receives a score
equal to the total count minus its rank position. This change accommodates the LangChain BM25Retriever
interface, which does not expose raw scores, while maintaining consistent scale for fusion with other
retrievers.
2.4.2 Orchestration Weight Adjustments
The Confidence strategy's weights were adjusted for the full corpus, particularly for FACTOID queries.
We increased GraphRAG's weight substantially from 0.5 to 0.8, while slightly decreased BM25 from 1.4 to
1.3 and increased Dense from 0.9 to 1.0. This change underscores the growing role of GraphRAG in larger
search spaces, where relevant factual information is spread across many documents, making entity-based
graph traversal increasingly important for identifying and connecting the required evidence. The weights
for SEMANTIC and HYBRID query types remain unchanged between configurations, as the relationship
between conceptual queries and retriever capabilities does not fundamentally change with corpus size.
2.5 Evaluation Results
2.5.1 Answer Coverage (Hit Rate)
Hit rate measures semantic answer coverage (how often the retrieved context contains tokens from the
gold answer, requiring at least two-token overlap).
On the subsample (Figure 4), Confidence achieves the highest hit rate at 0.920, followed by Voting at
0.900 and Waterfall at 0.880. This demonstrates the value of query-adaptive routing when the search
space is manageable.
On the full corpus (Figure 5), Voting takes the lead with 0.940, while Confidence maintains 0.920
and Waterfall remains at 0.880. We suggest that consistent multi-retriever fusion provides more robust
coverage when the search space is larger, as adaptive routing may miss relevant documents that don't fit
neatly into query type categories.
3

2.5.2 Formal IR Metrics
On the subsample (Figure 6), Waterfall achieves the best MRR (0.447), indicating it effectively places
the most relevant document at the top. However, Voting leads in precision and ranking quality metrics
(P@10: 0.342, nDCG@10: 0.350).
On the full corpus (Figure 7), Voting shows higher values across all metrics with the best MRR (0.189),
precision (P@5: 0.083, P@10: 0.096), and ranking quality (nDCG@10: 0.104). Importantly, Voting shows
the smallest performance degradation when scaling, that demonstrates robustness to corpus size increase.
2.6 Qualitative Analysis: Answer Synthesis Examples
The following examples illustrate how different orchestration strategies affect answer quality across both
datasets.
Example 1: Factual Query ”Who were the rectors of ETH between 2017 and 2022?”
On the subsample, all strategies either fail to find the answer or produce incorrect responses (Figure 8).
On the full corpus, both Voting and Confidence correctly identify Sarah Springman as rector from
2017 until her departure in January 2022, while Waterfall still returns ”NOT FOUND IN CONTEXT”
(Figure 9). This demonstrates how the larger document collection contains more relevant content, and
multi-retriever fusion helps surface it effectively.
Example 2: Domain-Specific Query ”Does ETH organize any competitions?”
The gold answer mentions various competitions including the notable Cybathlon. On the subsample,
responses reference peripheral events like a photo contest or Olympic athletes (Figure 10). On the full
corpus, Confidence successfully retrieves information about the Cybathlon, which directly matches the
gold answer's most prominent example (Figure 11). This illustrates how query-adaptive routing can
surface highly relevant but semantically distinct content when the retrieval space is larger.
2.7 Conclusions
Voting achieved the best results across all experiments and scaled better from subsample to full corpus.
Waterfall performed well on the subsample but showed weaker results on the full corpus. The conditional
GraphRAG activation saves computation time, but sometimes skips relevant documents. Confidence did
not perform better than simpler approaches despite its adaptive design. The query classifier was not
accurate enough to provide consistent improvements. The architectural adaptations for the full corpus
were successful. Removing Query Expansion simplified the BM25 pipeline without hurting performance.
Increasing GraphRAG weights helped retrieve relevant documents from the larger collection.
3 Step 3. Core Evaluation
3.1 Evaluation Methodology
A broad set of Information Retrieval (IR) metrics was used to evaluate system performance. Precision
(@1, @3, @5, @10) measures relevance among top-ranked documents, Recall (@5, @10, @100) captures
coverage of relevant documents, MRR reflects the rank of the first relevant result, and NDCG (@5, @10)
assesses ranking quality. System efficiency was evaluated using average, P95, and P99 latency, as well as
end-to-end processing time. The evaluation pipeline executes retrieval for each query, records per-query
latency, computes IR metrics using pytrec eval, and aggregates results via macro-averaging.
3.2 Quantitative Evaluation Results
3.2.1 Performance on Subsample Corpus
On the subsample corpus of 817 documents, the Waterfall strategy achieved the highest MRR of 0.447,
followed by Voting at 0.421 and Confidence at 0.403 (Figure 12). This indicates that on smaller corpora,
the conditional GraphRAG fallback mechanism in Waterfall provides effective coverage when initial
retrieval is insufficient.
Regarding efficiency, the Confidence strategy exhibited the lowest average latency at 3.08 seconds,
approximately 2.5 times faster than Waterfall (7.78 seconds) and 1.8 times faster than Voting (5.66
seconds) (Figure 12). The efficiency advantage of Confidence stems from its intelligent query routing that
assigns appropriate weights based on query classification, avoiding unnecessary retrieval from all agents.
4

Precision metrics showed comparable performance across strategies at higher cutoffs (Figure 12).
Voting achieved the highest P@10 (0.342) and NDCG@10 (0.350), suggesting better ranking quality
when more results are considered. Recall@100 was similar across all strategies (approximately 0.28-0.30),
indicating that all approaches can retrieve most relevant documents given sufficient depth.
3.2.2 Performance on Full Corpus
Scaling to the full corpus of 7,544 documents revealed significant performance changes (Figure 13). The
Confidence strategy emerged as the best performer with MRR of 0.205, followed by Voting at 0.190
and Waterfall at 0.161. This reversal from subsample results demonstrates that query-adaptive weight
assignment becomes more important as corpus size increases.
The efficiency characteristics also changed substantially. Waterfall became the fastest strategy at 3.05
seconds average latency, followed closely by Confidence at 3.36 seconds (Figure 13). This improvement
occurred because Waterfall's overlap criterion is more easily satisfied on the larger corpus, reducing the
frequency of GraphRAG fallback invocations. Voting exhibited considerably higher latency due to the
increased cost of GraphRAG retrieval over a larger knowledge graph.
All strategies showed lower precision and recall on the full corpus, which is expected given the larger
search space. However, the relative performance differences became more noticeable. Confidence and
Voting achieved P@5 of approximately 0.10-0.12, while Waterfall dropped to 0.058, indicating that
Waterfall's conditional approach may miss relevant documents when the overlap threshold is not triggered
(Figure 13).
3.3 Qualitative Analysis
3.3.1 Agent Complementarity
The AgentComplementarityAnalyzer examined overlap among BM25, Dense retrieval, and GraphRAG
agents across 15 test queries (Figure 14). On the subsample corpus, only 1.22% of retrieved documents
were shared by all three agents, indicating strong complementarity. BM25-only documents accounted for
37.95%, reflecting BM25's strength in exact keyword matching. Graph-only documents comprised 33.05%,
demonstrating GraphRAG's ability to retrieve relational and entity-centric information. Dense-only
documents represented 14.23%, highlighting semantic retrieval's contribution for queries using different
terminology than source documents.
On the full corpus, the distribution became more balanced: BM25-only at 30.34%, Dense-only at
29.99%, and Graph-only at 30.60%, with only 0.30% overlap among all three. This equalisation suggests
that Dense retrieval becomes proportionally more valuable on larger corpora where semantic matching
helps bridge vocabulary gaps across a wider range of documents.
These complementarity results validate the multi-agent fusion approach: each retriever contributes
substantially unique documents, and combining their results through RRF captures information that any
single agent would miss.
3.3.2 Failure Pattern Analysis
The FailureAnalyzer identified queries with NDCG@10 below 0.5. On the subsample, failure counts
ranged from 16 to 18 across strategies, while on the full corpus this increased to 20-22 failures. The
average query length among failures was approximately 8.3-8.5 words, indicating no strong correlation
between query length and retrieval difficulty.
Question type distribution among failures was dominated by ”What” questions (7 failures), followed by
”Who” and ”How” questions (4-5 each). Recurring failure patterns included queries requiring enumeration
of multiple entities (such as listing all ETH rectors between specific years), queries about specific quotes
or statements attributed to named individuals, and queries about niche projects or initiatives with limited
textual coverage in the corpus.
Common failure examples persisted across both corpus sizes and all strategies. Queries like ”Who are
famous ETH alumni?” require comprehensive enumeration across many documents, while queries about
specific projects like ”What is e-Sling?” demand retrieval of documents that may use varying terminology
or abbreviations.
5

3.4 System Efficiency and Statistical Analysis
Latency measurements reveal distinct efficiency profiles across strategies and corpus sizes (Figure 15). On
the subsample, the Confidence strategy achieved a 2.5x speedup over Waterfall due to intelligent query
routing that avoids unnecessary agent invocations. On the full corpus, this relationship partially inverted:
Waterfall became fastest because its overlap criterion prevented most GraphRAG calls, while Voting's
parallel retrieval from all agents incurred the highest cost.
Statistical significance testing using paired t-tests at α= 0.05 indicated that MRR differences between
strategies were not statistically significant on either corpus (Figure 16). On the subsample, p-values
ranged from 0.36 to 0.63 for all pairwise comparisons. On the full corpus, the Voting versus Confidence
comparison approached significance (p = 0.065), suggesting that with a larger query set, the Confidence
strategy's advantage might reach statistical significance.
The lack of significance is attributed to the small sample size of 25 queries, high per-query variance
in retrieval outcomes, and modest absolute effect sizes. Despite this, the consistent direction of results
across multiple metrics supports practical conclusions about strategy selection.
4 Step 4: Advanced and Bonus Features
4.1 Adaptive Orchestration with Reinforcement Learning
An ‘AdaptiveOrchestrator‘ based on Q-learning [ 7] was implemented to overcome limitations of static
weight assignment. Query types (factoid, semantic, balanced) form the state space, while orchestration
strategies (bm25 heavy, dense heavy, graph heavy, balanced) constitute the action space. Rewards are
derived from retrieval success measured by answer overlap, and learning proceeds via an epsilon-greedy
policy with learning rate 0.1 and discount factor 0.9.
Training on 20 queries revealed different optimal strategies depending on corpus size (Figure 17).
On the subsample, the system learned to prefer graph heavy strategies for factoid queries and balanced
approaches for semantic queries, achieving 75% success rate and 0.775 average feedback score. On the
full corpus, the system converged toward balanced strategies across all query types, with reduced success
rate (30%) and lower feedback (0.370).
The learned Q-values on the subsample showed highest values for dense heavy on factoid queries
(0.587) and graph heavy on balanced queries (0.731). On the full corpus, balanced strategies dominated
with values around 0.30-0.48 across query types. This shift indicates that as corpus size increases, no
single agent provides consistently superior results, making balanced fusion more robust.
4.2 Explainable Orchestration
The ‘ExplainableOrchestrator‘ provides transparent rationales for orchestration decisions by analysing
query features, classifying query type, assigning agent weights, and documenting the retrieval and fusion
process. For each query, the system outputs feature analysis (length, presence of digits, named entities,
question words), determined query complexity, weight assignments with explanations, and a step-by-step
decision flow.
For example, the query ”Who was president of ETH in 2003?” is classified as factoid based on the
presence of a specific date, resulting in BM25 weight of 1.4, Dense weight of 0.9, and Graph weight of 0.5.
The rationale explicitly states that exact matching is prioritized for queries containing specific identifiers.
4.3 Adversarial Query Evaluation
An ‘AdversarialQueryGenerator‘ was used to test system robustness under challenging conditions, including
ambiguous queries, code-switched English–German queries, and paraphrased queries.
On the subsample, paraphrased queries achieved 75% success rate across strategies, code-switched
queries reached 40%, and ambiguous queries showed 0-50% success with only Voting handling them
partially (Figure 18). On the full corpus, performance degraded: paraphrased queries dropped to 50%
success, while code-switched (40%) and ambiguous (0%) remained similar (Figure 19).
The degradation for paraphrased queries on the full corpus suggests that semantic matching becomes
more difficult when relevant documents are distributed among a larger pool of candidates. All strategies
showed identical adversarial performance on the full corpus, indicating that robustness challenges are
shared across orchestration approaches rather than being strategy-specific.
6

4.4 Human-in-the-Loop Simulation
A ‘HumanInTheLoopSimulator‘ was implemented to model iterative refinement using simulated human
feedback. Retrieved results are scored using lenient matching criteria based on exact matches, entity
overlap, and keyword coverage. Feedback is used to reinforce successful strategies and penalize failures,
enabling adaptive learning over multiple iterations.
On the subsample, 16 interactions produced 9 approvals and 7 rejections, with average feedback score
of 0.584 and final success rate of 100% after iterative refinement (Figure 20). On the full corpus, 12
interactions yielded 10 approvals and only 2 rejections, with higher average feedback (0.694) and 100%
success rate. The reduced rejection rate on the full corpus may reflect the broader document coverage
providing more opportunities for partial matches.
Both simulations demonstrated that iterative human feedback can recover from initial retrieval failures,
achieving perfect task completion within 3 iterations maximum. This validates the human-in-the-loop
approach for production deployment where user feedback can continuously improve system performance.
5 Discussion and Conclusions
The evaluation demonstrates that orchestration strategy effectiveness depends significantly on corpus scale.
On smaller corpora, the Waterfall strategy's conditional GraphRAG fallback provides the best MRR,
while on larger corpora, the Confidence strategy's query-adaptive weight assignment scales more smoothly.
The Voting strategy provides consistent middle-ground performance but at higher computational cost.
Strong complementarity among BM25, Dense, and GraphRAG agents validates the multi-agent
fusion approach. Each retriever contributes approximately 30-38% unique documents, with minimal
three-way overlap (0.3-1.2%). This complementarity becomes more balanced on larger corpora, where
Dense retrieval's semantic matching becomes proportionally more valuable.
The advanced features demonstrate pathways for system improvement. Adaptive Q-learning captures
non-intuitive strategy preferences but requires corpus-specific training. Explainable orchestration supports
debugging and trust-building for production deployment. Adversarial evaluation reveals robustness gaps,
particularly for ambiguous queries that lack specific entities. Human-in-the-loop simulation shows that
iterative feedback can recover from initial retrieval failures and improve task completion.
For practical deployment, the Confidence strategy offers the best balance of effectiveness and efficiency
on larger corpora, but its performance may vary due to instability in the query classifier. Its query
classification overhead is minimal compared to the savings from intelligent agent routing, and its
MRR degradation is the lowest among evaluated strategies. Future work should focus on improving
query classification accuracy, expanding adversarial robustness, and integrating real human feedback for
continuous learning.
7

6 Appendix
Figure 1: Subsample retriever results on fixed size chunks
Figure 2: Subsample reranking results
8

Figure 3: Full corpus retriever results on fixed size chunks
Figure 4: Subsample Hit Rate
Figure 5: Full corpus Hit Rate
Figure 6: Orchestration performance on subsample (817 documents)
9