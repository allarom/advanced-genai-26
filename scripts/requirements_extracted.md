 Multi -Agent Orchestration for RAG Systems  
Organizers  
The project is organized by:  
Prof. em. Dr. Gerd Kortemeyer , Michigan State University; Rectorate and AI -Center, ETH Zurich    
Dr. Guang Lu , Lecturer for Data Science, Lucerne University of Applied Sciences and Arts  
in collaboration with:  
Dr. Diego Antognini , Senior Research Scientist, Google DeepMind  
as a joint task of industry and academia for the HSLU Applied Information and Data Science Master’s Program in the 
course Advanced Generative AI . 
Overview  
This capstone builds on last semester’s multilingual RAG projects. Instead of redoing the entire pipeline, you will extend 
a high -performing baseline solution into a multi -agent system . Your task is to design, implement, and evaluate 
orchestration mechanisms  that coordinate retrieval, re -ranking, and answer generation across multiple agents.  
The challenge: retrieval strategies (BM25, dense embeddings, GraphRAG) each have strengths and weaknesses. A multi -
agent system can orchestrate these strategies adaptively, making retrieval more robust, explainable, and efficient.  
Deadlines  
The students form groups of size three and submit their solutions by 23 December  2025 at the latest, passing on the 
codebase and the report to the submission link : 
https://docs.google.com/spreadsheets/d/19aGGDR7T7vew6W_HaI9m3XJUazPlFOXz_k6lUe2hHEw/edit?gid=0#gid=0   
provided by Dr. Diego Antognini . The contribution of each individual group member must be clearly stated in the 
submission .   
Guidance  and Requirements  
Step 1. Baseline Setup (15 Points)  
• Start from the provided notebook (best solution from last semester).  
• Reproduce baseline results with BM25, Dense Retrieval, GraphRAG, Hybrid Retrieval, and re -ranking.  
• Report baseline metrics (Precision@k, Recall@k, MRR).  
Deliverables:  
• Verified working notebook.  
• Short baseline report.  
Step 2. Multi -Agent System Design ( 30 Points)  
Objective:  
Transform the pipeline into a multi -agent architecture  with specialized roles coordinated by an orchestrator.  
Suggested Agent Roles:  
• Query Understanding Agent  – reformulates or classifies queries.  
• Retriever Agents  – BM25, Dense, GraphRAG.  
• Fusion Agent  – merges and deduplicates results.  
• Re-Ranker Agent  – applies advanced re -ranking models.  
• Answer Synthesizer Agent  – generates the final answer.  
• Critic Agent  – verifies factual consistency, triggers re -retrieval if needed.  
Tasks:  
• Implement at least two orchestration mechanisms , e.g.:  


o Parallel + Fusion (ensemble)  
o Sequential Routing (waterfall)  
o Confidence -Based  Routing  
o Voting / Consensus  
o LLM-Orchestrated Dialogue  
o Critic Loop (self -verification)  
• Compare orchestration strategies on accuracy, efficiency, and explainability.  
Deliverables:  
• Multi -agent orchestration system (code notebook).  
• Report on design choices and orchestration mechanisms.  
Step 3. Evaluation and Analysis ( 15 Points)  
Metrics:  
• Quantitative:  Precision@k, Recall@k, MRR.  
• Qualitative:  Orchestrator explainability, complementarity of agents, failure analysis.  
• System Efficiency:  Measure latency and computational cost.  
Deliverables:  
• Evaluation results with tables, charts, and visualizations.  
o Synthetic Q&A pairs with relevant text chunks  
o Golden Q&A benchmarks provided  
• Comparative analysis of at least two orchestration mechanisms.  
    Extra Challenges (Optional, up to 15 Bonus Points)  
Ambitious teams may extend their work with one or more of the following:  
1. Adaptive Orchestration with reinforcement learning  – train the orchestrator to dynamically choose retrieval 
strategies based on query type and past success.  
2. Explainable Orchestration  – require the orchestrator to output a rationale for its choices (e.g., “Chose BM25 
because query contained names; added Dense retrieval for semantic coverage” ). 
3. Human -in-the-Loop Supervision  – integrate or simulate human reviewers who can override orchestration.  
4. Benchmark Expansion & Adversarial Queries  – create and evaluate against adversarial cases (ambiguous, code -
switched EN/DE, mixed -topic).  
Step 4. Final Report & Communication (15 Points)  
• Clarity & Structure (5 pts):  Concise and logically structured.  
• Critical Reflection (5 pts):  Discuss strengths, weaknesses, lessons learned.  
• Professionalism (5 pts):  Code readability, documentation, reproducibility.  
Grading Breakdown ( 75 Points Total)  
• Baseline Setup:  15 pts  
• Multi -Agent System Design:  30 pts  
• Evaluation & Analysis:  15 pts 
• Final Report & Communication:  15 pts  
• Extra Challenges:  up to +15 bonus (max final = 75) 
Final Deliverables  
• Code notebook with working multi -agent RAG orchestration system.  
• Final report (5 –10 pages).  
• Evaluation results (quantitative + qualitative).  
• (Optional) Extensions addressing extra challenges.  