# Reliable and Adaptive Agentic RAG Systems

## Organizers
- Prof. em. Dr. Gerd Kortemeyer, Michigan State University
- Rectorate and AI-Center, ETH Zurich
- Dr. Guang Lu, Lecturer for Data Science, Lucerne University of Applied Sciences and Arts
- Dr. Diego Antognini, Senior Research Scientist, Google DeepMind

Joint task for the HSLU Applied Data Science and AI Master's Program — Advanced Generative AI.

---

## Overview

This capstone builds on the work of the previous two semesters.
- **Semester 1:** Multilingual RAG (data prep, retrieval, re-ranking, answer evaluation)
- **Semester 2:** Multi-agent architecture (agents coordinated retrieval, fusion, critique, answer synthesis)
- **This semester:** Focus shifts to **reliability and adaptation**.

A strong agentic RAG system should determine:
- Whether evidence is sufficient
- Whether the answer is properly supported
- Whether retrieved information is conflicting
- Whether clarification is needed
- Whether it should abstain instead of producing an unreliable answer

The system should also **adapt its orchestration behavior** depending on the query, evidence quality, and earlier outcomes.

---

## Objective

Build on an existing strong RAG or multi-agent RAG baseline and extend it into a system that can reason not only about **what to retrieve and answer**, but also about **whether it should**.

Deliverables:
- Codebase / Notebook
- Final Report
- Evaluation Results
- Benchmark Extension
- System Demonstration

---

## Step 1. Baseline Setup and Failure Analysis (15 Points)

### Tasks and Requirements

**Reproduce the Baseline System**
- Start from a strong RAG or multi-agent RAG solution from previous semester(s)
- Ensure pipeline runs end-to-end
- Verify retrieval, orchestration, re-ranking, answer synthesis

**Measure Baseline Performance**
- Report baseline performance on available benchmark
- Include retrieval quality, answer quality, efficiency (latency, cost)
- Clearly state which configuration is the reference baseline

**Analyze Failure Cases**
- Examine representative poor-performance examples
- Organize into structured failure taxonomy:
  - **retrieval failure:** relevant evidence was not retrieved
  - **ranking failure:** relevant evidence was retrieved but ranked too low
  - **synthesis failure:** evidence was present but the answer was wrong or unsupported
  - **orchestration failure:** the system chose a poor strategy or sequence
  - **trust failure:** the system was overconfident or underconfident
- Document concrete examples with query, expected outcome, and actual behavior

**Deliverables**
- Working baseline reproduction
- Performance summary with metrics
- Failure taxonomy with examples
- Clear motivation for the reliability extensions

---

## Step 2. Reliability-Aware Multi-Agent Architecture (15 Points)

### Tasks and Requirements

**Design a Reliability-Aware Multi-Agent Architecture**
- Define specialized agents and responsibilities
- Minimum components:
  - query understanding or profiling
  - retrieval coordination
  - evidence analysis
  - answer generation
  - verification or critique
  - orchestration or policy control
- Architecture must clearly separate **answer production** from **answer checking** and **decision control**

**Define Reliability Signals**
- Evidence sufficiency
- Agreement across retrieved chunks
- Grounding strength
- Contradiction indicators
- Ambiguity indicators
- Confidence or uncertainty scores

**Specify the Decision Logic**
- System must explicitly decide what to do next based on intermediate results
- Possible actions:
  - proceed to answer generation
  - perform another retrieval pass
  - rewrite the query
  - trigger a critique loop
  - ask for clarification
  - abstain from answering

**Make the Orchestration Interpretable**
- Record or expose why important decisions were made
- Example: why a second retrieval pass was triggered, why the system abstained

**Deliverables**
- Reliability-aware multi-agent RAG architecture
- Description of agents, interactions, and orchestration logic
- Explanation of reliability signals
- Clear justification of design choices

---

## Step 3. Reliability and Adaptation Mechanisms (20 Points)

Students must implement **at least four** of the following mechanisms (A-H).

### A. Evidence Sufficiency Estimation
Determine whether retrieved evidence is sufficient to answer reliably.
- May be based on semantic coverage, support from multiple chunks, missing information detection

### B. Groundedness / Support Verification
Check whether the generated answer is supported by retrieved evidence.
- Claim-by-claim verification, answer-to-source alignment, citation checking, verifier-style prompting

### C. Contradiction Detection
Detect conflicting information in retrieved evidence.
- Respond by reporting uncertainty, summarizing conflict, retrieving additional evidence, or abstaining

### D. Clarification Strategy
Identify under-specified or ambiguous questions.
- Ask a clarification question instead of answering too early

### E. Abstention Mechanism
Allow the system to abstain when evidence is insufficient or unreliable.
- Must be linked to explicit signals, not used arbitrarily

### F. Self-Reflection / Critique Loop
Implement a critic, verifier, or reflective agent.
- Reviews draft answer and decides whether revision or another action is needed

### G. Recovery Mechanism
When likely failure is detected, recover through adaptive actions:
- retrieving again with a rewritten query
- switching retrieval strategy
- re-ranking again
- calling a stronger verifier
- switching from answer mode to clarification mode
- switching from answer mode to abstention mode

### H. Trust / Confidence Scoring
Implement a confidence or trust score for answers.
- Must explain what it means and how it is computed

### Adaptation Requirement
At least one mechanism must **go beyond analysis** and actually **change orchestration behavior**:
- route differently depending on query type
- choose different recovery action depending on detected failure
- change behavior based on previous outcomes
- use feedback or past logs to improve strategy selection

**Deliverables**
- Integrated reliable and adaptive agentic RAG system
- Implementation of at least four reliability/adaptation mechanisms
- Clear explanation of failure detection and response
- Evidence that the system adapts behavior in meaningful cases

---

## Step 4. Evaluation and Analysis (10 Points)

### Tasks and Requirements

**Evaluate the Best System Configuration**
- Apply the best reliable and adaptive pipeline to the benchmark
- Compare directly against the baseline from Step 1

**Evaluate Retrieval and Answer Quality**
- Standard metrics:
  - Precision@k, Recall@k, MRR
  - Semantic similarity-based answer measures
  - BLEU, ROUGE where useful
- Clearly explain which metrics and why

**Evaluate Reliability-Oriented Behavior**
- Reliability metrics:
  - grounded answer rate
  - unsupported claim rate
  - contradiction handling quality
  - correct abstention rate
  - false abstention rate
  - recovery success rate
  - usefulness of clarification
  - alignment between confidence and correctness
- Students may define additional well-justified metrics

**Extend the Benchmark**
- Add cases that test reliable and adaptive behavior:
  - ambiguous queries
  - insufficient-evidence queries
  - conflicting-evidence queries
  - adversarial or misleading queries
  - multilingual or code-switched queries (if relevant)

**Perform Comparative and Qualitative Analysis**
- Compare proposed system to baseline
- Analyze whether mechanisms improve trustworthiness and robustness
- Discuss tradeoffs: quality, reliability, latency, complexity
- Include at least one **ablation study**
- Present representative qualitative examples:
  - successful grounded answer
  - revised answer after critique
  - clarification case
  - abstention case
  - difficult failure case

**Deliverables**
- Quantitative evaluation results (tables/visualizations)
- Reliability-focused benchmark extension
- Comparative analysis against baseline
- At least one ablation study
- Qualitative discussion of behavior and limitations

---

## Step 5. Final Report and Communication (5 Points)

### Tasks and Requirements

**Write a Clear Final Report**
- Describe:
  - baseline system
  - failure analysis
  - proposed architecture
  - reliability mechanisms
  - adaptation logic
  - experiments and results
  - limitations and future work
- Concise, structured, analytical

**Provide Critical Reflection**
- Which mechanisms were most useful?
- Did self-reflection, abstention, or adaptation truly improve the system?
- What did not work as expected and why?

**Ensure Professionalism and Reproducibility**
- Readable, documented code
- Clear experimental settings
- Inspectable system decisions (logs, outputs, saved examples)

**Deliverables**
- Final report
- Clean and documented codebase / notebook
- Clear presentation of findings and conclusions

---

## Suggested Benchmark Extension

Categories:
- Standard answerable questions
- Ambiguous questions (should trigger clarification)
- Insufficient-evidence questions (should lead to abstention)
- Conflicting-evidence questions (retrieved material disagrees)
- Adversarial or misleading questions (test robustness)
- Multilingual or code-switched questions (if appropriate)

The benchmark extension should be documented clearly and used in evaluation.

## Evaluation Focus

A strong project will show the system can:
- recognize when evidence is weak or incomplete
- avoid unsupported or overconfident answers
- recover intelligently when likely failure is detected
- adapt its behavior to the situation
- make orchestration decisions more transparent

---

## Extra Challenges (Optional, up to 5 Bonus Points)

Ambitious teams may extend with one or more of the following:

1. **Learning-Based Adaptive Orchestration**
   Train a controller, bandit, or lightweight policy learner that improves orchestration choices based on observed success and failure.

2. **Confidence Calibration**
   Study whether the system's confidence estimates align with actual correctness.

3. **Human-in-the-Loop Supervision**
   Integrate or simulate human feedback to refine, correct, or override system behavior.

4. **Memory-Based Adaptation**
   Allow the system to adapt decisions based on prior interactions or earlier failures.

5. **Trust-Aware Explanations**
   Require the system to explain not only its answer, but also why it trusted the evidence enough to answer, revise, clarify, or abstain.

---

## Use of Coding Tools and External Resources

Students are allowed to use coding assistants and other software-based tools for code generation, debugging, refactoring, or documentation support.

### Requirements
- **Transparency:** State which coding tools were used and for which parts
- **Clear attribution:** Distinguish tool-supported work from own contributions (design, implementation, experiments, analysis, conclusions)
- **Documentation in report:** Short statement describing:
  - which tools were used
  - how they were used
  - which parts were mainly tool-assisted
  - which parts were primarily student-developed
- **Citation:** Cite external resources clearly and consistently
- **Academic integrity:** All submitted work remains the student's responsibility. Must be able to explain and justify code, design choices, experiments, and findings.
