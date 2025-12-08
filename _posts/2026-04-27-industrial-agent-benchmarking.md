---
layout: distill
title: Industrial Agent Benchmarking: What 300+ Real Evaluations Reveal About the Limits of Agentic AI
description: A community-scale evaluation of multi-agent systems using AssetOpsBench and a privacy-preserving Codabench pipeline, exposing real-world failure modes and methodological gaps in industrial AI agent evaluation.
date: 2026-04-27
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-industrial-agent-benchmarking.bib

toc:
  - name: Introduction
  - name: Why Industrial Agent Evaluation Needs a Rethink
  - name: AssetOpsBench: A Foundation for Real-World Agent Evaluation
    subsections:
      - name: Data Modalities and Scenario Design
      - name: The Six-Dimensional Evaluation Framework
  - name: Why Codabench Was Necessary
  - name: Building AssetOpsBench-Live: From Offline Benchmarks to Real-Time Evaluation
    subsections:
      - name: Pipeline Architecture
      - name: Hidden Scenarios and Privacy Constraints
      - name: Methodology for Selecting Development vs Testing Instances
  - name: Community Results: 225 Users, 300+ Agents, One Reality Check
    subsections:
      - name: Leaderboard Behavior
      - name: Observed Failure Patterns
      - name: Emergent Failure Modes
      - name: Evaluation and Ranking Methodology
  - name: Case Study: Missing Clarifying Questions and Its Impact
  - name: What We Learned About Industrial Agents
  - name: Implications for Agentic AI
  - name: Future Directions
  - name: Conclusion
  - name: References
---

# Introduction

Large Language Model (LLM) agents have quickly become the default way researchers test “autonomous” reasoning, planning, and multi-step tool use. Yet almost all widely used benchmarks evaluate tasks that are far removed from real industrial systems such as web navigation, coding tasks, synthetic puzzles, or curated IT workflows. In practice, industrial operations demand something more difficult: **multi-agent coordination over heterogeneous telemetry, failure modes, work orders, and safety-critical tasks, all under privacy constraints.**
To understand whether current agents can survive this environment, we built a two-part evaluation ecosystem:

1. **AssetOpsBench** — a multi-modal industrial agent benchmark grounded in real sensor telemetry, failure models, and maintenance workflows [@assetopsbench2025].  
2. **AssetOpsBench-Live** — a real-time, privacy-preserving, Codabench-powered competition platform we deployed to 225+ users, yielding over 300 code-submitted agent evaluations [@assetopsbenchlive2026]. 

This blog post presents what we learned after systematically evaluating LLM agents, not via prompt logs, but through **real code, real trajectories, and real industrial tasks**.

The outcome is blunt: **current agentic AI is far less reliable than leaderboard scores suggest.** The challenge surfaces failure modes that traditional agent benchmarks simply never expose.

<figure>
<img src="../assets/img/assetopsbench.png" alt="Failure mode distribution" width="80%">
  <figcaption><b>Figure 1.</b>Architecture of the Multi-Agent System: Time Series Foundation Model (TSFM) Agent,
Failure Mode Sensor Relations (FMSR) Agent,Work Order (WO) Agent</figcaption>
</figure>

---

# Why Industrial Agent Evaluation Needs a Rethink

Industrial tasks do not behave like web navigation or software debugging. A single query may require:

- retrieving time-series telemetry  
- correlating it with failure models  
- understanding historical work orders  
- invoking specialized analytic tools  
- generating explanations or interventions  
- coordinating with multiple sub-agents  

Existing agent benchmarks (AgentBench, MultiAgentBench, MLE-Bench) do not include these characteristics. They evaluate agents on clean, single-modality tasks with minimal domain constraints. The result is misleading optimism.

Real industrial agent evaluation requires:

- **multi-agent sequences**, not single policies  
- **ambiguity, missing data, and drift**, not curated inputs  
- **tool invocation correctness**, not text-only reasoning  
- **ground-truth procedural forms**, not free-form answers  
- **privacy-preserving execution**, not public trajectories  

AssetOpsBench was designed specifically to fill this gap.

---

# AssetOpsBench: A Foundation for Real-World Agent Evaluation

AssetOpsBench is the first benchmark built around **authentic industrial asset operations** — chillers, AHUs, IoT telemetry, work orders, FMEA data, and domain-authored task scenarios. It contains:

- **2.3M sensor telemetry points**  
- **4.2K work orders**  
- **53 structured failure modes**  
- **140+ curated scenarios across 4 agent classes**  
- **both single-agent and multi-agent workflows**  

It evaluates not whether the agent “responds reasonably,” but whether it follows the **right sequence**, uses the **correct tools**, retrieves the **correct data**, and produces **verifiable results**.

## Data Modalities and Scenario Design

Tasks are grounded in real operational needs:

- anomaly detection  
- failure mode reasoning  
- KPI forecasting  
- sensor–failure alignment  
- work order summarization  
- multi-agent end-to-end fault analysis  

Scenarios include metadata such as task type, expected output form, operational category, and required sub-agents.

This explicit structuring allows *procedural* evaluation instead of subjective grading.

## The Six-Dimensional Evaluation Framework

Every agent run is scored along six axes:

1. **Task Completion** — Did the agent finish the task?
2. **Retrieval Accuracy** — Did it fetch the correct sensor/asset data?
3. **Result Verification** — Is the technical output correct?
4. **Action Sequence Correctness** — Were tools invoked in a logical order?
5. **Clarity & Justification** — Is the reasoning interpretable to operators?
6. **Hallucination Rate** — Did it avoid fabricating data?

These metrics are essential because agents routinely “sound correct” without doing the correct work which is a recurring theme we quantify later.

---

# Why Codabench Platform Was Necessary for Evaluation

Benchmarks alone do not solve the fundamental issues with evaluating industrial agents:

- **Data privacy** blocks public scenario sharing  
- **Models often need code execution**, not text-only reasoning  
- Competitions that only return scores **stifle learning and increase dropout**  
- Industrial environments require **reproducibility, versioning, and containerization**

Codabench solves these operational and research constraints:

- Submissions are **Dockerized**  
- Evaluation runs on **hidden scenarios**  
- Trajectories never leave the secure environment  
- Only final **scores + failure clusters** are returned  
- Multi-track design supports both **Planning** and **Execution** agents  

This transforms the benchmark into a **community-scale evaluation ecosystem**.

---

# Building AssetOpsBench-Live: From Offline Benchmarks to Real-Time Evaluation

AssetOpsBench-Live extends AssetOpsBench into an online, reproducible competition environment.

## Pipeline Architecture
<figure>
  <img src="../assets/img/combined_architecture.png"
       alt="AssetOpsBench-Live System Architecture and Failure Mode Visualization"
       style="width:100%; max-width:1100px;">

  <figcaption>
    <b>Figure.</b> AssetOpsBench-Live: (left) System architecture integrating AssetOpsBench with Codabench, 
    showing local development, global evaluation, and feedback; 
    (right) visualization of clustered failure modes.
  </figcaption>
</figure>


The full pipeline (see system diagram) includes a high-level description of the workflow, which consists of three key steps:

1. **Step 1:** Users build a Planning or Execution agent, run example scenarios locally, containerize it as a ZIP, and submit to Codabench for evaluation.

2. **Step 2:** Submissions are validated to ensure exactly one .py and one .json file (with required keys), then executed on one public scenario (previously runnable locally) and several hidden track-specific scenarios.
 
3. **Step 3:**  Covers scoring, result generation, status updates, and robust error handling. Evaluation provides scenario-level and model-level analyses across the six dimensions included in Section 3.2. 
This pipeline allows anyone to evaluate their agent realistically without accessing private data.

## Hidden Scenarios and Privacy Constraints

Industrial data cannot be public.  
Codabench enables:

- Running code inside a locked environment  
- Evaluating on scenarios that participants never see  
- Returning only structured scores and failure summaries  
- Ensuring deterministic, reproducible runs  

This was essential for wide community participation.

## Two Tracks

The competition was conducted in two distinct tracks, each designed to stress-test different aspects of multi-agent reasoning and execution.

### **Track 1: Planning-Oriented Multi-Agent Orchestration**

This track challenged participants to design prompts that convert complex multi-agent interactions into clear, structured DAG-style plans. Successful agents needed to demonstrate correct sequencing, communication across agents, selective tool usage, fallback behavior, and overall orchestration quality.

### **Track 2: Execution-Oriented Dynamic Multi-Agent Workflow**

Track 2 focused on the execution side. Participants were asked to move beyond rigid sequential pipelines and develop adaptive, fault-tolerant workflows capable of parallel operation, multi-agent collaboration, shared context propagation, and dynamic decision paths that respond to evolving task states.

---

## Methodology for Selecting Development vs Testing Instances

### **Track 1 Feedback Summary**

Track 1 served as the foundation for analyzing how agents behave under realistic, multi-step industrial constraints. Each submission was evaluated across all six dimensions using a privacy-preserving LLM-as-Judge protocol.

Even without revealing execution traces, the evaluation system produced actionable diagnostic signals by analyzing tool-invocation patterns, intermediate outputs, and reasoning structures grounded in real telemetry and work-order data.

The resulting failure clusters exposed failure modes that almost never surface in synthetic benchmarks:
- missing-data handling failures  
- verification inconsistencies  
- reasoning–action mismatches  
- hallucinations  
- improper or unsafe tool usage  

This structured feedback loop allowed developers to pinpoint whether weaknesses stemmed from planning, agent coordination, or execution logic. Ultimately, Track 1 demonstrated the value of iterative, failure-mode–driven refinement: agents became more robust only after developers understood how drift, ambiguity, and multi-hop dependencies break naive agent designs.



### **Track 2 Utterance Selection**

Every utterance in AssetOpsBench-Live belongs to a specific operational category—intent interpretation, alert triage, meter analytics, maintenance recommendations, multi-agent instructions, or contextual reasoning. Track 1’s utterances were manually curated to ensure balanced representation across these categories and to incorporate realistic issues such as telemetry gaps, ambiguous alerts, and procedural constraints.

For Track 2, we expanded the dataset using a controlled semantic-neighbor strategy:

1. **For each Track-1 utterance**, we identified its top semantic neighbors *within the same operational category*.  
2. We **filtered out** any neighbors already used in Track 1.  
3. We **selected the closest unused neighbor** as the Track-2 counterpart.

This procedure generated paired prompts that share operational grounding but differ in phrasing, contextual cues, or linguistic structure—forcing agents to generalize rather than memorize.

The resulting Track-2 dataset maintains:
- balanced task-type distribution  
- controlled semantic diversity  
- robustness against overfitting  
- realistic phrasing-drift challenges  

This methodology ensures that agents are evaluated not only on correctness, but also on **consistency, reasoning stability, and resilience to subtle linguistic variation**, all of which are essential for real industrial deployments.

---
# Community Results: 225 Users, 300+ Agents, One Reality Check

Over the course of the evaluation:

- **225+ users** participated  
- **300+ containerized agent submissions** were executed  
- Across **two tracks** (Planning, Execution)  
- Using **dozens of custom agent architectures** and coding styles  

This is one of the largest-scale industrial agent evaluations to date.

## Participation Insights and Competition Dynamics

The AssetOpsBench-Live challenge ran continuously over a three-month window, allowing participants to iteratively refine their agents while receiving structured, privacy-preserving diagnostic feedback. This long-running format produced several notable patterns:

- **Sustained participation:** Hundreds of submissions arrived steadily throughout the competition, reflecting sustained engagement over the full duration of the challenge. By lowering the barrier to entry with container templates, sample agents, and clear instructions for extending the agent codebase, we enabled both newcomers and experienced practitioners to participate effectively.

- **Developer experience improvements:** By providing a lightweight entry point for extending or creating new agents, we ensured that participants did not need deep familiarity with industrial systems to begin experimenting. This broadened the contributor base and allowed teams to focus on reasoning strategies instead of infrastructure setup.

- **Submission distribution:** Although we cannot reveal platform URLs or identifiable metadata, aggregated distributions showed diverse iteration behaviors across teams. Some participants submitted frequently, steadily improving their agents through diagnostic feedback, while others converged quickly with only a few targeted submissions. These heterogeneous usage patterns align with the feedback-driven design of AssetOpsBench-Live.

These participation dynamics demonstrate that real-world industrial evaluation can be made accessible without sacrificing rigor, enabling a broad community to meaningfully engage with multi-agent system development.

## Leaderboard Behavior

We evaluated leading frontier and open-source models:

- gpt-4.1  
- mistral-large  
- llama-4-maverick  
- llama-4-scout  
- llama-3-70b  
- granite-3-8b  

Key observations:

- Tool-as-Agent significantly outperformed Plan-Execute in correctness  
- Frontier models led, but none surpassed reliability thresholds required for industrial deployment  
- Smaller models collapsed under multi-step reasoning and retrieval constraints  

## Observed Failure Patterns

Across **881 recorded trajectories** (from AssetOpsBench), failure categories revealed systemic weaknesses:

- **31.2%** — ineffective error recovery  
- **23.8%** — overstated task completion  
- **21.4%** — ambiguous or extraneous formatting  
- **10.3%** — unhandled tool errors  
- **8%** — failure to incorporate feedback  

These patterns also appeared consistently in the Codabench evaluations.

## Emergent Failure Modes

Beyond the 14 predefined failure modes, agents produced:

- **185 traces with one new failure mode**  
- **164 traces with two new emergent failures**  

Semantic clustering revealed 6 canonical emergent categories, including:

- incorrect error handling  
- invalid action formatting  
- mistaken internal state assumptions  
- compound failures involving multiple sub-agents  

This underscores the need for *adaptive* taxonomies as agent complexity increases.

---

# Evaluation and Ranking Methodology

This section defines the official scoring, selection, and ranking workflow used in the AssetOpsBench-Live multi-agent challenge. The design centers on a privacy-preserving **LLM-as-a-Judge** evaluation protocol that prioritizes correctness, safety, and deployability rather than simple text matching.

## 1. Continuous Evaluation on Public Scenarios

During the active competition phase, each submitted containerized agent is evaluated on **11 scenarios**:
- **1 public reference scenario**
- **10 dynamic hidden scenarios** that change for every submission

Each scenario is scored across six binary criteria (1 = Pass, 0 = Fail) included in the evaluation framework.

### Score Calculation

Each submission receives a **Score_Public**, defined as:

`
Score_Public = ( Σ(Passed Criteria) / (Total Scenarios × 6) ) × 100`
`

- **Total scenarios:** 11  
- **Maximum possible points:** 66  

Participants also receive **clustered diagnostic feedback** (e.g., *Reasoning–Action Mismatch*, *Missing-Data Handling Error*) to iteratively refine their agents without revealing private industrial data.

---

## 2. Selection of the "Best Submission"

To prevent overfitting to dynamic public scenarios, exactly **one submission per team** is selected for final evaluation.

**Selection rule:** the submission with the **highest Score_Public** during the competition window.

Only this frozen container is used for the private evaluation phase.

---

## 3. Final Evaluation on Private Scenarios

The frozen agent is then evaluated on a held-out set of **10 private scenarios**, identical across teams to ensure a fair final benchmark.

Two metrics are computed:

### A. Score_Private (0–100)
Computed using the same six-dimensional binary scoring framework as Score_Public.

### B. Tau-match (τ-match) (0–1)
A semantic trajectory similarity metric comparing the agent’s reasoning sequence to a human-authored **Gold Standard**.  
This captures whether the agent *thought correctly*, not just whether it produced the right final answer.

---

## 4. Composite Track Scoring

Each track (Planning and Execution) receives a composite score:

`Composite_Track = 0.60 × Score_Public + 0.30 × Score_Private + 0.10 × Tau-match`


This weighting reflects:
- performance during the live competition,
- generalization to private scenarios,
- reasoning quality.

---

## 5. Final Weighted Ranking

Industrial deployments prioritize **reliable execution** over planning alone.  
Therefore, the final score aggregates track performance with execution weighted higher:

`Final Score = 0.40 × Composite_Planning + 0.60 × Composite_Execution`


Teams are ranked in descending order of this Final Score.

---

## 6. Tie-Breaking Rule

If teams tie on Final Score:

1. **Primary tie-breaker:** Higher Composite_Execution score  
2. **Secondary tie-breaker:** If identical, ranks are shared  

---

## Observations from the Final Results

<figure>
  <img src="../assets/img/R1.png"
       alt="Final results"
       style="width:100%; max-width:1100px;">

  <figcaption>
    <b>Figure.</b> Distribution of Team Performance Across Planning, Execution, and Final Composite Scores
  </figcaption>
</figure>

- **Planning track:** larger variation (Std Dev ≈ 9.6), reflecting diverse reasoning strategies  
- **Execution track:** tighter clustering (Std Dev ≈ 5.7), suggesting greater consistency once a plan is defined  
- **Overall:** mean (53.81) and median (53.65) are nearly identical → a highly balanced competition  

---

## Impact of the Feedback Mechanism

We analyzed how team performance evolved across submissions using first-vs-best score curves, submission count correlations, and improvement distributions.

<figure>
  <img src="../assets/img/R2.png"
       alt="Final results"
       style="width:100%; max-width:1100px;">

  <figcaption>
    <b>Figure.</b> Effect of Iterative Feedback on Agent Performance Across the Competition
  </figcaption>
</figure>

Key findings:

- Most teams **significantly improved** from their first to best submissions, often by **20–50 points**
- More iterations correlated with stronger peak performance  
- Iterative refinement based on diagnostic feedback clearly accelerated learning  

These trends show that the **feedback-driven evaluation loop** was essential for helping participants:
- identify failure modes,
- refine tool usage,
- correct reasoning patterns,
- and progressively improve multi-agent reliability.

---
# Case Study: Missing Clarifying Questions and Its Impact

One concrete insight from the challenge:

> Nearly 10% of all failures traced back to **not asking clarifying questions** when required.

Enabling clarifying-question behavior (`ask=True`) produced dramatic improvements:

| Model | Before | After |  
|-------|--------|--------|  
| LLaMA-4 Maverick | 59% | **66%** |  
| LLaMA-3-405B | 44% | **61%** |  

This demonstrates the value of targeted diagnostics — participants used feedback to improve models, not guesses.

---

# What We Learned About Industrial Agents

From both benchmark and competition data, several conclusions are unavoidable:

1. **Agents frequently hallucinate successful completion without performing required steps.**  
2. **Correct tool invocation is much harder than textual reasoning.**  
3. **In-context examples are necessary for 80%+ of successful runs.**  
4. **Plan-Execute agents over-plan even simple tasks.**  
5. **Tool-As-Agent is more accurate but slower — a practical deployment tradeoff.**  
6. **Evaluation must consider multi-agent coordination, not just single-turn quality.**  
7. **Human-like ambiguity (missing data, drift, inconsistent logs) drastically increases failure rates.**

These insights do not appear in synthetic agent benchmarks — only in real industrial workflows.

---

# Implications for Agentic AI

Our results suggest several broader implications:

- **Current agentic architectures are not ready for safety-critical industrial automation.**  
- **Benchmarks need real operational context**, not sanitized tasks.  
- **Failure-mode discovery should be a built-in component** of agent evaluation.  
- **Competitions accelerate agent improvement more than paper benchmarks alone.**  
- **Privacy-aware evaluation is necessary for industry adoption.**

Industrial AI cannot rely on prompt engineering and small-scale toy tasks.  
We need rigorous, reproducible, scenario-grounded evaluation pipelines.

---

# Future Directions

Building on this work, we plan to:

- incorporate compute-cost constraints  
- add real-time latency and deadline requirements  
- expand to new industries (steel, energy, data centers)  
- introduce richer multi-agent orchestration tasks  
- develop open leaderboards with long-horizon evaluations  
- evaluate agent reward models and long-term memory systems  

These extensions aim to push agentic AI toward real deployability.

---

# Conclusion

Industrial environments expose weaknesses in LLM agents that traditional benchmarks never reveal.  
By combining AssetOpsBench with a Codabench-powered real-time pipeline, we created a community-scale, privacy-preserving benchmark that evaluates agents not on their *style*, but on their *actions*.

The results are unequivocal:

**Agentic AI is promising — but not yet reliable for real-world industrial operations.  
Robust evaluation is the missing link, and community-driven benchmarks are the path forward.**

