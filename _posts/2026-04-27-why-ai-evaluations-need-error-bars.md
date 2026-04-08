---
layout: distill
title: "Why AI Evaluations Need Statistical Rigor"
description: "AI evaluations often rely on single-run scores even though models, agents, and judges are inherently stochastic, making many reported differences unstable. This post surveys statistical tools—error bars, reliability measures, Bayesian models—that show and help manage this variance. Overall, it highlights how incorporating established statistical practices can make evaluations more trustworthy and informative."
date: 2026-04-27
future: true
authors:
  - name: Zairah Mustahsan
    affiliations:
      name: You.com

tags: [evaluation, statistics, llms, agents, benchmarking]
math: true

bibliography: 2026-04-27-why-ai-evaluations-need-error-bars.bib

toc:
  - name: "Introduction"
  - name: "1. LLMs as Stochastic Systems"
    subsections:
      - name: "1.1 Intrinsic sources of stochasticity"
      - name: "1.2 Stochasticity in multi-step agent behavior"
      - name: "1.3 Stochasticity in evaluation: LLM-as-judge"
      - name: "1.4 Environmental and contextual randomness"
  - name: "2. Why Stochasticity Breaks Evaluation"
  - name: "3. Existing Research: Bringing Statistics into LLM Evaluation"
  - name: "4. What is Still Missing"
  - name: "5. Moving Forward"
  - name: "6. Conclusion"
---


## Introduction

As large language models (LLMs) and agentic systems advance, the field increasingly depends on fine-grained evaluation to compare models, guide research directions, and make deployment decisions. Yet evaluation pipelines often treat LLMs as deterministic functions, even though they are fundamentally stochastic systems with variability arising from sampling methods, hardware nondeterminism, environmental randomness, and evaluation procedures.

This mismatch leads to unstable benchmarks, unreliable model comparisons, inconsistent agent outcomes, and significant uncertainty when using LLMs as judges. Recent research has begun to quantify this instability and propose statistical techniques, from frequentist error bars to Bayesian latent-state models, reliability metrics, and large-scale variance audits. But adoption is uneven, and the field lacks a cohesive statistical framework for evaluating stochastic intelligence.

This post synthesizes existing research into a unified perspective and outlines practical recommendations for improving evaluation practice. The goal is not to introduce new methods, but to demonstrate that the tools already exist and that incorporating statistical thinking is both feasible and urgently needed.

---

## 1. LLMs as Stochastic Systems

Understanding why evaluation requires statistical tools begins with acknowledging that LLMs and agents are not deterministic systems.

### 1.1 Intrinsic sources of stochasticity
Even with the temperature set to 0, outputs are not fully stable. Song et al. (2024) <d-cite key="song2024llmstability"></d-cite> demonstrated that LLMs exhibit variability across multiple runs at temperature=0 due to:
- GPU-level nondeterminism
- floating-point implementation differences
- parallel kernel non-associativity
- asynchronous operations

The study evaluated GPT-3.5 Turbo, GPT-4o, and Llama-3 models across multiple tasks and introduced several metrics to quantify stability:
- Total Agreement Rate (TAR): proportion of prompts for which all repeated runs produced identical raw outputs
- TARa: agreement after normalizing/cleaning outputs
- Min–max spread: difference between minimum and maximum accuracy across runs

Across models, TAR was low even for seemingly deterministic settings, and TARa, while higher, was far from perfect. The conclusion: LLMs are rarely deterministic at the raw output level.

### 1.2 Stochasticity in multi-step agent behavior
Agent benchmarks compound these effects. In systems like SWE-bench <d-cite key="swebench2023"></d-cite>, 
WebArena <d-cite key="webarena2023"></d-cite>, 
or OSWorld <d-cite key="osworld2024"></d-cite>, 
identical starting conditions do not guarantee identical outcomes. The updated DevStral benchmark<d-cite key="devstral2025"></d-cite>, for example, explicitly uses three independent attempts per task even at temperature 0 because of persistent nondeterminism: file system state, metadata changes, and agent-induced environmental variability.

The KAMI benchmark (2024)<d-cite key="kami2024benchmark"></d-cite>, analyzing over 5.5 billion tokens across 170,000 test items, concluded that:
“LLM behavior can be notoriously stochastic, so agentic fitness must demonstrate run-to-run reliability in addition to overall accuracy.”

Therefore, the success of an agent can depend on stochastic choices made early in an execution chain.

### 1.3 Stochasticity in evaluation: LLM-as-judge
LLM-as-judge, widely used to scale human evaluation, introduces a second layer of variability:
- The judge is a stochastic model.
- Its outputs depend on prompt templates and sampling settings.
- Small linguistic changes in candidate answers lead to different judgments.
- Model biases and prior knowledge shape scoring behavior.

LLMEval-3 (2024)<d-cite key="llmeval32024"></d-cite>, a longitudinal study, found judgment variability of 0.5–1.6 percentage points across repeated judge evaluations, even when the target model outputs were fixed, and only judge randomness varied.

### 1.4 Environmental and contextual randomness
Other sources of variance include:
- context-window effects
- ordering sensitivity
- retrieval randomness in RAG-based agents
- tool-use variability
- system-level interference (e.g., time-sensitive APIs)

Even benign environmental differences can shift evaluation outcomes.

<p align="center">
  <img src="{{ '/assets/img/2026-04-27-why-ai-evaluations-need-error-bars/layers-stochasticity.png' | relative_url }}" width="100%">
</p>
<p align="center"><em>Figure 1: Sources of stochasticity across model, agent, and judge layers.</em></p>


## 2. Why Stochasticity Breaks Evaluation
Recognizing that LLMs and agents are stochastic systems is only the first step. The deeper issue is how this stochasticity interacts with evaluation pipelines that implicitly assume stability, independence, and reproducibility. Once those assumptions break, many standard evaluation methods—single-run accuracy, point estimates, deterministic judge calls, and even widely used significance tests—no longer behave as expected. The consequences show up across three levels: model outputs, evaluation metrics, and evaluation processes.

### 2.1 Instability of single-run evaluations
The simplest and most pervasive failure mode appears when evaluations rely on a single run of a model. Because LLMs produce samples from a distribution, not deterministic answers, repeated executions of the same benchmark can yield different results. Meta’s Quantifying Variance in Evaluation Benchmarks (2024)<d-cite key="meta2024quantifying"></d-cite> demonstrated that changing only the evaluation seed—without modifying the model, prompt, or data—can shift accuracy by 1–3 percentage points for mid-sized models. Even for frontier models, where variance is typically smaller, the effect remains measurable.

This leads to a second problem: _non-monotonicity_ across checkpoints. Model families that appear to improve steadily in capability sometimes show temporary regressions or fluctuations simply because single-run evaluations under-measure variance. Without repeated sampling, these fluctuations look like real performance changes rather than noise.

Taken together, these results mean that small differences in point estimates—often the basis for leaderboard rankings or ablation conclusions—may not reflect meaningful differences in capability at all.

### 2.2 Violations of independence assumptions
Most evaluation metrics assume that observations are independent. In practice, few modern LLM tasks satisfy this condition. Reading comprehension datasets contain multiple questions tied to the same passage; agent benchmarks contain multi-step trajectories where errors propagate; and most preference or open-ended evaluations cluster items by topic, task type, or difficulty.

When items form clusters, the effective sample size is dramatically smaller than the raw number of questions suggests. Cluster effects inflate variance, making standard error estimates overly optimistic. Miller (2024)<d-cite key="miller2024errorbars"></d-cite> formalizes this through the design effect, showing that intracluster correlation coefficients in the 0.2–0.4 range—common in natural language datasets—can increase true standard errors by a factor of two or more. If these dependencies are ignored, significance tests routinely declare differences meaningful when they are not.

### 2.3 Compounding uncertainty in multi-step and agent evaluations
Agent evaluations introduce an additional layer of complexity: uncertainty compounds over time. A single stochastic choice early in a trajectory—retrieving one document rather than another, selecting a slightly different plan, or producing a subtly different chain-of-thought—can cascade into entirely different outcomes. Even when the temperature is set to zero, nondeterministic hardware operations, environmental randomness, and tool responses lead to divergent behaviors.

This makes multi-step tasks particularly sensitive to variance. The DevStral and SWE-bench evaluations explicitly rerun each agent multiple times because identical initial conditions do not guarantee identical outcomes. In these settings, averaging over a single attempt per task misrepresents the distribution of possible behaviors, and comparing agents using only mean accuracy hides important properties like reliability, robustness, and failure modes.

### 2.4 Human evaluation as a noisy measurement process
Human judgments are often presented as ground truth, but they, too, behave like stochastic measurements. Human raters differ in expertise, familiarity with task expectations, and tolerance for ambiguity; they fatigue over time; and they apply evaluation rubrics inconsistently. Psychometrics has long used reliability coefficients to quantify these effects, yet such metrics are rare in LLM evaluation.

Seo et al. (2024) <d-cite key="seo2024medicaldocs"></d-cite>, in their study of LLM-generated medical documentation, measured inter-rater reliability using the intraclass correlation coefficient (ICC) and found moderate to strong agreement, but not perfect consistency. The study also showed that test–retest reliability was less than 1.0, meaning that evaluators changed their minds when re-rating the same outputs. Most LLM or agent benchmarks do not report such metrics, making it difficult to know whether human evaluations are stable enough to support fine-grained model comparisons.

### 2.5 A second-order source of variance: LLM-as-judge
Perhaps the most consequential evaluation method affected by stochasticity is LLM-as-judge. Because LLMs are now widely used to replace or supplement human raters, any variance in judging behavior directly translates into variance in evaluation scores.

LLMEval-3 (2024)<d-cite key="llmeval32024"></d-cite> quantified this by holding model outputs fixed and varying only the judge’s sampling. Even under controlled conditions, scores varied by 0.5–1.6 percentage points across repeated judgments. These fluctuations arise from several factors: stochastic sampling in the judge itself, sensitivity to prompt phrasing, asymmetries in how different models articulate reasoning, and systematic preferences embedded in the judge’s training data.

Crucially, when the model under evaluation and the judge are both stochastic, the total uncertainty is additive. Yet most evaluation pipelines treat judge outputs as if they were deterministic decisions. This creates a false sense of precision: a model “wins” a comparison because the judge happened to prefer it in a noisy setting.


## 3. Existing Research: Bringing Statistics into LLM Evaluation
The measurement failures described above are not new to other fields. Statistics, psychometrics, and biostatistics have long dealt with noisy measurements, clustered data, and imperfect raters. Over the last one to two years, several works have begun importing these tools into LLM and agent evaluation. This section summarizes some of the most relevant strands: frequentist error bars and hypothesis tests, Bayesian measurement error models, reliability metrics, agent-specific issues, and variance-aware benchmarking.

### 3.1 Confidence intervals and hypothesis testing
Evan Miller’s *Adding Error Bars to Evals* (2024)<d-cite key="miller2024errorbars"></d-cite> makes a straightforward but under-applied argument: evaluation is an experiment, and we should analyze it like one.  
If a model is evaluated on $(n)$ questions with empirical accuracy $(\hat{p})$, a first-pass standard error is:

$$
SE \approx \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}
$$

This yields confidence intervals and lets us test whether differences between models are statistically significant. The core conceptual move is to treat evaluation questions as samples from a larger “super-population” of possible items—essentially, acknowledging that the benchmark is only a finite window into a much larger task space.

<p align="center">
  <img src="{{ '/assets/img/2026-04-27-why-ai-evaluations-need-error-bars/point-vs-ci.png' | relative_url }}" width="100%">
</p>
<p align="center"><em>Figure 2: Overlapping confidence intervals show ambiguous rankings. 
Two models that differ by 1 percentage point in accuracy but have overlapping 95% confidence intervals. Without error bars, this looks like a clear ranking; with uncertainty, the “winner” is ambiguous.</em></p>


However, this simple binomial model breaks down as soon as questions are no longer independent. In reading comprehension, multiple questions may refer to the same passage; in agent tasks, one step influences the next; in structured eval suites, items are grouped by document, topic, or environment. In such settings, the effective sample size is much smaller than the raw item count.

Miller addresses this using cluster-robust standard errors. If questions cluster—say, five questions per passage across twenty passages—the relevant unit is often the passage, not the question. The design effect,

$$
DE = 1 + (m - 1)\rho
$$

where $(m)$ is the cluster size and $\rho$ is the intracluster correlation, captures how much variance is inflated by within-cluster similarity. For realistic values of $\rho$ (e.g., 0.2–0.4), this can increase standard errors by a factor of two or three, turning apparently “significant” model differences into overlaps once uncertainty is correctly computed.

For planning-style evaluations or A/B comparisons, the paper goes further and provides sample size formulas. To detect a minimum effect $\delta$ at significance level $\alpha$ with power $(1-\beta)$:

$$
n \ge (z_{\alpha/2} + z_{\beta})^2 \cdot \frac{2\sigma^2}{\delta^2}
$$

where $\sigma^2$ can be estimated from pilot evaluations. The recommendation is practical: report confidence intervals alongside point estimates, adjust for clustering when items are not independent, and plan evaluation sample sizes to achieve meaningful power rather than recycling whatever dataset happens to be available.

### 3.2 Bayesian measurement error models

Zhang and Martinez’s *From Stochasticity to Signal* (2024)<d-cite key="zhang2024stochasticity"></d-cite> reframes LLM stochasticity as a measurement error problem and uses it rather than fighting it.  
Consider using an LLM to classify text—for instance, labeling customer satisfaction. For each item $(i)$, we can query the LLM multiple times and obtain different ratings $(r_{ij})$. A naive pipeline either:

- takes the first output as “ground truth,” ignoring uncertainty, or  
- uses majority voting across samples, which aggregates noise but does not model it.

Zhang and Martinez instead treat the unknown true label $(y_i)$ as a latent variable, and the multiple LLM responses $\{r_{ij}\}$ as noisy observations of that latent state. For a binary task, the model includes:

- $(y_i)$: latent true class for item $i$  
- $(r_{ij})$: the $j$-th LLM rating for item $i$  
- $(\theta)$: base rate of the positive class  
- $(\alpha)$: false positive rate  
- $(\beta)$: false negative rate  

Using a Bayesian latent class model, they jointly infer $(y_i)$, $(\theta)$, $(\alpha)$, and $(\beta)$ from the observed ratings. This connects directly to long-standing work in biostatistics on evaluating diagnostic tests when no gold standard exists.

The practical payoff is twofold:

- **Item-level:** the model yields posterior probabilities for each label rather than brittle point decisions.  
- **Population-level:** it produces calibrated estimates of base rates and their uncertainty—crucial for decisions like “did this change actually improve user satisfaction?”

This framework also extends to the case where the LLM is used as a noisy annotator for training data. <d-cite key="baumann2025noisylabels"></d-cite> show that such noisy labeling can introduce prevalence-dependent bias into downstream evaluations; explicit modeling of the error process, as in Zhang and Martinez, offers a principled way to correct for this.

### 3.3 Reliability metrics from psychometrics

Psychometrics has long treated human evaluation as a noisy measurement process and developed tools to quantify reliability. One such tool is the intraclass correlation coefficient (ICC), which measures how consistently different raters evaluate the same items.

Seo et al. (2024)<d-cite key="seo2024medicaldocs"></d-cite>, in a study of LLM-generated medical documentation, asked four physicians to rate emergency department records across multiple dimensions—appropriateness, accuracy, structure, conciseness, and clinical validity. They computed ICC$(3,k)$, a two-way mixed-effects model for absolute agreement, and found values between 0.653 and 0.887: moderate to strong reliability, but far from perfect. Test–retest reliability, measured by having the same evaluators rate the same records at different times, was also below 1.0, indicating that human judgments drift.

Formally, ICC can be expressed in terms of ANOVA components:

$$
ICC = \frac{BMS - WMS}{BMS + (k - 1)WMS}
$$

where $(BMS)$ is the between-subject mean square, $(WMS)$ is the within-subject mean square, and $(k)$ is the number of raters. This ratio reflects the fraction of total variance attributable to true differences between items, as opposed to rater noise.

Despite being well suited to multi-rater, multi-item settings, ICC and similar metrics are rarely reported in LLM evaluation. This is especially notable in agent benchmarks, where repeated trials on the same tasks are standard and run-to-run variability is high. In those settings, reliability metrics would quantify how much of the observed variance stems from the system under test versus the evaluation procedure itself.


### 3.4 The current state of agent evaluation

Agent evaluations introduce additional statistical challenges that are only partly addressed in current work. Surveys such as Yehudai et al. (2025)<d-cite key="yehudai2025agentsurvey"></d-cite> and Trivedi et al. (2024)<d-cite key="trivedi2024agentsmatter"></d-cite> 
 point to several recurring issues.

A first issue is the **task realism vs. reproducibility tradeoff**. Complex environments like WebArena or OSWorld more closely resemble real-world scenarios but introduce many uncontrolled variables, making it hard to attribute performance differences to model changes alone. Simpler environments like HumanEval offer cleaner measurement but cover a narrow slice of potential behavior.

A second challenge is **metric inconsistency**. Different frameworks emphasize different objectives—task success rate, functional correctness, time-to-completion, cost, or human preference—making cross-benchmark comparison difficult. AgentBench, WebArena, AssistantBench, and KAMI all operationalize “doing well” differently, and most collapse those metrics into a single figure without reporting uncertainty.

A third factor is the heavy reliance on **LLM-as-judge** to score outputs. Judge stochasticity and bias introduce variance that is rarely modeled explicitly. LLMEval-3<d-cite key="llmeval32024"></d-cite> mitigates this by repeating evaluations and reporting fluctuations, showing less than two percentage points of variability across runs, but this is still not standard practice.

Finally, **cost** often receives too little attention. *AI Agents That Matter* (Trivedi et al., 2024)<d-cite key="trivedi2024agentsmatter"></d-cite>  shows that maximizing accuracy alone can lead to over-engineered, expensive agents that are not Pareto-efficient. Simple baselines that exploit stochasticity—such as retrying a call several times at temperature 0—can match or exceed the performance of more elaborate architectures on benchmarks like HumanEval at dramatically lower cost. This makes a strong case for treating evaluation as a multi-objective comparison rather than a one-metric contest.

### 3.5 Quantifying variance in practice: seeds, items, and distributions

Several recent works study evaluation variance directly at scale. Meta’s *Quantifying Variance in Evaluation Benchmarks* (2024)<d-cite key="meta2024quantifying"></d-cite> systematically measures how benchmark results shift under different random seeds, bootstrap resampling, and model checkpoints. Key findings include:

- variance is substantial and task-dependent,  
- pretraining checkpoints can differ by 1–3 percentage points across seeds, and  
- even frontier models show meaningful variability under repeated evaluation.

To make better use of limited evaluation budgets, Meta applies **item response theory (IRT)**, a framework from educational testing that models each item’s difficulty and each model’s latent ability. IRT enables selecting subsets of items that are maximally informative about ability, reducing evaluation size without compromising measurement precision.

Complementary work on **risk-aware benchmarking** treats evaluation as comparing multivariate performance distributions rather than just average metrics. Tools such as stochastic dominance or copulas allow comparisons that focus on worst-case behavior or tail risk, and significance is assessed using bootstrapped confidence intervals. This brings distributional thinking into the center of model comparison rather than leaving it as an afterthought.

## 4. What is Still Missing

Taken together, the work surveyed so far shows that the community has begun to import serious statistical ideas into LLM and agent evaluation. Yet several gaps remain before these ideas become routine practice rather than isolated examples.

The first and most visible gap is the lack of standardized reporting for uncertainty. Some papers now report confidence intervals, but many still present only point estimates. Among those that do include uncertainty, methods vary widely: bootstrap versus analytic approximations, different confidence levels, and different handling (or ignoring) of clustering. Miller (2024)<d-cite key="miller2024errorbars"></d-cite> proposes concrete reporting formats and examples, but there is not yet a shared convention analogous to what exists in other empirical sciences.

A second missing piece is agent-specific reliability analysis. Metrics like ICC and test–retest reliability are directly applicable to multi-turn agent tasks, where repeated runs on the same scenario are the norm. In principle, they could help disentangle how much variance is due to the underlying task difficulty versus the measurement process or the stochasticity of the agent itself. In practice, they almost never appear. In our survey of the literature, we found only isolated uses of such metrics in agent-like settings; most work reports success rates or pass@k values without quantifying how stable those numbers are across runs.

A third underdeveloped area is benchmark contamination and longitudinal behavior. Statistical tests exist for identifying suspicious performance patterns that might indicate data leakage or partial exposure—for example, item response patterns that are inconsistent with a clean training process—but they are rarely applied. Similarly, we have limited visibility into how ranks and gaps evolve over time as models are updated via fine-tuning, RLHF, or context distillation. Are models that are close on a benchmark snapshot also close six months later? Do small gaps persist, vanish, or invert? These are, at their core, statistical questions about stability and drift, but they are rarely treated as such.

The field also mostly ignores multiple testing and effect sizes. When ten models are compared on twenty benchmarks, the number of implicit pairwise comparisons is large, and so is the chance of spurious “wins.” Family-wise error rate control (e.g., Bonferroni or Holm–Bonferroni) and false discovery rate methods are standard in many other domains; in LLM evaluation they are almost absent. Likewise, statistical significance is often reported without any measure of practical significance. For many decisions, Cohen’s d, odds ratios, or percentage-point effect sizes with context would be more informative than p-values alone.

Finally, there is a broader gap around distributional thinking. Most evaluations still focus on average-case metrics, even when the underlying applications are risk-sensitive. Work on risk-aware benchmarking, stochastic dominance, and tail-focused comparisons is promising, but not yet integrated into everyday practice. Similarly, tools like item response theory, which make evaluation more sample-efficient while preserving measurement precision, are still exceptions rather than defaults.

---

## 5. Moving Forward

The encouraging part of this story is that the necessary tools already exist. Confidence intervals, hypothesis tests, measurement error models, reliability coefficients, and item response theory have all been developed, debated, and refined in other domains. The open question is not whether we can build new statistics for LLMs, but whether we are willing to treat statistical evaluation as a default rather than an optional embellishment.

Several practical themes emerge from the work surveyed here.

One is the value of leaning into stochasticity instead of trying to wish it away. Approaches like the retry baseline in *AI Agents That Matter* and the Bayesian measurement model in *From Stochasticity to Signal* <d-cite key="zhang2024stochasticity"></d-cite> treat variability as additional information. Multiple runs are not “noise to be averaged out” so much as repeated measurements of the underlying process. When evaluations are designed to sample that variability deliberately, they yield a clearer picture of both central tendency and spread.

Another theme is the importance of reporting uncertainty as a matter of routine. As Miller (2024)<d-cite key="miller2024errorbars"></d-cite> illustrates, computing standard errors and confidence intervals for many common metrics is straightforward, and cluster-aware adjustments are conceptually simple once the data structure is explicit. For most projects, the engineering effort required to implement these calculations is small compared to the effort of training, fine-tuning, and serving the models being evaluated. What is missing is habit: readers expect single numbers, and authors optimize for that expectation.

A further pattern is the need to match variance estimators to the structure of the data. When items cluster, when trajectories share prefixes, or when tasks are dependent, naive formulas systematically understate uncertainty. Cluster-robust methods, design effects, and simple resampling techniques are standard in applied statistics and econometrics. Building these into shared evaluation libraries would allow most researchers to benefit from them without needing to re-derive anything.

There is also a growing recognition that accuracy alone is not the only axis that matters. Especially for agents, real-world deployments care about latency, cost, energy, stability, and sometimes worst-case rather than average-case behavior. Pareto frontiers that jointly display accuracy and cost, or robustness and performance, make trade-offs visible that single-number leaderboards obscure. That kind of visualization can change which models or architectures look attractive.

<p align="center">
  <img src="{{ '/assets/img/2026-04-27-why-ai-evaluations-need-error-bars/accuracy-cost-pareto.png' | relative_url }}" width="100%">
</p>
<p align="center"><em>Figure 3: Hypothetical accuracy–cost trade-off for different agent architectures
Simple retry-style baselines can be closer to the Pareto frontier than more elaborate agents, suggesting that evaluation should account for both performance and cost.</em></p>

Finally, it is hard to ignore how much could be gained by borrowing systematically from neighboring fields. Psychometrics offers tools like ICC and test–retest frameworks; educational testing brings IRT and principled item selection; biostatistics provides latent class and measurement error models; econometrics contributes methods for clustered, panel, and longitudinal data. Many of the evaluation pathologies in LLMs—noisy labels, imperfect judges, clustered items, moving distributions—look very familiar through those lenses.

The aim is not to make evaluation more elaborate for its own sake. The goal is to make evaluation honest: explicit about uncertainty, grounded in appropriate estimators, and designed to separate real improvements from fluctuations that are inevitable in stochastic systems. The statistical foundations are already in place. The remaining work is cultural—deciding that high-stakes claims about model capability and safety should rest on measurement practices that are as mature as the models they assess.

---

## 6. Conclusion

LLMs and agents are stochastic systems. Their evaluations—whether through benchmarks, human raters, or automated judges—are themselves stochastic processes. Ignoring this reality leads to misleading scientific claims, unstable engineering systems, and unreliable deployment decisions.

The statistical tools needed to address these challenges already exist. They come from biostatistics, psychometrics, econometrics, machine learning, and evaluation science. Integrating them into standard practice is less a technical problem than a cultural one.

Adopting statistical thinking will not make evaluation harder—it will make it trustworthy. And in a field increasingly shaped by small margins, complex agents, and automated judgment systems, trust in evaluation is essential.
