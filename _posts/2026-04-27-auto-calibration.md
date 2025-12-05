---
layout: distill
title: (LLM-)Judges on autopilot
description: "How do you evaluate Large Language Model (LLM)-based systems in production at scale? Most teams turn to an LLM-as-a-judge: an approach that grasps the nuances of natural language where classical metrics fall short. But these judge models have their own “will”: sometimes they follow instructions precisely, sometimes they don't. To address this inconsistency, the judge prompt is <i>calibrated</i> to align with known, trusted cases. The problem? Manual calibration is time-consuming and error-prone. In this blog post, we explore auto-calibration techniques inspired by recent prompt-optimization research. We tackle <i>context collapse</i> by iteratively processing data in batches, similarly to a machine learning training pipeline. Along the way, we share some surprising findings about what works and what doesn't—including cases where simpler approaches outperform more sophisticated ones."
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
authors:
  - name: Anonymous

# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-auto-calibration.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: What is an LLM-as-a-judge?
  - name: What are we evaluating?
  - name: What about the data?
    subsections:
      - name: The challenge of high-quality ground truth
  - name: Who are you to judge me?
    subsections:
      - name: "Our first attempt: manual calibration"
      - name: Take your cheatsheet out, the game begins
      - name: "ACE Up Your Sleeve: Agentic Context Engineering"
      - name: Any BACI, please?
  - name: What did we learn?
    subsections:
    - name: "Starting with the basics: when less is more"
    - name: "Learning in batches: the missing ingredient"
    - name: The final verdict
  - name: Key Takeaways


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
    font-size: 16px;
  }
---

<div class="row mt-4">
    <div class="col text-center">
        {% include figure.liquid path="assets/img/2026-04-27-auto-calibration/judge_funny.png" class="img-fluid d-block mx-auto" max-width="90%" %}
    </div>
</div>

## What is an LLM-as-a-judge?

Modern evaluation pipelines increasingly rely on LLM-as-a-judge <d-cite key="zheng2023llmjudge"></d-cite> to assess the quality of AI-generated responses. This approach uses an LLM to act as a judge, assessing the quality of the generated outputs against specific criteria. The judge can be either a different model or the same one that generated the response. An example prompt for the <i>answer relevance</i> metric is shown below <d-cite key="opik_answer_relevance"></d-cite>.

> You are an expert in NLP evaluation metrics, specifically trained to assess the relevance of answers. Your role is to evaluate the relevance of a given answer based on the user's input. Follow these steps to complete the evaluation. [...]

However, this approach is inherently self-referential: an LLM produces the answer and another LLM evaluates it. What assures us that the judge is trustworthy? The primary way to establish trust is by verifying its alignment with known cases. For example, one could ask a pool of experts to rate a small subset of responses; if the LLM-as-a-judge and the experts agree on those evaluations, it's reasonable to assume the automated judge will also perform reliably on the remaining examples.

Typically, adjusting the LLM-as-a-judge prompt to align with human feedback is cumbersome and largely manual: ground-truth collection requires human experts, and prompt calibration requires engineering expertise. This reliance on human judgment makes the process less reproducible and prone to inconsistency.

In this article, we investigate how to automate LLM-as-a-judge prompt calibration. Recent work introduced the use of a "cheatsheet", a set of reference instructions that guides the LLM's reasoning. Methods such as Dynamic Cheatsheet (DC) <d-cite key="suzgun2025dynamiccheatsheet"></d-cite> and Agentic Context Engineering (ACE) <d-cite key="zhang2025agenticcontextengineering"></d-cite> iteratively refine the cheatsheet based on observed examples, treating prompt optimization as an online learning problem.

However, when processing samples one at a time and updating the cheatsheet sequentially, important details are lost and performance degrades over time—a phenomenon known as <i>context collapse</i> <d-cite key="zhang2025agenticcontextengineering"></d-cite>. We observe this issue in our experimental evaluation, and address it with a batch-based approach inspired by classical machine learning training pipelines. 
Before diving into the methods, let's first discuss what we're evaluating and the data used.


## What are we evaluating?

Our reference use case is assessing <i>answer relevance</i> for responses generated by an internal AI assistant for network troubleshooting. Answer relevance is a metric that evaluates how pertinent an AI system's response is to the user's query <d-cite key="opik_answer_relevance"></d-cite>. To measure answer relevance, we provide the LLM-as-a-judge with an input query and the corresponding output response. The judge produces a score between 0 and 1, along with a justification for its rating. Example (anonymized):

<table style="border: none; border-collapse: collapse;">
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; padding: 8px;"><strong>user</strong></td>
    <td style="border: none; padding: 8px;">"Show me the latest issues"</td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; padding: 8px;"><strong>system</strong></td>
    <td style="border: none; padding: 8px;">"Fetching issues from the past 24 hours…We detected 5 alerts in the last 24 hours, 5 of which are major. The latest major alert was 'Packet Loss Internet - Alert - XYZ' applied on the target 'xyz'. The alert started at Dec 1, 2025 05:36:00 and lasted for 145 seconds impacting 1 test and 1 agent across the following 1 location: XYZ. The alert triggered because the packet loss, 15.0%, was outside of the threshold of 3%. We also detected 4 application outages. The latest app outage started at November 30, 2025 21:15:02 & lasted for 180 seconds impacting 5 servers & 2 locations. The outage occurred in these locations: London, England, UK; New York, New York, US."</td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; padding: 8px;"><strong>score</strong></td>
    <td style="border: none; padding: 8px;">1.0</td>
  </tr>
  <tr style="border: none;">
    <td style="border: none; vertical-align: top; padding: 8px;"><strong>reason</strong></td>
    <td style="border: none; padding: 8px;">"The answer addresses the question exhaustively and clearly."</td>
  </tr>
</table>

> <b><i>Note:</i></b> Unlike traditional metrics with clear mathematical definitions, this score represents a qualitative judgment guided by high-level criteria. 

## What about the data?

Let's establish some notation that we'll use throughout this article. Given a dataset of input/output pairs with associated scores and reasons, for a sample $i$ we denote:

| Symbol | Description |
|--------|-------------|
| $x_i$ | User input query |
| $y_i$ | System output |
| $s_i$ | LLM-as-a-judge generated score |
| $r_i$ | LLM-as-a-judge generated reason |
| $\tilde{s}_i$ | Ground-truth score (human-annotated) |
| $\tilde{r}_i$ | Ground-truth reason (human-annotated) |

Later in this article, we will denote the cheatsheet at iteration $t$ as $M_t$ (where $M$ stands for memory). In addition, when discussing batch-based approaches to prompt calibration, we will use $B$ to denote the set of sample indices belonging to a batch.

### The challenge of high-quality ground truth

To assess the performance of LLM-as-a-judge metrics, we need ground-truth scores and reasons. Obtaining reliable scores is far from trivial. For our purposes, we relied on human annotations: experts followed rating guidelines (e.g., "subtract 0.1 to 0.3 for unnecessary verbosity or repetition") to produce scores with quantitative justifications. In other words, for each input/output pair $(x_i, y_i)$, a human annotator provided a ground-truth score and reason, which we denote as $\tilde{s}_i$ and $\tilde{r}_i$.

However, upon careful review, we found that many scores didn't align with expected ratings. Subjectivity is inherent when evaluating LLM-generated content. To reduce this bias, we had multiple annotators perform ratings independently and retained only those records where they showed strong agreement.

Another challenge is sample diversity. LLMs can produce multiple valid responses for a given input, making random train/test splits potentially problematic. To better represent a realistic production scenario, we split the training and test sets based on a temporal cutoff: samples before a certain date were used for training, and those after for testing.

## Who are you to judge me?

Now that we have established our evaluation metric and collected ground truth annotations, the key question becomes: how do we calibrate the judge prompt to align its evaluations with human judgments? In this section, we explore different approaches to calibration.

### Our first attempt: manual calibration

Traditional calibration relies on manually crafted prompts. The workflow depends on human annotation samples and uses techniques such as few-shot prompting to address problematic queries. This process is typically iterative:

1. Collect ground-truth scores from human annotators
2. Compute alignment between human annotations and LLM-as-a-judge metrics
3. Identify discordant samples where human and model scores diverge
4. Refine the judge prompt to minimize score discrepancy

Engineering teams perform multiple rounds of manual calibration over time to ensure that human annotators and the LLM-as-a-judge metrics converge. Beyond being time-consuming and error-prone, this approach raises several technical concerns: (a) step 4 typically involves adding few-shot examples, which results in extremely long and over-engineered prompts; (b) ensuring that existing functionalities are preserved at each calibration round requires careful selection and analysis of reference samples. This process requires an expert engineer and can take several days. These limitations motivated us to explore more automated approaches.

### Take your cheatsheet out, the game begins

Dynamic Cheatsheet Cumulative (DC-Cu) <d-cite key="suzgun2025dynamiccheatsheet"></d-cite> was introduced in April 2025 by researchers at <i>Stanford University</i> and <i>Together AI</i>. The key idea is to treat prompt optimization as an online learning problem where an LLM processes samples sequentially, updating the cheatsheet after each one.

DC-Cu distinguishes between two roles: a <i>Generator</i> LLM and a memory <i>Curator</i> LLM. The Generator takes the input query $x_i$ and the current cheatsheet $M_i$ to produce an output $y_i$:

$$y_i = \text{Generator}(x_i, M_i)$$

The Curator then evaluates the output and refines the cheatsheet, keeping only the most useful and generalizable strategies:

$$M_{i+1} = \text{Curator}(M_i, x_i, y_i)$$

Since no ground truth is available, the Curator itself judges response quality. 
In practice, this self-verification approach is achievable for math questions and puzzles where the correctness of the solution can be verified using tools such as a calculator or code-execution environment.

> <b><i>Note:</i></b> Unlike math questions and puzzles, LLM-as-a-judge evaluations are not self-verifiable. There is no objective tool to verify whether a relevance score is correct. Therefore, we need ground-truth human annotations.

We adapted DC-Cu for supervised auto-calibration by providing the Memory Curator with ground truth scores $\tilde{s}_i$ and reasoning $\tilde{r}_i$ alongside the input/output pairs $(x_i, y_i)$ (see Fig. 1). We also modified the prompts to ensure these human annotations guide the cheatsheet generation process. 

<div class="row mt-4">
    <div class="col text-center">
        {% include figure.liquid path="assets/img/2026-04-27-auto-calibration/2025_12_01_DC.svg" class="img-fluid d-block mx-auto" max-width="90%" %}
    </div>
</div>
<div class="caption text-left">
    Fig. 1: Overview of the DC‑Cu method adapted to Judge auto‑calibration. Input–output pairs $(x_i, y_i)$ are retrieved from the database and evaluated by the Judge, which assigns a relevance score and reason $(s_i, r_i)$. The predicted score–reason pair is concatenated with the original input–output pair and the ground‑truth score–reason pair $(\tilde{s}_i, \tilde{r}_i)$, and passed to the Curator, which updates the cheatsheet $M_i$ to align the Judge's scoring behavior with the ground truth.
</div>

In our adapted DC-Cu framework, the process alternates between two phases. In the **Judgment Phase**, the Judge evaluates the input/output pair using the current cheatsheet $M_i$ to produce a score and reasoning:

$$s_i, r_i = \text{Judge}(x_i, y_i, M_i)$$

In the **Curation Phase**, the Curator updates the cheatsheet by comparing the Judge's predictions with the ground truth annotations ($\tilde{s}_i$, $\tilde{r}_i$):

$$M_{i+1} = \text{Curator}(M_i, x_i, y_i, \tilde{s}_i, \tilde{r}_i, s_i, r_i)$$

The Curator uses the discrepancy between predicted and ground truth values to refine the evaluation criteria stored in the cheatsheet.

### ACE Up Your Sleeve: Agentic Context Engineering

Agentic Context Engineering (ACE) <d-cite key="zhang2025agenticcontextengineering"></d-cite> builds upon DC-Cu by further refining its architectural structure. In ACE, the Memory Curator role is decomposed into two specialized components: a <i>Reflector</i>, which synthesizes insights from both correct and erroneous outputs, and a <i>Curator</i>, which integrates these insights into context updates. 
This separation prevents overburdening a single agent with the dual responsibilities of quality assessment and cheatsheet evolution. Additionally, ACE introduces a grow-and-refine mechanism that implements incremental updates to avoid full cheatsheet rewrites, pruning redundant entries through semantic analysis to ensure the cheatsheet remains both comprehensive and concise.

Inspired by ACE, we developed a customized implementation for the LLM-as-a-Judge supervised use case by extending DC-Cu with three key enhancements: (i) incorporating ground-truth scores and reasons during training, (ii) introducing a Reflector LLM, and (iii) enabling batching and epoch-based training. Further details are provided in the following section.

### Any BACI, please?

In this section, we introduce BACI (Batching Agentic Context Iteratively), our proposed strategy for automated judge calibration. The overall architecture is illustrated in Fig. 2.

<div class="row mt-4">
    {% include figure.liquid path="assets/img/2026-04-27-auto-calibration/2025_12_01_BACI.svg" class="img-fluid"  %}
</div>

<div class="caption text-left">
    Fig. 2: Overview of the BACI method. A batch $B$ of annotated samples is retrieved from the database, where each sample contains an input–output pair $(x_i, y_i)$ and the corresponding ground‑truth score–reason $(\tilde{s}_i, \tilde{r}_i)$.
    The Judge uses the cheatsheet rules to evaluate each input–output pair $(x_i, y_i)$ individually, producing a predicted score and reason $(s_i, r_i)$.
    Each prediction is then concatenated with its input-output pair and ground-truth, and the batch is forwarded to the Reflector. The Reflector compares ground-truth and predicted scores along with their reasons, identifying relationships among samples to extract insights $I_t$ about the Judge's errors. The Curator incorporates these insights into the cheatsheet $M_t$, aligning the Judge's scores with the ground truth. This process repeats for $k$ epochs to progressively optimize the cheatsheet.    
</div>

BACI incorporates **batching** as a core component to **iteratively** optimize the **agentic context**. During training, the Judge individually evaluates each sample $i$ within a batch $B$ using the current cheatsheet $M_t$ (where $t$ denotes the iteration number):

$$s_i, r_i = \text{Judge}(x_i, y_i, M_t) \quad \forall i \in B$$

The batch is then passed to the Reflector, which extracts insights $I_t$ by comparing the Judge's predictions with the human-provided ground truth across all samples in the batch:

$$I_{t} = \text{Reflector}(\{(x_i, y_i, \tilde{s}_i, \tilde{r}_i, s_i, r_i)\}_{i \in B})$$

These insights are fed to the Curator, which updates the cheatsheet accordingly:

$$M_{t+1} = \text{Curator}(M_t, I_t)$$

In subsequent iterations, the Judge uses the updated cheatsheet to generate new scores and reasons. This iterative process is repeated for all batches, with the cheatsheet being continuously refined at each step. The entire cycle is run for $k$ epochs, like a standard machine learning pipeline but employing gradient-free optimization.

At test time, we provide the Judge with the final version of the cheatsheet, refined during training. The Judge uses this cheatsheet to evaluate new, unseen samples by generating scores and reasons based on the accumulated knowledge. The final cheatsheet serves as a distilled summary of the most relevant evaluation patterns learned during training, guiding the Judge's evaluations in the test phase.

Compared to ACE, our Curator is responsible for both adding new evaluation instructions and de-duplicating entries. This design makes our method more lightweight than the original ACE approach, which maintains embeddings for each instruction in the cheatsheet.

In BACI, the Reflector extracts insights, and the Curator is instructed to perform updates by adding instructions (i) only if they are sufficiently different from existing ones, (ii) refining entries that lack important aspects, and (iii) discarding items that are similar to those already present. The combination of batching and the Reflector-Curator architecture helps us avoid context collapse and redundancy of instructions. In particular, the batch size plays a crucial role in this process --- as we will demonstrate in the following section.

> <b><i>Note:</i></b> we use Claude Sonnet 4.5, which has a nearly unlimited context window (200K base, can be extended up to 1 million tokens) for the LLM Judge, Reflector, and Curator. When using a model with a smaller context window, a trade-off in the batch size might be needed.

## What did we learn?

### Starting with the basics: when less is more

<div class="l-page">
  <iframe src="{{ 'assets/html/2026-04-27-auto-calibration/score_comparison_other_methods.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption text-left">
    Fig. 3: Score comparison on test set for the baseline methods. The score for each sample has been computed as the average across 10 predictions. The standard deviaton for each score can be seen by hovering over the point.
</div>

We compared the score distributions across our test set for different baselines (see Fig. 3). We first evaluated what happens when the Judge receives no special instructions, using only the basic definition of Answer Relevance, with no training, evolution, or use case specific guidelines. Surprisingly, on average, this “empty cheatsheet” approach performs on par with our manually calibrated prompt. This is remarkable given that an empty cheatsheet contains no domain-specific details.

Even more unexpectedly, the DC-Cu method performs worse than the previous baselines. As noted in <d-cite key="zhang2025agenticcontextengineering"></d-cite>, the main issue is context collapse: over time, the LLM Curator tends to generate shorter, less informative summaries, leading to a sharp decline in performance.

These observations lead to our first key insight:  

>It is better to provide no instructions in the cheatsheet than to include suboptimal instructions that may cause confusion or conflicting behavior.

This also explains why the carefully calibrated prompt did not outperform the baselines on the test set, despite meticulous fine-tuning, adjusted dataset scoring, and multiple few-shot examples.

### Learning in batches: the missing ingredient

<div class="l-page">
  <iframe src="{{ 'assets/html/2026-04-27-auto-calibration/score_comparison_BACI.html' | relative_url }}" frameborder='0' scrolling='no' height="600px" width="100%"></iframe>
</div>
<div class="caption text-left">
    Fig. 4: Score comparison on test set for BACI with different batch sizes and DC-Cu. The score for each sample has been computed as the average across 10 predictions. The standard deviaton for each score can be seen by hovering over the point.
</div>

We evaluated two BACI configurations: BACI-1 uses a batch size of one with a single training epoch, while BACI-32 uses a batch size of 32 and trains for five epochs. Figure 4 compares their performance against ground truth and DC-Cu. Here are the key observations:

* BACI-1 significantly outperforms DC-Cu. This improvement stems primarily from the separation of concerns between the Curator and Reflector components. As noted in the ACE work, this architectural separation --- where insight extraction (Reflector) and cheatsheet updating (Curator) are distinct processes --- helps mitigate context collapse across iterations.

* BACI-32 outperforms BACI-1. The larger batch size and multiple training epochs enable the system to observe the entire training dataset repeatedly, refining the cheatsheet iteratively. Crucially, processing samples in larger batches allows the Reflector to identify more generalizable patterns rather than overfitting individual examples.

### The final verdict

Bringing it all together: Table 1 summarizes all experimental results, showing average Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) <d-cite key="bishop2006pattern"></d-cite> across 10 test runs. BACI-32 (bold) clearly outperforms all competing methods on our dataset.

<div align="center" markdown="1">

| Method | MAE | RMSE |
|---|---|---|
| Empty Cheatsheet | 0.134 ± 0.002 | 0.201 ± 0.004 |
| Manually calibrated | 0.139 ± 0.010 | 0.219 ± 0.016 |
| DC-Cu | 0.272 ± 0.006 | 0.308 ± 0.007 |
| BACI-1 | 0.147 ± 0.003 | 0.212 ± 0.002 |
| BACI-32 | **0.111** ± 0.003 | **0.198** ± 0.004 |

</div>
<div class="caption text-left">
    Table 1: Experimental results comparing all methods. MAE and RMSE averaged across 10 test runs.
</div>

Statistical validation using the Wilcoxon signed-rank test <d-cite key="wilcoxon1945individual"></d-cite> confirms: (1) manual calibration provided no significant improvement over an empty cheatsheet ($p = 0.62$); (2) BACI-32 significantly outperforms both BACI-1 and the empty cheatsheet ($p < 0.01$).

## Key Takeaways

**Context collapse is real.** Our empirical analysis confirms that iterative, sample-by-sample approaches to prompt calibration suffer from a critical issue: context collapse. The solution lies in applying proper machine learning strategies—processing samples in batches and tuning hyperparameters such as batch size and number of epochs.  

**Manual calibration faces similar challenges.** Even manually crafted prompts can suffer from analogous issues, as they're typically adjusted iteratively on a static set of examples. Moreover, human bandwidth limits the number of samples that can be examined, making the process tedious, error-prone, and nearly impossible to scale.

**Sometimes, less is more.** Perhaps our most surprising finding: the strong performance of the empty cheatsheet baseline. This serves as a clear warning that wrong guidance can be worse than no guidance at all. When a simple solution works, there's no need to overcomplicate it.

**Data quality matters --- a lot.** The quality of training data is just as important as the calibration method itself. While this principle applies to all machine learning, it is especially critical for generative AI: we cannot expect an LLM to generate meaningful insights from inconsistent or low-quality input data. In our work, we observed a substantial subjectivity bias in human annotations. Data cleaning was crucial and required time and resources. Despite the impressive capabilities of modern LLMs, human judgment remains indispensable—at least for now, AI cannot fully replace expert reviewers.
