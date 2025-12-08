---
layout: distill
title: "Mimicking or Reasoning: Rethinking Multi-Modal In-Context Learning in Vision-Language Models"
description: Vision-language models (VLMs) are widely assumed to exhibit in-context learning (ICL), a property similar to that of their language-only counterparts. While recent work suggests VLMs can perform multimodal ICL (MM-ICL), studies show they often rely on shallow heuristics such as copying or majority voting, rather than true task understanding. We revisit this assumption by evaluating VLMs under distribution shifts, where support examples come from a dataset different from the query. Surprisingly, performance often degrades with more demonstrations, and models tend to copy answers rather than learn from them. To investigate further, we propose a new MM-ICL with reasoning pipeline that augments each demonstration with a generated rationale alongside the answer. We conduct extensive and comprehensive experiments on both perception- and reasoning-required datasets with open-source VLMs ranging from 3B to 72B and proprietary models such as Gemini 2.0 and 2.5. We conduct controlled studies varying shot count, retrieval method, rationale quality, and distribution. Our results show limited performance sensitivity across these factors, indicating that current VLMs fail to effectively utilize demonstration-level information and thus do not inherit the strong few-shot abilities of large language models (LLMs). We further conduct a mechanistic analysis showing that VLMs exhibit weak prefix matching and lack induction-head-like behavior, which potentially explains the failure of MM-ICL.
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

# must be the exact same name as your blogpost
bibliography: 2026-04-27-rethinking-mmicl.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Related Works
  - name: Rethinking the Success of Multimodal In-Context Learning
    subsections:
      - name: A Classification of MM-ICL tasks
      - name: A Closer Look at A Performance Gap
  - name: MM-ICL with Reasoning for Vision-Language Models
  - name: Experimental Results
    subsections:
      - name: "Format Always Matters: a Case study on the Format Inconsistency Issue"
      - name: Does MM-ICL with Reasoning Help? Zero-Shot vs. Few-Shot
      - name: Assessing the Role of Retriever Methods
      - name: ID versus OOD with Modern VLMs
  - name: "Why MM-ICL Falls Short: An Attention-Level Perspective"
  - name: Conclusion, Limitations and Future Direction

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
## Introduction
Vision-language models (VLMs), inspired by the success of large language models (LLMs), are widely believed to exhibit the ability of in-context learning (ICL), i.e., learning from a few examples provided in the prompt without any parameter updates. This capability has been well-documented in LLMs  <d-cite key="brown2020language, wei2022emergent, dong2022survey,khattab_demonstrate-search-predict_2023, zhou_least--most_2023,ge_innate_2025"/>, and recent work suggests that VLMs may inherit similar behavior through large-scale multimodal pretraining  <d-cite key="zong2024vl"/> and are capable of performing multimodal in-context learning (MM-ICL)  <d-cite key="qin2024factors,baldassini2024makes,awadalla2023openflamingo,bai_qwen-vl_2023"/>. However, several studies  <d-cite key="baldassini2024makes, qin2024factors,chen_can_2024"/> question whether current VLMs are truly learning from demonstrations. Instead, they find that VLMs often rely on shallow heuristics such as copying recent similar responses or defaulting to majority-vote patterns over the demonstrations-rather than acquiring a deeper understanding of the task.

To further probe this issue, we begin by testing under distribution shift, where support and query examples originate from different datasets. Counterintuitively, we observe that model performance often plateaus or even degrades as the number of shots increases, despite being given more demonstrations. This contrasts with the in-distribution case, where performance reliably improves with more demonstrations. We also find failure cases where the model simply copies answers from the demonstrations, rather than learning from them. These observations raise a central question: *do VLMs truly learn from in-context demonstrations, or are they just matching superficial patterns?*

To explore this question, we propose to evaluate whether VLMs can move beyond surface-level pattern matching and truly learn from in-context demonstrations in a new setting. Rather than providing only final answers, we enrich each demonstration with a detailed **reasoning process**  <d-cite key="jiang_mme-cot_2025,wang_multimodal_2025,lu_mathvista_2024,hao_can_2025,gao_interleaved-modal_2024,yang_formal_2024,stefanik_can_2023"/>, i.e., explicit step-by-step rationales that make the task-solving strategy clear. By increasing the informational content of each example, we aim to give models a stronger learning signal and a better chance to internalize the methodology behind the task, rather than relying on shallow cues. To achieve this, we leverage the capabilities of **VLM Reasoners**   <d-cite key="shen2025vlm, xu2024llava, wang2025vl"/>, which inherently generate rationales and answers simultaneously, to assess whether access to intermediate reasoning steps helps models generalize more effectively from demonstrations. Our contributions are as follows:

**(1)** To the best of our knowledge, this paper is the first to study the MM-ICL of VLMs from the lens of reasoning. Using information-enhanced demonstrations with reasoning components, we benchmark the MM-ICL capability of modern VLMs and reach conclusions with more solid evidence.

**(2)** To fairly evaluate MM-ICL for VLM reasoners, we introduce a new **MM-ICL with Reasoning** pipeline that resolves a key format mismatch in prior work: instead of supplying only answers in demonstrations while expecting rationale-plus-answer outputs, we provide demonstrations with both a Pseudo Reasoning (a generated rationale) and an answer. This consistent support-query format improves performance across models and datasets over inconsistent ones.

**(3)** We conduct extensive controlled studies by varying shot count, retrieval method, rationale quality, and distribution. Our analysis reveals that MM-ICL are largely insensitive to these factors, showing limited performance variation across different configurations. We reveal a counterintuitive failure mode showing that current VLMs do not effectively leverage demonstration-level information, challenging the belief that they inherit few-shot learning abilities from LLMs.

**(4)** We provide an attention-level perspective, showing that VLMs demonstrate weak prefix matching and no clear induction-head–like behavior, potentially explaining their limited MM-ICL performance.

## Related Works
**Multimodal In-Context Learning.** Large VLMs have the emerging ability to answer an unseen question or perform a new task without additional training, a capability known as zero-shot learning. Moreover, researchers have found that these models can often achieve better performance when multiple demonstrations of solutions to similar tasks are presented to the model before querying the question  <d-cite key="brown2020language"/>. LLMs have shown strong ICL abilities—learning from demonstrations without parameter updates  <d-cite key="brown2020language, wei2022emergent, dong2022survey"/>. VLMs, built on LLMs and pretrained on large-scale multimodal data, are believed to inherit similar capabilities. Recent benchmarks  <d-cite key="zong2024vl"/> and follow-up studies  <d-cite key="qin2024factors, xu2024introspection"/> have evaluated MM-ICL across tasks and analyzed factors such as retrieval, prompt design, and modality contributions. However, these efforts assume that VLMs are capable of MM-ICL, without first establishing whether models actually understand and learn from demonstrations. This motivates a deeper investigation into what VLMs learn in the MM-ICL setting.

**Vision-Language Reasoning Models** To enhance reasoning in VLMs, recent work has focused on post-training techniques and curated datasets. Reinforcement learning (RL)-based approaches, such as Group Relative Policy Optimization (GRPO), have been applied to improve performance across tasks like referring expression comprehension and open-vocabulary detection  <d-cite key="shen2025vlm, wang2025vl"/>. In parallel, non-RL strategies—such as preference optimization  <d-cite key="wang2024enhancing"/>—have shown success in reducing hallucination and improving multi-step reasoning. Additionally, structured reasoning datasets like LLaVA-CoT  <d-cite key="xu2024llava"/> offer a complementary path by enabling fine-tuning with explicit reasoning supervision. Together, these advances reflect growing interest in building VLMs that can reason more reliably and systematically.

## Rethinking the Success of Multimodal In-Context Learning
### A Classification of MM-ICL tasks
The tasks used for MM-ICL capability benchmarking can be roughly categorized into two categories, depending on whether the problem is well-defined without demonstrations.

**Case I: well-defined tasks without demos:** Visual Question Answering (based on common sense or factual knowledge), Image Captioning, etc. Tasks with solutions uniquely determined by the query. 

**Case II: ill-defined tasks without demos:** Operator Induction, Open-Ended Object Recognition (with synthetic category), etc. Tasks are well-defined only when demonstrations of successfully solved cases are presented <d-cite key="zong2024vl"/>.

<div id="fig-case12">
{% include figure.liquid 
    path="assets/img/2026-04-27-rethinking-mmicl/case12.pdf" 
    class="img-fluid" 
%}
</div>
<div class="caption">
    Figure 1: Examples from Case I/II tasks. Case II tasks are ill-defined if no demonstrations are given.
</div>

We present examples of Case I/II tasks in [Figure 1](#fig-case12). While both cases have been studied in the MM-ICL literature, it is less clear how to quantify whether the model succeeds in learning from the demonstration examples for tasks of Case I. Many essential tasks of great practical utility related to perception or reasoning fall in Case I. These tasks have instructions that are clear to understand and follow even when no demonstration is provided, and thus are tractable for VLMs to solve in a zero-shot setting to some degree. Therefore, it's natural to raise the question: can ICL *truly* enhance a model's capability to solve these tasks as a type of inference-time scaling technique? To answer this question, we investigate how to benchmark and determine *fairly* VLMs' ability to learn from demonstrations in a multimodal scenario, as a primary focus of this work.

### A Closer Look at A Performance Gap
To understand whether VLMs can learn from demonstrations, we begin by exposing them to a more realistic setting, where the demonstration data originates from a different dataset with a distinct distribution from that of the queries  <d-cite key="mosbach2023fewshotfinetuningvsincontext"/>. Note that this setting closely reflects the real-world user case of ICL, as it's often unrealistic to provide highly relevant demonstrations with ground-truth answers to aid the learning process for entirely new, unseen questions. We denote this setting as **Out-of-Distribution** (OOD), in contrast to the **In-Distribution** (ID) setting where we select demonstrations from the training split of the query dataset. We still enforce that the OOD support set shares the same task type as query data to ensure the evaluation is reasonable. 

The primary motivation behind this experiment design is to evaluate whether the model *truly* understands how to perform the target tasks better by mastering the methodology behind them, or it only gains a superficial understanding through memorizing information presented in the demonstrations. To make an analogy, this setting provides the student (VLM) with a test paper (query data) that contains problems not merely variants of the questions (OOD support set) it has seen, but can still be solved using similar methods (same task type). 

We consider the task of Visual Question Answering (VQA), a classic problem that falls under Case I discussed above. We use TextVQA  <d-cite key="singh2019towards"/> and OK-VQA  <d-cite key="reichman_outside_2023"/> and evaluate the performance of OpenFlamingo  <d-cite key="awadalla2023openflamingo"/> and IDEFICS2  <d-cite key="laurenccon2024matters"/>. These datasets and models are popular choices for studying MM-ICL in the literature  <d-cite key="baldassini2024makes, qin2024factors"/>. Each model will be asked to answer the question in a short response with $1$-$2$ words using the instruction prompt ``"Answer the question using a single word or phrase."`` whenever necessary. The accuracy of the response is computed through **exactly matching** it to the provided answer candidates. The results are presented in [Figure 2](#fig-compare-retrieval).

<div id="fig-compare-retrieval">
{% include figure.liquid 
    path="assets/img/2026-04-27-rethinking-mmicl/Picture2.pdf" 
    class="img-fluid"
%}
</div>
<div class="caption">
    Figure 2: Left: Performance difference between ID and OOD using a random retriever.  
    Middle: Performance of different retrieval methods on OK-VQA.  
    ID uses OK-VQA as the support set, while OOD uses TextVQA as the support set.  
    We include the unimodal retriever to highlight that the multimodal retriever achieves the best performance in the ID setting, consistent with <d-cite key="qin2024factors" />.  
    Right: Incorrect answer formatting directly increases the error rate.
</div>

We notice an intriguing performance difference between ID and OOD settings ([Figure 2](#fig-compare-retrieval) Left) with the increase number of shots. We observe that the accuracy of OpenFlamingo **monotonically increases** as the number of shots grows, while such trends are rarely observed in the OOD setting. In the OOD settings, there is a minimal or no increase in accuracy with more demonstrations. This is rather counterintuitive since the model was offered with strictly more correct information (though some of it was less relevant due to OOD); thus, the model should not perform worse than in the zero-shot setting. Additionally, as shown in [Figure 2](#fig-compare-retrieval) (Middle), retrieval-based methods consistently outperform random selection in the ID setting, while the opposite holds in the OOD setting. These observations suggest that the hidden factor that drives the performance gap between ID and OOD cases is not directly relevant to the model's capability to solve such tasks.

One possible factor behind this phenomenon is the response format. When presented with ID demonstrations, the query and support examples share a similar format, making it easier for the model to pick up and follow the expected answer style. However, when presented with numerous OOD examples in distinct formats, the model struggles to generalize the formatting instructions, leading to degraded performance. An example of such an error is presented in [Figure 2](#fig-compare-retrieval) (Right), where a correct answer from OpenFlamingo is deemed as wrong due to a mismatch of answer formats (two words instead of one). A similar conclusion is also mentioned in  <d-cite key="zong2024vl"/>, where an LLM is used to judge whether a response semantically aligns with the ground-truth answer, instead of relying on an exact-matching function. Compared to exact-matching, LLM judges are less sensitive to answer formats and therefore are more robust for accuracy evaluation. Using LLM judges would drastically reduce the performance gain margin over zero-shot results, supporting our observation that the VLMs like OpenFlamingo **might not learn anything more than formatting** from the demonstrations.

<div id="fig-demos">
{% include figure.liquid 
    path="assets/img/2026-04-27-rethinking-mmicl/demos.pdf" 
    class="img-fluid"
%}
</div>
<div class="caption">
    Figure 3: Success and failure of MM-ICL with IDEFICS2.
</div>

We also present examples of success and failure of MM-ICL in [Figure 3](#fig-demos). In the presented cases, when the query is spuriously correlated with image-question pairs in demonstrations, the model latches onto the superficial similarity and directly copies the support answer. In contrast, when an irrelevant example is included, it disrupts this superficial pattern matching, preventing the model from copying and exposing its inability to truly "learn from" relevant demonstrations to answer the query.

## MM-ICL with Reasoning for Vision-Language Models
While the results presented in the previous section are surprising, these experiments would not be enough to claim anything about whether large VLMs have the *true* ICL capability for Case I tasks for the following reasons. First, despite being widely selected as benchmarks for MM-ICL tasks due to their decent performance, OpenFlamingo and IDEFICIS2 no longer represent the state-of-the-art VLMs, thus it would be biased to draw any conclusions just based on their evaluation performances. Furthermore, except for the question-answer pair, very limited information is provided during the ICL stage, which potentially prevents VLMs from extracting deeper, more meaningful information beyond the answer format and further restricts the performance gain from an increased number of shots.  

To address these issues, we propose to benchmark the MM-ICL ability of **modern VLMs** in a new setting, where we provide the model with information-enriched demonstrations to maximize the utility of each example. We achieve such information augmentation by introducing **reasoning process** into the demo, and each presented example would contain a detailed step-by-step thinking process instead of a single answer. By doing so, we lower the bar for models to learn from demonstrations by adding more explicit information to each support data. We also consider a variety of datasets related to both general perception and specialized reasoning to make the study more comprehensive and trustworthy. 

<div id="tab-reasoning-models" style="display:flex; justify-content:center; width:100%;">
  <table style="border-collapse:collapse; text-align:center; width:70%;">
    <caption style="caption-side:top;">
      Table 1: VLMs and their associated reasoner version used in the evaluation.
    </caption>

    <thead>
      <tr>
        <th style="text-align:left;">Base Model</th>
        <th>Size</th>
        <th>Reasoning Model</th>
      </tr>
    </thead>

    <tbody>

      <!-- Qwen2.5-VL -->
      <tr>
        <td rowspan="3" style="text-align:left; font-weight:bold;">Qwen2.5-VL</td>
        <td>3B</td>
        <td>VLM-R1</td>
      </tr>
      <tr>
        <td>7B</td>
        <td>VL-Rethinker 7B</td>
      </tr>
      <tr>
        <td>72B</td>
        <td>VL-Rethinker 72B</td>
      </tr>

      <!-- InternVL 2.5 -->
      <tr>
        <td rowspan="2" style="text-align:left; font-weight:bold;">InternVL 2.5</td>
        <td>4B</td>
        <td>InternVL 2.5-4B-MPO</td>
      </tr>
      <tr>
        <td>8B</td>
        <td>InternVL 2.5-8B-MPO</td>
      </tr>

      <!-- Llama -->
      <tr>
        <td style="text-align:left; font-weight:bold;">Llama-3.2V</td>
        <td>11B</td>
        <td>LLaVA-CoT</td>
      </tr>

    </tbody>
  </table>
</div>

We focus on evaluating VLMs which has an open-source reasoning variant, such as Qwen2.5-VL  <d-cite key="bai2025qwen2"/> and VLM-R1  <d-cite key="shen2025vlm"/>, VL-Rethinker  <d-cite key="wang2025vl"/>; InternVL2.5  <d-cite key="chen2024expanding"/> and InternVL2.5-MPO  <d-cite key="wang2024enhancing"/>; Llama-3.2V  <d-cite key="llama32"/> and LLaVA-CoT  <d-cite key="xu2024llava"/>. We list the parameter size of each version in [Table 1](#tab-reasoning-models). Through enforcing such a pairing relationship, we can better compare the ICL capabilities across models from a fair perspective. To evaluate the effects of reasoning components on MM-ICL performance, we consider four different protocols. The prompts for each protocol are depicted in [Figure 4](#fig-reasoning-demo-variants).

<div id="fig-reasoning-demo-variants">
{% include figure.liquid
    path="assets/img/2026-04-27-rethinking-mmicl/Picture3.pdf"
    class="img-fluid"
%}
</div>
<div class="caption">
    Figure 4: Prompt format for each MM-ICL protocol. GT stands for ground truth.
</div>

**(1) Base model without reasoning demos.** This is the standard setting of MM-ICL, where each demo is a concatenation of image, question, and ground truth answer.

**(2) Reasoning model without reasoning demos.** This setting is similar to Protocol (1), and serves as the standard MM-ICL baseline for reasoning models. As before, the support sample is a concatenation of image, question, and ground truth answer. The key difference is that the reasoning model is expected to generate both the final answer and a rationale.

**(3) Base model with reasoning demos.** This setting evaluates whether providing additional information (i.e., ground truth or generated rationales) in the support set helps the model learn from them and predict the correct answer for the query through ICL. If the dataset includes ground truth rationales, they are concatenated with the image and question in the support samples. 

**(4) Reasoning model with reasoning demos.**  This setting is similar to Protocol (3), except that the reasoning in the demo needs to be formatted in the same way as the reasoner model is trained. This is to address the issue of **format inconsistency**, which is discussed below.

<div id="fig-reasoning-icl">
{% include figure.liquid
    path="assets/img/2026-04-27-rethinking-mmicl/workflow.pdf"
    class="img-fluid"
%}
</div>
<div class="caption">
    Figure 5: Visualization of the full pipeline for ICL with VLM Reasoner.
</div>

We argue that the standard MM-ICL Protocol (2) for reasoning models introduces a **format inconsistency**. While reasoning models are trained with interleaved inputs—image, question, rationale, and answer—the demonstrations in MM-ICL typically include only the answer. In contrast, the model is expected to generate both the explanation and the answer for the query. This mismatch can lead to suboptimal performance, e.g., the model may focus solely on developing the answer and ignore the rationale, resulting in degraded output quality (similar findings in  <d-cite key="zheng2025cursecotlimitationschainofthought"/>) (see following sections).

To address this, we newly introduce **Protocol (4)** for the best practice of MM-ICL with VLM reasoners, which contains a two-stage process. First, we prompt the model with each support sample to generate both a rationale (**Pseudo Reasoning**) and an answer, ensuring consistency with the expected output format. We then concatenate these generated reasoning-augmented demonstrations with the original input of each support to form a coherent and format-aligned context for the query.

However, since the generated rationales may vary in quality, we introduce two strategies to improve reliability: (1) Ground truth rationale reformulation: If ground truth rationales are available, we use them as input to the model to reformat the rationale into the desired structure (**Gold Reasoning**). (2) Correctness-based filtering: We use the correctness of the generated answer as a heuristic to filter out support samples with misleading rationales. The whole pipeline is illustrated in [Figure 5](#fig-reasoning-icl).

## Experimental Results
We consider datasets that focus on perception and reasoning in the following experiments.

**Perception Datasets.** TextVQA <d-cite key="singh2019towards" /> and OK-VQA <d-cite key="reichman_outside_2023" /> focus on reading text in images and answering commonsense questions, respectively. They use their own answer matching metrics (e.g., string normalization and consensus-based accuracy) for evaluation.

**Reasoning Datasets.** ScienceQA <d-cite key="lu2022learnexplainmultimodalreasoning" />, A-OKVQA <d-cite key="schwenk2022aokvqabenchmarkvisualquestion" />, and M³CoT <d-cite key="chen2024m3cotnovelbenchmarkmultidomain" /> target multi-step reasoning. ScienceQA features science questions accompanied by images and provides expert-written explanations as rationales. A-OKVQA offers curated natural language rationales aligned with commonsense reasoning. M³CoT uses chain-of-thought rationales generated via prompting to guide multi-hop reasoning. We follow the VLMEvalKit <d-cite key="duan2024vlmevalkit" /> setup, which uses GPT-4o mini as a judge to assess answer quality.

For each dataset, we use the training split to construct the support set. The query set is taken from the test split if ground-truth answers are available; otherwise, we use the validation split.
Unless otherwise specified, we use Protocol (1) for VLM base models and Protocol (4) for VLM reasoners.

### Format Always Matters: a Case study on the Format Inconsistency Issue
To assess whether modern VLMs still suffer from the mismatch in format of the MM-ICL demonstrations, we experiment with a reasoning-aware support-query format. Specifically, we compare two setups: 
**(1) Inconsistent** (Protocol (2)): Each demonstration contains only the final answer.
**(2) Consistent** (Protocol (4)): Each demonstration contains the full reasoning process, including rationale and answer, mirroring the model’s expected generation format.

To ensure a fair comparison, we use **Pseudo Reasoning**, where the rationale component in each demonstration is generated by the model itself based on the original support set inputs. This avoids the need for additional supervision and keeps all settings grounded in the same available information.

<div style="overflow-x:auto; max-width:100%; justify-content:center;" id="tab-compare-format">
  <table>
    <caption style="caption-side:top;">Table 2: Comparison of inconsistent and consistent support-query format with VLM reasoners.</caption>
    <thead>
      <tr>
        <th rowspan="2" style="text-align:left;">Ablation</th>
        <th colspan="4" style="text-align:center;">A-OKVQA</th>
        <th colspan="4" style="text-align:center;">ScienceQA</th>
        <th colspan="4" style="text-align:center;">M&#179;CoT</th>
      </tr>
      <tr>
        <th style="text-align:center;">1</th>
        <th style="text-align:center;">2</th>
        <th style="text-align:center;">4</th>
        <th style="text-align:center;">8</th>
        <th style="text-align:center;">1</th>
        <th style="text-align:center;">2</th>
        <th style="text-align:center;">4</th>
        <th style="text-align:center;">8</th>
        <th style="text-align:center;">1</th>
        <th style="text-align:center;">2</th>
        <th style="text-align:center;">4</th>
        <th style="text-align:center;">8</th>
      </tr>
    </thead>
    <tbody>

      <!-- VLM-R1 -->
      <tr>
        <th colspan="13" style="text-align:center; background-color:#f3f4f6;">VLM-R1</th>
      </tr>

      <tr>
        <td style="text-align:left;">inconsistent</td>
        <td style="text-align:center;">81.31</td><td style="text-align:center;">81.31</td>
        <td style="text-align:center;">80.44</td><td style="text-align:center;">79.39</td>
        <td style="text-align:center;">81.71</td><td style="text-align:center;">81.95</td>
        <td style="text-align:center;">81.31</td><td style="text-align:center;">80.61</td>
        <td style="text-align:center;">51.64</td><td style="text-align:center;">53.36</td>
        <td style="text-align:center;">51.51</td><td style="text-align:center;">50.60</td>
      </tr>

      <tr>
        <td style="text-align:left;">consistent</td>
        <td style="text-align:center;">82.45</td><td style="text-align:center;">81.83</td>
        <td style="text-align:center;">81.48</td><td style="text-align:center;">81.14</td>
        <td style="text-align:center;">82.30</td><td style="text-align:center;">83.39</td>
        <td style="text-align:center;">82.45</td><td style="text-align:center;">82.94</td>
        <td style="text-align:center;">53.19</td><td style="text-align:center;">53.80</td>
        <td style="text-align:center;">54.36</td><td style="text-align:center;">52.89</td>
      </tr>

      <tr>
        <td style="text-align:left;">&Delta;</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.14</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.52</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.04</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.75</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.59</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.44</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.14</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+2.33</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.55</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.44</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+2.85</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+2.29</span></td>
      </tr>

      <!-- VL-Rethinker-7B -->
      <tr>
        <th colspan="13" style="text-align:center; background-color:#f3f4f6;">VL-Rethinker-7B</th>
      </tr>

      <tr>
        <td style="text-align:left;">inconsistent</td>
        <td style="text-align:center;">85.76</td><td style="text-align:center;">85.24</td>
        <td style="text-align:center;">84.28</td><td style="text-align:center;">83.76</td>
        <td style="text-align:center;">89.04</td><td style="text-align:center;">88.94</td>
        <td style="text-align:center;">88.89</td><td style="text-align:center;">89.14</td>
        <td style="text-align:center;">66.22</td><td style="text-align:center;">67.60</td>
        <td style="text-align:center;">66.39</td><td style="text-align:center;">66.82</td>
      </tr>

      <tr>
        <td style="text-align:left;">consistent</td>
        <td style="text-align:center;">85.50</td><td style="text-align:center;">84.98</td>
        <td style="text-align:center;">84.98</td><td style="text-align:center;">85.68</td>
        <td style="text-align:center;">90.23</td><td style="text-align:center;">90.23</td>
        <td style="text-align:center;">90.18</td><td style="text-align:center;">90.23</td>
        <td style="text-align:center;">68.08</td><td style="text-align:center;">69.15</td>
        <td style="text-align:center;">68.59</td><td style="text-align:center;">68.55</td>
      </tr>

      <tr>
        <td style="text-align:left;">&Delta;</td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.26</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.26</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.70</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.92</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.19</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.29</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.29</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.09</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.86</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.55</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+2.20</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.73</span></td>
      </tr>

      <!-- LLaVA-CoT -->
      <tr>
        <th colspan="13" style="text-align:center; background-color:#f3f4f6;">LLaVA-CoT</th>
      </tr>

      <tr>
        <td style="text-align:left;">inconsistent</td>
        <td style="text-align:center;">85.85</td><td style="text-align:center;">85.33</td>
        <td style="text-align:center;">84.28</td><td style="text-align:center;">83.14</td>
        <td style="text-align:center;">91.52</td><td style="text-align:center;">90.98</td>
        <td style="text-align:center;">87.65</td><td style="text-align:center;">84.78</td>
        <td style="text-align:center;">55.26</td><td style="text-align:center;">51.42</td>
        <td style="text-align:center;">44.26</td><td style="text-align:center;">42.28</td>
      </tr>

      <tr>
        <td style="text-align:left;">consistent</td>
        <td style="text-align:center;">86.20</td><td style="text-align:center;">85.59</td>
        <td style="text-align:center;">84.54</td><td style="text-align:center;">83.23</td>
        <td style="text-align:center;">92.81</td><td style="text-align:center;">91.97</td>
        <td style="text-align:center;">91.57</td><td style="text-align:center;">90.48</td>
        <td style="text-align:center;">54.62</td><td style="text-align:center;">53.62</td>
        <td style="text-align:center;">52.29</td><td style="text-align:center;">50.99</td>
      </tr>

      <tr>
        <td style="text-align:left;">&Delta;</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.35</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.26</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.26</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.09</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.29</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.99</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+3.92</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+5.70</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.64</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+2.20</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+8.03</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+8.71</span></td>
      </tr>

    </tbody>
  </table>
</div>

[Table 2](#tab-compare-format) shows results across three reasoning models (VLM-R1, VL-Rethinker-7B, and LLaVA-CoT) on three benchmarks (A-OKVQA, ScienceQA, M³CoT) under different shots. Consistent formatting where demonstrations include both rationale and answer consistently outperforms inconsistent formatting across models and datasets, especially in high-shot settings (e.g., +8.71 on M³CoT with 8 shots using LLaVA-CoT). This suggests that aligning the demonstration format with the model’s expected output is crucial for effective MM-ICL with reasoning, even for capable model VLMs. **We stick to this consistent formatting pipeline for reasoning models for other experiments.**

### Does MM-ICL with Reasoning Help? Zero-Shot vs. Few-Shot
With the best practice for MM-ICL with VLM reasoners established, we now proceed to determine whether VLMs can successfully perform MM-ICL with information-enriched demos. We present the results in [Table 3](#tab-per-dataset-bolded) and [Table 4](#tab-reason-per-dataset-bolded). As evident from the tables, in the majority of cases, MM-ICL with a few demonstrations does not exceed the performance when no demonstrations are presented. An improvement is observed in some rare cases, while the performance gain is often minimal. These results suggest that current VLMs/VLM reasoners indeed **can barely learn from demonstration data** even if the presented example data is ID. This demonstrates an essential weakness of current VLMs compared to their language-only counterparts, where ICL is often considered to be widely capable and beneficial <d-cite key="baldassini2024makes, qin2024factors" />.  

<div style="display:flex; justify-content:center;" id="tab-per-dataset-bolded">
  <table>
    <caption style="caption-side:top;">Table 3: Perception datasets accuracy across models and shots. Best values across shots in <strong>bold</strong>.</caption>
    <thead>
      <tr>
        <th rowspan="2" style="text-align:left;">Models</th>
        <th colspan="2" style="text-align:center;">TextVQA</th>
        <th colspan="2" style="text-align:center;">OK-VQA</th>
      </tr>
      <tr>
        <th style="text-align:center;">0-shot</th>
        <th style="text-align:center;">best few-shot</th>
        <th style="text-align:center;">0-shot</th>
        <th style="text-align:center;">best few-shot</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Qwen2.5-VL-3B-Instruct</td>
        <td style="text-align:center;"><strong>79.13</strong></td><td style="text-align:center;">78.70</td>
        <td style="text-align:center;">54.09</td><td style="text-align:center;"><strong>56.63</strong></td>
      </tr>
      <tr>
        <td>VLM-R1</td>
        <td style="text-align:center;"><strong>74.87</strong></td><td style="text-align:center;">74.54</td>
        <td style="text-align:center;">40.00</td><td style="text-align:center;"><strong>42.97</strong></td>
      </tr>
      <tr>
        <td>Qwen2.5-VL-7B-Instruct</td>
        <td style="text-align:center;"><strong>85.39</strong></td><td style="text-align:center;">84.79</td>
        <td style="text-align:center;">58.74</td><td style="text-align:center;"><strong>62.68</strong></td>
      </tr>
      <tr>
        <td>VL-Rethinker-7B</td>
        <td style="text-align:center;"><strong>76.46</strong></td><td style="text-align:center;">73.01</td>
        <td style="text-align:center;"><strong>32.43</strong></td><td style="text-align:center;">30.14</td>
      </tr>
      <tr>
        <td>Llama-3.2-11B-Vision-Instruct</td>
        <td style="text-align:center;">53.87</td><td style="text-align:center;"><strong>74.63</strong></td>
        <td style="text-align:center;">20.05</td><td style="text-align:center;"><strong>44.03</strong></td>
      </tr>
      <tr>
        <td>LLaVA-CoT</td>
        <td style="text-align:center;">72.85</td><td style="text-align:center;"><strong>74.39</strong></td>
        <td style="text-align:center;"><strong>48.91</strong></td><td style="text-align:center;">47.46</td>
      </tr>
      <tr>
        <td>InternVL2.5-4B</td>
        <td style="text-align:center;">78.68</td><td style="text-align:center;"><strong>78.88</strong></td>
        <td style="text-align:center;">49.88</td><td style="text-align:center;"><strong>54.79</strong></td>
      </tr>
      <tr>
        <td>InternVL2.5-4B-MPO</td>
        <td style="text-align:center;">72.85</td><td style="text-align:center;"><strong>72.90</strong></td>
        <td style="text-align:center;">42.12</td><td style="text-align:center;"><strong>42.55</strong></td>
      </tr>
      <tr>
        <td>InternVL2.5-8B</td>
        <td style="text-align:center;"><strong>79.03</strong></td><td style="text-align:center;">78.73</td>
        <td style="text-align:center;">57.20</td><td style="text-align:center;"><strong>59.62</strong></td>
      </tr>
      <tr>
        <td>InternVL2.5-8B-MPO</td>
        <td style="text-align:center;">73.36</td><td style="text-align:center;"><strong>73.67</strong></td>
        <td style="text-align:center;">44.49</td><td style="text-align:center;"><strong>49.01</strong></td>
      </tr>
      <tr>
        <td>Gemini 2.0 Flash-non-thinking</td>
        <td style="text-align:center;">77.13</td><td style="text-align:center;"><strong>78.53</strong></td>
        <td style="text-align:center;">40.47</td><td style="text-align:center;"><strong>50.49</strong></td>
      </tr>
      <tr>
        <td>Gemini 2.0 Flash-thinking</td>
        <td style="text-align:center;"><strong>77.87</strong></td><td style="text-align:center;">76.64</td>
        <td style="text-align:center;">41.17</td><td style="text-align:center;"><strong>43.63</strong></td>
      </tr>
    </tbody>
  </table>
</div>

<div style="display:flex; justify-content:center" id="tab-reason-per-dataset-bolded">
  <table>
    <caption style="caption-side:top;">Table 4: Reasoning datasets accuracy across models and shots. Higher accuracy between 0-shot and best few-shot is <strong>bolded</strong>.</caption>
    <thead>
      <tr>
        <th rowspan="2" style="text-align:left;">Models</th>
        <th colspan="2" style="text-align:center;">A-OKVQA</th>
        <th colspan="2" style="text-align:center;">ScienceQA</th>
        <th colspan="2" style="text-align:center;">M&#179;CoT</th>
      </tr>
      <tr>
        <th style="text-align:center;">0-shot</th>
        <th style="text-align:center;">best few-shot</th>
        <th style="text-align:center;">0-shot</th>
        <th style="text-align:center;">best few-shot</th>
        <th style="text-align:center;">0-shot</th>
        <th style="text-align:center;">best few-shot</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Qwen2.5-VL-3B-Instruct</td>
        <td style="text-align:center;"><strong>85.41</strong></td><td style="text-align:center;">82.01</td>
        <td style="text-align:center;"><strong>81.61</strong></td><td style="text-align:center;">81.11</td>
        <td style="text-align:center;"><strong>51.77</strong></td><td style="text-align:center;">51.34</td>
      </tr>

      <tr>
        <td>VLM-R1</td>
        <td style="text-align:center;"><strong>85.07</strong></td><td style="text-align:center;">82.45</td>
        <td style="text-align:center;">82.30</td><td style="text-align:center;"><strong>83.39</strong></td>
        <td style="text-align:center;">53.11</td><td style="text-align:center;"><strong>54.36</strong></td>
      </tr>

      <tr>
        <td>Qwen2.5-VL-7B-Instruct</td>
        <td style="text-align:center;">88.56</td><td style="text-align:center;"><strong>88.65</strong></td>
        <td style="text-align:center;"><strong>88.99</strong></td><td style="text-align:center;">87.65</td>
        <td style="text-align:center;"><strong>63.03</strong></td><td style="text-align:center;">60.53</td>
      </tr>

      <tr>
        <td>VL-Rethinker-7B</td>
        <td style="text-align:center;"><strong>85.68</strong></td><td style="text-align:center;"><strong>85.68</strong></td>
        <td style="text-align:center;">89.64</td><td style="text-align:center;"><strong>90.23</strong></td>
        <td style="text-align:center;">67.90</td><td style="text-align:center;"><strong>69.15</strong></td>
      </tr>

      <tr>
        <td>Qwen2.5-VL-72B-Instruct</td>
        <td style="text-align:center;"><strong>91.44</strong></td><td style="text-align:center;">91.18</td>
        <td style="text-align:center;">91.18</td><td style="text-align:center;"><strong>91.57</strong></td>
        <td style="text-align:center;"><strong>70.23</strong></td><td style="text-align:center;">70.02</td>
      </tr>

      <tr>
        <td>VL-Rethinker-72B</td>
        <td style="text-align:center;">88.82</td><td style="text-align:center;"><strong>89.34</strong></td>
        <td style="text-align:center;"><strong>94.40</strong></td><td style="text-align:center;">93.75</td>
        <td style="text-align:center;">74.85</td><td style="text-align:center;"><strong>76.40</strong></td>
      </tr>

      <tr>
        <td>Llama-3.2-11B-Vision-Instruct</td>
        <td style="text-align:center;"><strong>84.02</strong></td><td style="text-align:center;"><strong>84.02</strong></td>
        <td style="text-align:center;">83.99</td><td style="text-align:center;"><strong>84.43</strong></td>
        <td style="text-align:center;">42.45</td><td style="text-align:center;"><strong>43.74</strong></td>
      </tr>

      <tr>
        <td>LLaVA-CoT</td>
        <td style="text-align:center;"><strong>87.42</strong></td><td style="text-align:center;">86.20</td>
        <td style="text-align:center;"><strong>94.55</strong></td><td style="text-align:center;">92.81</td>
        <td style="text-align:center;"><strong>56.26</strong></td><td style="text-align:center;">54.62</td>
      </tr>

      <tr>
        <td>InternVL2.5-4B</td>
        <td style="text-align:center;"><strong>85.85</strong></td><td style="text-align:center;">84.37</td>
        <td style="text-align:center;"><strong>97.17</strong></td><td style="text-align:center;">96.43</td>
        <td style="text-align:center;"><strong>55.74</strong></td><td style="text-align:center;">54.44</td>
      </tr>

      <tr>
        <td>InternVL2.5-4B-MPO</td>
        <td style="text-align:center;"><strong>84.89</strong></td><td style="text-align:center;">83.58</td>
        <td style="text-align:center;"><strong>97.32</strong></td><td style="text-align:center;">96.88</td>
        <td style="text-align:center;"><strong>64.50</strong></td><td style="text-align:center;">58.54</td>
      </tr>

      <tr>
        <td>InternVL2.5-8B</td>
        <td style="text-align:center;"><strong>87.42</strong></td><td style="text-align:center;">86.90</td>
        <td style="text-align:center;"><strong>98.07</strong></td><td style="text-align:center;">97.77</td>
        <td style="text-align:center;"><strong>62.42</strong></td><td style="text-align:center;">59.92</td>
      </tr>

      <tr>
        <td>InternVL2.5-8B-MPO</td>
        <td style="text-align:center;"><strong>87.25</strong></td><td style="text-align:center;">86.03</td>
        <td style="text-align:center;"><strong>98.56</strong></td><td style="text-align:center;">98.22</td>
        <td style="text-align:center;"><strong>73.51</strong></td><td style="text-align:center;">68.98</td>
      </tr>

      <tr>
        <td>Gemini 2.0 Flash-non-thinking</td>
        <td style="text-align:center;">89.52</td><td style="text-align:center;"><strong>90.04</strong></td>
        <td style="text-align:center;">88.00</td><td style="text-align:center;"><strong>89.69</strong></td>
        <td style="text-align:center;">62.51</td><td style="text-align:center;"><strong>64.28</strong></td>
      </tr>

      <tr>
        <td>Gemini 2.0 Flash-thinking</td>
        <td style="text-align:center;"><strong>91.27</strong></td><td style="text-align:center;">90.66</td>
        <td style="text-align:center;">91.47</td><td style="text-align:center;"><strong>92.46</strong></td>
        <td style="text-align:center;">71.40</td><td style="text-align:center;"><strong>74.68</strong></td>
      </tr>

      <tr>
        <td>Gemini 2.5 Flash-non-thinking</td>
        <td style="text-align:center;">90.04</td><td style="text-align:center;"><strong>90.22</strong></td>
        <td style="text-align:center;"><strong>93.85</strong></td><td style="text-align:center;">74.52</td>
        <td style="text-align:center;"><strong>74.59</strong></td><td style="text-align:center;">55.31</td>
      </tr>

      <tr>
        <td>Gemini 2.5 Flash-thinking</td>
        <td style="text-align:center;">90.39</td><td style="text-align:center;"><strong>90.48</strong></td>
        <td style="text-align:center;">95.14</td><td style="text-align:center;"><strong>95.19</strong></td>
        <td style="text-align:center;"><strong>72.48</strong></td><td style="text-align:center;">72.00</td>
      </tr>

    </tbody>
  </table>
</div>

The failure of VLMs in MM-ICL could stem from two factors: 1) the models lack the ability to learn from demonstrations, 2) the support rationales themselves are of low quality and therefore uninformative. To reduce the possibility of 2), we enhance the rational capability by filtering incorrect samples and injecting ground-truth rationales, as shown in [Table 5](#tab-quality-r) (full tables in the Appendix).

<div style="overflow-x:auto; max-width:100%; display:flex; justify-content:center;" id="tab-quality-r">
  <table>
    <caption style="caption-side:top;">
      Table 5: Comparison of the quality of the rationales. Filter: filter out the incorrect support samples. gt R: add ground truth rationale as the input for support set reasoning generation.
    </caption>
    <thead>
      <tr>
        <th rowspan="2" style="text-align:left;">Ablation</th>
        <th colspan="4" style="text-align:center;">A-OKVQA</th>
        <th colspan="4" style="text-align:center;">ScienceQA</th>
        <th colspan="4" style="text-align:center;">M&#179;CoT</th>
      </tr>
      <tr>
        <th style="text-align:center;">1</th>
        <th style="text-align:center;">2</th>
        <th style="text-align:center;">4</th>
        <th style="text-align:center;">8</th>
        <th style="text-align:center;">1</th>
        <th style="text-align:center;">2</th>
        <th style="text-align:center;">4</th>
        <th style="text-align:center;">8</th>
        <th style="text-align:center;">1</th>
        <th style="text-align:center;">2</th>
        <th style="text-align:center;">4</th>
        <th style="text-align:center;">8</th>
      </tr>
    </thead>
    <tbody>
      <!-- VL-Rethinker-7B -->
      <tr>
        <th colspan="13" style="text-align:center; background-color:#f3f4f6;">VL-Rethinker-7B</th>
      </tr>
      <tr>
        <td style="text-align:left;">baseline</td>
        <td style="text-align:center;">85.50</td><td style="text-align:center;">84.98</td><td style="text-align:center;">84.98</td><td style="text-align:center;">85.68</td>
        <td style="text-align:center;">90.23</td><td style="text-align:center;">90.23</td><td style="text-align:center;">90.18</td><td style="text-align:center;">90.23</td>
        <td style="text-align:center;">68.08</td><td style="text-align:center;">69.15</td><td style="text-align:center;">68.59</td><td style="text-align:center;">68.55</td>
      </tr>
      <tr>
        <td style="text-align:left;">+filter</td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.35</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.17</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.43</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.61</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.49</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.05</span></td>
        <td style="text-align:center;">+0.00</td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.49</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.18</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.17</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.52</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.65</span></td>
      </tr>
      <tr>
        <td style="text-align:left;">+gt R</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.35</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.43</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.17</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.35</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.10</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.44</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.30</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.15</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.95</span></td>
        <td style="text-align:center;"><span style="color:#b32424;">-1.42</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.26</span></td>
        <td style="text-align:center;">+0.00</td>
      </tr>
      <tr>
        <td style="text-align:left;">+gt R &amp; filter</td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.35</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.05</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.96</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.09</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.49</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.49</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.20</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.35</span></td>

        <td style="text-align:center;"><span style="color:#1a7f37;">+0.04</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.82</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.38</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.99</span></td>
      </tr>

      <!-- InternVL2.5-8B-MPO -->
      <tr>
        <th colspan="13" style="text-align:center; background-color:#f3f4f6;">InternVL2.5-8B-MPO</th>
      </tr>
      <tr>
        <td style="text-align:left;">baseline</td>
        <td style="text-align:center;">86.03</td><td style="text-align:center;">84.89</td><td style="text-align:center;">84.54</td><td style="text-align:center;">81.92</td>
        <td style="text-align:center;">98.07</td><td style="text-align:center;">98.22</td><td style="text-align:center;">97.57</td><td style="text-align:center;">96.48</td>
        <td style="text-align:center;">68.98</td><td style="text-align:center;">67.56</td><td style="text-align:center;">65.96</td><td style="text-align:center;">63.37</td>
      </tr>
      <tr>
        <td style="text-align:left;">+filter</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.13</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.61</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.26</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.84</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.10</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.10</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.10</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.10</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.04</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.69</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.21</span></td>
        <td style="text-align:center;"><span style="color:#b32424;">-2.41</span></td>
      </tr>
      <tr>
        <td style="text-align:left;">+gt R</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.78</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.05</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.09</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.92</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.05</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.10</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.25</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.35</span></td>

        <td style="text-align:center;"><span style="color:#1a7f37;">+0.99</span></td>
        <td style="text-align:center;">+0.00</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+1.43</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.39</span></td>
      </tr>
      <tr>
        <td style="text-align:left;">+gt R &amp; filter</td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.52</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.52</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.61</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+2.18</span></td>

        <td style="text-align:center;">+0.00</td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.65</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.25</span></td>
        <td style="text-align:center;"><span style="color:#d1242f;">-0.05</span></td>

        <td style="text-align:center;"><span style="color:#d1242f;">-0.04</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.56</span></td>
        <td style="text-align:center;"><span style="color:#1a7f37;">+0.48</span></td>
        <td style="text-align:center;">+0.00</td>
      </tr>
    </tbody>
  </table>
</div>

We found that improving rationale quality, either by filtering out incorrect support samples or injecting ground truth rationale, does not consistently lead to improved performance. In several cases, applying filters to remove incorrect samples slightly degrades performance. One possible reason behind the performance drop is that the filtering operation causes a reduction in support sample diversity and coverage, suggesting that support set sufficiency and diversity may play a more critical role than the information quality for the current VLMs <d-cite key="zhang2022automaticchainthoughtprompting, qin2024factors" />. This implies that the evaluated VLMs are insensitive to the information quality in the demos, further reinforcing the conclusion that **VLMs still lack true MM-ICL capabilities to effectively learning from demonstration data**. 

### Assessing the Role of Retriever Methods

<div id="fig-jices-two-plots-trimmed" style="text-align:center; width:100%;">
  <div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-rethinking-mmicl/mmretriever_random_diff_llamabase.pdf" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-rethinking-mmicl/mmretriever_random_diff_llavacot.pdf" class="img-fluid rounded z-depth-1" %}
    </div>
  </div>
</div>
<div class="caption">
Figure 6: Comparison of Multimodal Retriever vs. Random Selection on 6 vision-language datasets. Left: Llama-3.2-11B-Vision-Instruct, Right: LLaVA-CoT.
</div>

In [Figure 6](#fig-jices-two-plots-trimmed), we compare the performance of the multimodal retriever (MM-retriever) against random selection. For base models, MM-retriever improves performance on M³CoT, ScienceQA and OK-VQA—suggesting that simple retrieval based on input similarity can be beneficial when no rationales are involved in the context. However, for reasoning models, MM-retriever consistently underperforms compared to random selection, especially on reasoning-intensive datasets. 

Prior work has suggested that in-context learning often operates via majority voting or pattern matching over the demonstrations <d-cite key="baldassini2024makes, qin2024factors" />. In base models without explicit reasoning, retrieving demonstrations with similar inputs (e.g., similar image-question pairs) often yields support examples with highly similar answers. This facilitates a form of shallow copying, where the model infers the correct answer by identifying consistent patterns across demonstrations and can easily mimic the format or final answer from them, which outperforms random sampling, where the patterns of support and query samples are usually different.

However, for reasoning-augmented models, this heuristic breaks down. Even when the input similarity is high, the corresponding rationales can be diverse in content, structure, and logic chain. Because MM-retriever selects support examples based solely on input similarity (image and question), it does not account for whether the reasoning paths in the retrieved examples are consistent or relevant to the current query. As a result, the retrieved demonstrations may not form a coherent support set, making it harder for the model to extract useful patterns. In contrast, random sampling may introduce a more diverse set of rationales <d-cite key="zhang2022automaticchainthoughtprompting" />, which while not tailored, can better expose the model to varied reasoning styles and prevent overfitting to a specific (and possibly irrelevant) reasoning trajectory. This may explain why MM-retriever underperforms random sampling in reasoning-intensive MM-ICL settings, despite its advantage in more shallow or pattern-based tasks.

### ID versus OOD with Modern VLMs

<div id="fig-compare-id-ood" style="text-align:center; width:100%;">
  {% include figure.liquid path="assets/img/2026-04-27-rethinking-mmicl/ood_id_updated.pdf" %}
</div>
<div class="caption">
Figure 7: Performance difference between OOD and ID settings on ScienceQA and A-OKVQA.
</div>

To echo our original motivation and experiments in previous sections, we also benchmark the performances of MM-ICL with ID and OOD support sets on ScienceQA and A-OKVQA and summarize the results in [Figure 7](#fig-compare-id-ood). Unlike OpenFlamingo on TextVQA/OK-VQA, we notice a mixed trend across the number of shots between each model. We observe that for modern VLMs, when presented with the same number of demonstrations, it's possible that the OOD setting wins over the ID setting, regardless of datasets or models. This is possibly because, after we removed the format inconsistency, capable VLM/VLM reasoners are no longer easily misled by the answer format in the demos. With the use of LLM judges as answer evaluators, the conjectured performance gap between ID and OOD is further narrowed. However, we want to emphasize that these results do not imply the success of MM-ICL, since most of these few-shot results can't even match zero-shot performance. Interestingly, we also notice that the winner of this duel between OOD and ID tends to be consistent among different shots, as well as among models of the same type. For example, on A-OKVQA, Qwen2.5-VL-3B consistently performs better with OOD demos, and its reasoner variant VLM-R1 also inherits this ability. A similar phenomenon is observed for the pair of Llama-3.2V-11B and LLaVA-CoT on both datasets. This further suggests that the MM-ICL capability of a VLM, including its robustness to OOD support data, is not significantly impacted during the RL training for reasoning.

## Why MM-ICL Falls Short: An Attention-Level Perspective

<div id="fig-prefix-matching" style="text-align:center; width:100%;">
  {% include figure.liquid path="assets/img/2026-04-27-rethinking-mmicl/pfx_assemble.pdf" %}
</div>
<div class="caption">
Figure 8: Prefix matching score heatmaps for LLaMA-3.1-8B-Instruct, LLaMA-3.2-11B-Vision-Instruct, Qwen2.5-7B-Instruct and Qwen2.5-VL-7B-Instruct.
</div>

To better understand why MM-ICL underperforms compared to its language-only counterpart, we examine the underlying attention mechanisms that enable ICL. <d-cite key="olsson2022context"/>  demonstrated that in LLMs, induction heads are specialized attention heads that appear to be the primary source of ICL. These heads operate by attending from the second occurrence of a token back to its earlier occurrence and then boosting the probability of the token that followed, allowing models to exploit repeated structures in context. To measure this effect for VLMs, we adopt the *prefix matching* protocol following <d-cite key="crosbie2025inductionheadsessentialmechanism" />: we generate a sequence of 50 random tokens, excluding the most and least common tokens, repeat this sequence four times, and prepend a start-of-sequence token. We then compute the attention pattern and define the prefix matching score as the average attention mass from a given token back to the tokens that preceded it in earlier repeats. For VLMs, we adapt this setup by repeating the image and interleaving it with the text, and calculate the prefix matching scores for all image and text tokens, respectively. In [Figure 8](#fig-prefix-matching), we found that LLMs exhibit a noticeable band of heads with high prefix scores. In contrast, its VLM counterparts exhibit lower prefix scores, indicating a weakening of the induction heads. Moreover, within VLMs, prefix matching is substantially stronger for text tokens (e.g., 0.937 for Llama-3.2-11B-V and 0.979 for Qwen2.5-VL-7B) than for image tokens (0.612 and 0.600), highlighting that image representations are especially deficient in supporting induction-like behavior. This gap potentially explains why MM-ICL falls short: VLMs struggle with prefix matching and therefore fail to leverage repeated multimodal demonstrations in the way LLMs do with text, thus cannot reliably exploit even basic patterns, let alone reasoning. Our results point to a deeper architectural limitation and suggest that extending induction-like mechanisms into multimodal attention may be crucial for robust MM-ICL.

## Conclusion, Limitations and Future Direction

Our study revisits the assumption that VLMs perform genuine MM-ICL. Under varying conditions and distribution shifts, we find that current VLMs often fail to utilize demonstrations meaningfully, relying instead on shallow cues. Our proposed MM-ICL with reasoning pipeline provides a stronger testbed; however, models exhibit limited sensitivity to shot count, retrieval method, and rationale quality. We further provide explanation by showing that weak prefix matching and absent induction heads underlie MM-ICL failure. **Limitations and future directions.** We focus on diagnosing MM-ICL weaknesses in VLMs without proposing architectural or training interventions. Future work could investigate architectural and scaling factors that enable induction-head-like mechanisms to emerge in multimodal settings, which may ultimately strengthen their ability to perform MM-ICL.

<!-- ## Equations

This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
You just need to surround your math expression with `$$`, like `$$ E = mc^2 $$`.
If you leave it inside a paragraph, it will produce an inline expression, just like $$ E = mc^2 $$.

To use display mode, again surround your expression with `$$` and place it as a separate paragraph.
Here is an example:

$$
\left( \sum_{k=1}^n a_k b_k \right)^2 \leq \left( \sum_{k=1}^n a_k^2 \right) \left( \sum_{k=1}^n b_k^2 \right)
$$

Note that MathJax 3 is [a major re-write of MathJax](https://docs.mathjax.org/en/latest/upgrading/whats-new-3.0.html)
that brought a significant improvement to the loading and rendering speed, which is now
[on par with KaTeX](http://www.intmath.com/cg5/katex-mathjax-comparison.php). -->
