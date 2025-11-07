---
layout: distill
title: Research Directions in Multimodal Chain-of-Thought (MCoT) with Sketching
description: This article explores adding sketching to Multimodal Chain-of-Thought (MCoT)reasoning to enhance AI capabilities. It reviews past methods, identifies key gaps such as the lack of sketch-rationale datasets, and proposes advancing the field through targeted data collection, unified multimodal models, and reinforcement learning. Ethical considerations include mitigating cultural bias and visual misrepresentation in generated sketches.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# anonymize when submitting
authors:
  - name: Anonymous

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-mcot-sketching.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Motivation to incorporate drawing capabilities into AI
  - name: Related work
  - name: Future research for MCoT with sketching
    subsections:
      - name: Creating a new MCoT sketch dataset
      - name: Advancing MCoT with unified MLLMs
      - name: Improving MCoT with reinforcement learning (RL) and test-time scaling
  - name: Impact
  - name: Appendix A MCoT foundations
  - name: Appendix B MCoT template
---

## Introduction

Drawing and sketching are cognitive tools that humans use not only to express and communicate
thoughts, but also to generate new ones <d-cite key="Fan"></d-cite>. For this matter, we would like to equip any intelligent system with the same ability to improve and help it communicate its reasoning. First steps in this direction have been proposed within the field of Multimodal Chain-of-Thought (MCoT) where reasoning steps are enriched with data from different modalities, such as visuals. Therefore, future research on sketching should advance the design of MCoT reasoning strategies. Improving Multimodal Large Language Models (MLLMs) that perform such cross-modal reasoning is also relevant.

## Motivation to incorporate drawing capabilities into AI

Humans express and communicate ideas visually through drawing and sketching, which is a quick and
loose form of drawing. Drawing is a representation of thought, but also an activity that can support
ongoing cognition <d-cite key="Fan"></d-cite>. Drawing and sketching precede writing: The first documented drawings date back as far as 64,000 years <d-cite key="Hoffmann"></d-cite>. For that reason, Fan et al. <d-cite key="Fan"></d-cite> argue that drawing is one of the most enduring and versatile cognitive tools from which humans have benefited.

One explanation for the power of drawing and sketching can be derived from cognitive enhancement
and offloading strategies. According to Morrison and Richmond <d-cite key="Morrison2020"></d-cite>, technologies are used as external memories, facilitating other tasks by freeing up memory. Similarly, Osiurak et al. <d-cite key="Osiurak2018"></d-cite> show that tools such as maps can extend human’s cognitive abilities.

Given the relevance of drawing and sketching for human thought, expression, and communication,
we would want to equip any AI with the capability to also use this tool to advance and share its own
ideas. Sketching can not only be a window into how AI models process information, but it is fair to
assume that it can also support their reasoning.

Reasoning in large language models (LLMs) has been greatly improved with in-context learning (ICL) <d-cite key="Min2022RethinkingTR"></d-cite> and Chain-of-Thought (CoT) techniques <d-cite key="Nye, Wei"></d-cite>. ICL helps models with additional information added to the input to find appropriate responses for a given task. With CoT, the contextual information is specifically extended by a simulation of human reasoning steps, where a task is divided into subtasks for which intermediate solutions are given so that the model can derive its final answer from them. This can be achieved by eliciting reasoning through prompting, as with ’think step-by-step’ prompts (Zero-Shot-CoT <d-cite key="Kojima2022LargeLM"></d-cite>), or by providing the model with an explicit reasoning demonstration (also called a rationale) for a given problem (Few-Shot-CoT <d-cite key="Wei"></d-cite>). 

CoT has been extended with multimodal information <d-cite key="Wang2024ExploringTR, Wang2025MultimodalCR"></d-cite> where models receive more than text to guide them toward a correct answer. This information can consist of visual, auditory, or spatio-temporal data. Sketches would be additional visual information. They could also help models to offload complex tasks and retain intermediate memories, for example, of subtasks. Therefore, an implementation of the capability to sketch in order to enhance models’ reasoning abilities should expand existing research in MCoT. A detailed account of MCoT is given in **Appendix A**.

## Related work

Several recent approaches explore MCoT reasoning, though most do not fully integrate sketch
generation into the reasoning process.

Zhang et al. <d-cite key="Zhang2023MultimodalCR"></d-cite> propose a two-stage framework for multiple-choice reasoning for text and image inputs where a FLAN-AlpacaBase model <d-cite key="taori_alpaca_2023, Zheng2023JudgingLW"></d-cite> first produces a rationale, then derives the answer. Fusing text and image features from the input improves performance, but the system cannot generate new visual content. This limits applicability to reasoning scenarios that benefit from active visual exploration, such as diagram construction in geometry or mechanical design tasks.

Meng et al. <d-cite key="Meng2023ChainOI"></d-cite> extend CoT by having an LLM produce symbolic sketch-like diagrams (e.g., with SVG), rendered into images and re-encoded for reasoning. Their ’think image by image’ approach helps, for example, with geometric tasks. However, this gain comes at the cost of operational complexity: the pipeline depends on separate LLMs, rendering engines, and encoders, creating latency and integration challenges. Unified MLLMs avoid such fragmentation and may better support generalization by learning a shared latent space for both text and sketches.

In contrast to the previous two approaches, Liao et al. <d-cite key="Liao2025ImageGenCoTET"></d-cite> fine-tune unified MLLMs (SEED-LLaMA <d-cite key="Ge2023MakingLS"></d-cite> and SEED-X <d-cite key="Ge2024SEEDXMM"></d-cite>) on their ImageGen-CoT dataset. Reasoning steps of their models precede image generation. Test-time scaling is applied to select better outputs. While they demonstrate high-quality image generation, their evaluation focuses on aesthetics and relevance rather than measurable reasoning improvement. For reasoning-centric applications, visual fidelity without explicit reasoning gains may be insufficient.

Hu et al. <d-cite key="Hu2024VisualSS"></d-cite> and Vinker et al. <d-cite key="Vinker2024SketchAgentLS"></d-cite> develop agentic strategies (Sketchpad, Sketchagent) where models like GPT-4o <d-cite key="Hurst2024GPT4oSC"></d-cite> or Claude3.5-Sonnet <d-cite key="TheC3"></d-cite> can decide to produce or modify sketches during problem-solving by leveraging external vision models, Python or a domain-specific language (DSL) for sketches. Models with Sketchpad iterate over a ’thought’, ’action’ (to inject sketches), and ’observation’ pattern. With this approach, Hu et al. <d-cite key="Hu2024VisualSS"></d-cite> show that allowing models to decide to insert sketches during reasoning leads to notable performance gains. However, the framework relies on external vision models to rather enhance or dissect images and a Python sketch representation, which may not capture the nuances of freehand or abstract sketches common in human reasoning.

A truly multimodal approach for sketches would not use Python or DSLs to ’implicitly’ generate
figures that the model ingests as textual input. However, few multimodal datasets that combine
visuals with rationales exist. While QuickDraw <d-cite key="Jongejan_quick_draw"></d-cite> provides scale and diversity in sketch data, its lack of accompanying rationales prevents multimodal alignment learning. ScienceQA <d-cite key="lu2022learn"></d-cite> and ImageGen-CoT <d-cite key="Liao2025ImageGenCoTET"></d-cite> offer strong rationale-image pairs, but the absence of sketches means they primarily serve full-image reasoning rather than schematic reasoning. This gap suggests that the field currently lacks a dataset that balances sketch simplicity with reasoning, a pairing that could uniquely advance MCoT.

Overall, existing MCoT work shows that visual information, including sketches, can aid reasoning.
However, limitations remain: most systems either consume but do not create sketches, focus on image
quality rather than reasoning improvement, or require orchestration of multiple models instead of
unified generation. Furthermore, appropriate datasets with sketches in combination with rationales
are lacking.

## Future research for MCoT with sketching

Given the power of visual information for reasoning tasks, as shown by <d-cite key="Zhang2023MultimodalCR, Meng2023ChainOI, Liao2025ImageGenCoTET, Hu2024VisualSS"></d-cite>, some of the shortcomings of existing MCoT approaches can be addressed to better incorporate sketching in future research.

### Creating a new MCoT sketch dataset

To facilitate the training of MLLMs, the lack of an appropriate dataset with sketching and rationales is a limitation.

Sketch data should be gathered and grouped within different categories, depending on the downstream
task (consider Figure 1). In experimental studies with humans, Huey et al. <d-cite key="Huey2023VisualEP"></d-cite> point out that drawings differ according to their intended goal: visual explanations by the participants emphasized moving and interactive parts, while their visual depictions focused on salient features. Hu et al. <d-cite key="Hu2024VisualSS"></d-cite> show that adding auxiliary lines to geometric figures helps multimodal models such as GPT-4o to infer correct answers about these figures. Fan et al. <d-cite key="Fan"></d-cite> highlight that not all drawings are faithful depictions, but can also be abstractions whose meanings are conveyed by cultural conventions.

{% include figure.liquid path="assets/img/2026-04-27-mcot-sketching/sketches.png" class="img-fluid" %}**Figure 1:** Different types of sketches and drawings: (a) depicts a geometric form that has an auxiliary line, (b) emphasizes moving parts of a machine, (c) depicts the same machine in more detail, (d) represents figures from tetris whose next moves are indicated with arrows, (e) is a conventional sketch of a heart that does not resemble actual human hearts.

To integrate sketches into a CoT, training data should not only consist of images of drawings and
sketches, but combine these with textual rationales. This would enable multimodal alignment between
visual and linguistic reasoning steps. A typical template for this data could consist of instruction *I*, query *Q*, rationale *R*, and answer *A* where we could further divide *R* into ’thought’, ’sketch’, and ’observation’ with respective special tokens to guide the model, loosely following Hu et al. <d-cite key="Hu2024VisualSS"></d-cite>. An example template is given in **Appendix B**. Since ScienceQA and ImageGen-CoT already pair images with rationales, they could be extended with sketches to strengthen visual-textual alignment for their tasks.

### Advancing MCoT with unified MLLMs

To avoid multi-model orchestration and to leverage potential transfer-learning effects, further advancing reasoning of MLLMs with sketches is a promising direction. However, there exist only a few MLLMs <d-cite key="Yu2023ScalingAM, Zhao2025R1OmniEO, Zhang2023MultimodalCR, swerdlow2025unidisc"></d-cite> that can potentially handle sketch-to-text as well as text-to-sketch tasks within a unified architecture (consider Figure 2). The majority of current approaches such as Sketchpad pair VLMs such as Flamingo <d-cite key="Alayrac2022FlamingoAV"></d-cite>, PaLM-E <d-cite key="Driess2023PaLMEAE"></d-cite>, LLAVA <d-cite key="Liu2023VisualIT"></d-cite>, GPT-4o <d-cite key="Hurst2024GPT4oSC"></d-cite>, or Claude3-Opus and Claude3.5-Sonnet <d-cite key="TheC3"></d-cite> with text-to-image models.

Unified MLLMs can be divided into autoregressive (AR) and diffusion-based MLLMs. For example,
CM3Leon <d-cite key="Yu2023ScalingAM"></d-cite> from Meta is a Transfomer-based AR decoder that can generate both text and images. It is built on the CM3 model <d-cite key="Aghajanyan2022CM3AC"></d-cite>. CM3Leon has been trained on text-guided image editing, image-to-image grounding tasks where visual features can be derived from images, and text-to-image generations.

Swerdlow et al. <d-cite key="swerdlow2025unidisc"></d-cite> introduce a unified multimodal discrete diffusion model (UniDisc). While the model’s architecture consists of a Transformer (bidirectional) decoder, its training goal is not to auto-regressively predict the next tokens in a sequential manner (e.g., left to right for text or top to bottom for image patch rasters), but to predict the distribution of tokens via a denoising process that allows parallel predictions as well as later refinements. The training of UniDisc is realized with a denoising process of corrupted inputs (masking). In contrast to continuous diffusion models, Swerdlow et al. <d-cite key="swerdlow2025unidisc"></d-cite> use discrete noising and denoising for both images and texts. Swerdlow et al. <d-cite key="swerdlow2025unidisc"></d-cite> show that UniDisc outperforms the same architecture without a diffusion objective with respect to image and text classification tasks. The model is also capable of inpainting and infilling missing parts of an input, which no AR model can do. However, these performance gains come at a cost: UniDisc requires 13.2 times longer than its AR counterpart to reach equivalent loss levels <d-cite key="swerdlow2025unidisc"></d-cite>.

{% include figure.liquid path="assets/img/2026-04-27-mcot-sketching/MCoT.png" class="img-fluid" %}**Figure 2:** MCoT involving sketches with a Multimodal Large Language Model (MLLM). Black arrows represent sequential auto-regressive processing, while blue arrows illustrate the bidirectionality of diffusion models. The model’s reasoning is guided by special tokens, such as \<think\>.

Models like UniDisc provide an interesting model class for MCoT. While current diffusion language
models (DLMs) might not rival AR LLMs due to training inefficiencies <d-cite key="swerdlow2025unidisc"></d-cite> or speed <d-cite key="dream2025"></d-cite>, the
strength of multimodal DLMs in handling and generating multimodal data – as shown by Swerdlow
et al. <d-cite key="swerdlow2025unidisc"></d-cite> – warrants further research. Their ability to inpaint and infill would be particularly helpful for amending visualizations, which is a core aspect of explanatory sketching. Research in this direction could be informed by Diffusion-of-Thought (DoT) proposed by Ye et al. <d-cite key="Ye2024DiffusionOT"></d-cite>, who fine-tune a DLM for CoT. However, diffusion models require a fixed output size. This is a challenge that needs to be addressed to allow versatile reasoning over different tasks.

### Improving MCoT with reinforcement learning (RL) and test-time scaling

Existing work on MCoT <d-cite key="Zhang2023MultimodalCR, Meng2023ChainOI, Liao2025ImageGenCoTET"></d-cite> has mainly relied on supervised fine-tuning (SFT). However, other work in *reasoning* has shown that RL leads to improvements <d-cite key="DeepSeekAI2025DeepSeekR1IR, Ranaldi2025MultilingualRV, Zhao2025R1OmniEO"></d-cite>. Therefore, MCoT could be advanced with Direct Preference Optimization (DPO) <d-cite key="Rafailov2023DirectPO"></d-cite>, Reinforcement Learning with Verifiable Rewards (RLVR) <d-cite key="DeepSeekAI2025DeepSeekR1IR"></d-cite> and Group Relative Policy Optimization (GRPO) <d-cite key="Shao2024DeepSeekMathPT"></d-cite> strategies. One straight-forward application would be to use RLVR with GRPO, following Deepseek’s R1 <d-cite key="DeepSeekAI2025DeepSeekR1IR"></d-cite>, to reward accuracy ($$R_{acc}$$) and format ($$R_{format}$$) for rationales and answers based on generated sketches.

An appropriate reward for the generation of sketches could leverage AR-GRPO for autoregressive
MLLMs <d-cite key="Yuan2025ARGRPOTA"></d-cite>. AR-GRPO realizes rewards for the generation of images with a multi-faceted reward function that ensures (a) consistency with the textual input condition through CLIP <d-cite key="Radford2021LearningTV"></d-cite> and Human Preference Score v2 <d-cite key="Wu2023HumanPS"></d-cite>, (b) image quality with MANIQA <d-cite key="Yang2022MANIQAMA}"></d-cite>, and (c) a further realism reward through a VLM, such as Qwen2.5-VL-3B-Instruct <d-cite key="Bai2025Qwen25VLTR"></d-cite>. This function is used with GRPO to improve the quality of generated images. Since the proposed rewards by Yuan et al. <d-cite key="Yuan2025ARGRPOTA"></d-cite> focus on overall quality, a specific reward should be conceived for sketches. For example, a sketch can consist of a hierarchy of strokes whose meaning can be of different importance. It would be interesting to incorporate this somehow into the reward: Should sketches with a limited amount of strokes be prioritized?

In the wake of Liao et al. <d-cite key="Liao2025ImageGenCoTET"></d-cite>, existing MCoT could be further improved with Test-time scaling methods, sampling more CoTs and sketches to select the best candidates with an appropriate scoring method. This approach could also be used with agentic frameworks that pair VLMs with image generators and would not require any additional training of the models.

Beyond standard accuracy on downstream tasks, evaluation should measure how sketches contribute
to the reasoning process. This includes interpretability (e.g., can a human follow the model’s
reasoning with a sketch?), task completion time (one of the biggest bottlenecks because image
generation requires many tokens), error localization, and robustness under noisy or incomplete inputs. Additionally, user studies could assess subjective clarity and helpfulness of generated sketches.

## Impact

MLLMs with sketching would have an impact on AI in different domains. For example, agentic
systems such as Auto-GUI <d-cite key="Zhang2023YouOL"></d-cite> that interact with graphical user interfaces or websites could be enhanced by providing them with additional visual information with sketches. Similarly, embodied AI systems, such as EmbodiedGPT <d-cite key="Mu2023EmbodiedGPTVP"></d-cite> whose backbone uses a combination of vision and language models that help navigate the real world, could reason about their surroundings using sketches. MLLMs for STEM education could also benefit from the ability to make their reasoning more transparent with additional drawings as proposed in Meng et al. <d-cite key="Meng2023ChainOI"></d-cite>. In sum, sketching would help all reasoning models not only to enhance their thoughts, but also communicate them with more than
one modality.

As with language, sketches are not neutral representations. The ability of AI systems to generate and reason with sketches introduces risks of cultural bias, visual misrepresentation, and domain-specific inaccuracies. For example, the “heart” symbol in Figure 1(e) is globally recognized in popular culture but anatomically incorrect; in medical education, reasoning over such a schematic could reinforce misconceptions. Similar issues may arise if models default to culturally specific diagrammatic conventions, omit critical features due to dataset biases, or overgeneralize from training examples.

Ethical safeguards should address the entire MCoT-with-sketching workflow. Dataset curation must
ensure diversity of styles, cultural perspectives, and schematic conventions. Annotation guidelines
should clarify the intended use and accuracy requirements of sketches. Model evaluation should
include bias detection for visual outputs, alongside interpretability checks so users can trace how a sketch influenced reasoning.

## Appendix A MCoT foundations

Following Wang et al. <d-cite key="Wang2025MultimodalCR"></d-cite>, we can define prompt, instruction, query, answer, and rationale with $$P$$ , $$I$$, $$Q$$, $$A$$, and $$R$$, which are all token sequences. A Chain-of-Thought (CoT) would be:

\begin{equation}
    P_{CoT} = \{I, (x_1, e_1, y_1), ..., (x_n, e_n, y_n)\}
\end{equation}

where $$x_i \in Q$$ and $$y_i \in A$$ are questions with corresponding answers and $$e_i \in R$$ is an example rationale. The joint probability of generating an answer A and a rationale R given the prompt $$P_{CoT}$$ and a query $$Q$$ would be <d-cite key="Wang2025MultimodalCR"></d-cite>:

\begin{equation}
    p(A, R |P_{CoT}, Q) = p(R |P_{CoT}, Q) \cdot p(A |P_{CoT}, Q, R)
\end{equation}

where the model should output rationale $$R$$ with the tokens $$r_1, ..., r_i$$ before arriving at the answer $$A$$ consisting of the tokens $$a_1, ..., a_i$$. The goal in training a reasoning model $$F$$ is to jointly maximize the likelihood of equation (2).

Finally, all components $$P$$, $$Q$$, $$A$$, and $$R$$ can be enriched with multimodal information $$\mathcal{M}$$. For example with MCoT, a rationale $$R$$ should handle $$\mathcal{M}$$ input and generate multimodal information (e.g., a sketch) as well as text $$T$$, that is, $$R\in\{M, M\oplus T\}$$ <d-cite key="Wang2025MultimodalCR"></d-cite>.

## Appendix B MCoT template

``
{
    "instruction": "Find proofs for geometry problems.",
    "query": "Prove the angles of ABC provided in the attached image sum to 180. <image> VT_011 VT_115 VT_563 VT_101 ... VT_909 </image>",
    "rationale": "<think> I need to figure out how ABC are related in the image. The image shows a triangle. I need to prove that the angles of the triangle sum to 180. To find an answer, I draw a triangle: Let's call it ABC. <sketch> VT_421 VT_105 VT_983 VT_002 ... VT_778 </sketch> I extend the sides from A to B, from A to C, and from B to C. <sketch> VT_421 VT_105 VT_983 VT_001 ... VT_708 </sketch> I draw a line parallel to AB through point C. <sketch> VT_420 VT_105 VT_983 VT_001 ... VT_718 </sketch> <observe> The angles at point C created by the parallel line correspond to the interior angles at points A and B. When I add those angles up, they form a straight line at point C, which measures 180. Since those angles correspond exactly to the three interior angles of the triangle, the sum of the interior angles is 180. </observe> This proof follows from the alternate interior angles theorem. </think>",
    "answer": "The alternate interior angles theorem shows that all angles at point C created by the parallel line sum to 180. They further correspond to the interior angles at points A and B. Therefore, the angles of ABC provided in the attached image sum to 180."
}
``

MCoT template with instruction $$I$$, query $$Q$$, rationale $$R$$, and answer $$A$$ where $$R$$ is further divided into "thought", "sketch", and "observation" with respective special tokens to guide the model. VT_n tokens correspond to image tokens.