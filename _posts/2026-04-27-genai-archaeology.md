---
layout: distill
title: Generative AI Archaeology
description: We document the rise of the Generative AI Archaeologist, whose tools include linear algebra and probability theory, jailbreaking, and debuggers, compared to the metal detectors, pickaxes, and radar surveys of traditional archaeology. GenAI Archaeologists have reported findings both through luck by observing unexpected behaviour in publicly accessible models, and by exploiting the mathematical properties of models. In this blog, we survey five types of findings unearthed by GenAI Archaeologists and discuss the status of those findings.
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
bibliography: 2026-04-27-genai-archaeology.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Inferring Training Data
  - name: Discovering Model Training Procedures
  - name: Stealing Part of a Deployed Model
  - name: Exfiltrating System Prompts
  - name: Outlook
  - name: Acknowledgements
--- 

## Introduction

{% include figure.liquid path="assets/img/2026-04-27-genai-archaeology/intro_images.png" class="img-fluid" %}
Figure 1: Examples of archaeological discoveries (a) <a href="https://commons.wikimedia.org/w/index.php?curid=17377542">Göbekli Tepe</a> was found through professional survey, (b) the <a href="https://commons.wikimedia.org/w/index.php?curid=48964317">Terracotta Army</a> was found by farmers by complete luck, and (c) the <a href="https://commons.wikimedia.org/w/index.php?curid=164647291">Antikythera mechanism</a> was found by coincidence in shipwreck.

Archaeology reveals the secrets of human history through the traces left behind by our ancestors.
Some discoveries are the result of careful survey using sophisticated tools to uncover what cannot be seen by the naked eye.
One of the earliest examples of human village habitation is found at Göbekli Tepe, Turkey, first discovered in 1963, but only excavated in 1995.
This village offers important new insights about early farming culture that changed our understanding of early human history.
Other discoveries are the result of pure luck: the Terracotta Army of Xi'an, China, is one such example.
This grand funeral act for the First Emperor of China was described in ancient Chinese texts but lost to time; its location was only re-discovered in 1974 by a group of rural farmers. 
A little closer to the heart of computer scientists, the Antikythera mechanism was found in a shipwreck off the coast of Antikythera, Greece, in 1901.
This finding is more of a coincidence, since divers were already actively working in the area.
The mechanism is an analogue computer,  preserved for nearly 2,000 years, that was used to predict the position of celestial bodies.
Such is the sophistication of this device, that there is no evidence of similar complexity until 1,400 years later<d-cite key="marchant2006search"></d-cite>.

In contrast to studying historical physical artefacts, Generative AI models, such as LLMs, are digital objects that should not require anyone to search for lost knowledge.
However, the secretive nature of fully closed<d-cite key="eiras2024near"></d-cite> development has given rise to **Generative AI Archaeology**.
The tools of GenAI Archaeologists are a solid grasp of linear algebra and probability theory, jailbreaking, and `pdb`, compared to the metal detectors, pickaxes, and radar surveys of traditional archaeology.
GenAI Archaeologists have reported findings through luck by observing unexpected behaviour in publicly accessible models<d-cite key="nasr2025scalable"></d-cite><d-cite key="behnamghaderllm2vec"></d-cite>, and through careful study by exploiting the mathematical properties of the underlying model<d-cite key="carlini2021extracting"></d-cite><d-cite key="hayase2024data"></d-cite><d-cite key="finlayson2024logits"></d-cite><d-cite key="carlini2024stealing"></d-cite>.
In this blog, we will cover five findings unearthed by GenAI Archaeologists and discuss the current status of those findings.
We welcome comments on discoveries that we have overlooked, and discoveries that cover the broader class of machine learning systems.

## Inferring Training Data

{% include figure.liquid path="assets/img/2026-04-27-genai-archaeology/training_data.png" class="img-fluid" %}
Figure 2: (a) Hayase et al. show how to infer the distribution of data used to train a tokenizer based on how BPE constructs merge lists (image source: <a href="https://openreview.net/forum?id=EHXyeImux0">Figure 1</a>). (b) Nasr et al. demonstrate a simple attack to extract training data from language models by forcing them to repeat the same token (image source: <a href="https://openreview.net/forum?id=vjel3nWP2a">Figure 1</a>).

Training data is fuel that powers Large Language Models.
Open-science initiatives, such as Pythia<d-cite key="biderman2023pythia"></d-cite>, OLMo<d-cite key="groeneveld2024olmo"></d-cite>, and PleIAs<d-cite key="langlais2025pleias"></d-cite> document everything about their data, making it possible to understand the interplay between the training data and LLM<d-cite key="elazar2024whats"></d-cite><d-cite key="liu2025olmotrace"></d-cite>. 
This is in stark contrast to fully closed models<d-footnote>See news reports on the legal cases against OpenAI and Anthropic for recent examples of the lack of information shared about LLM training data: <a href="https://news.bloomberglaw.com/ip-law/openai-risks-billions-as-court-weighs-privilege-in-copyright-row">OpenAI Risks Billions as Court Weights Privilege in Copyright Row</a>. <a href="https://www.npr.org/2025/09/05/nx-s1-5529404/anthropic-settlement-authors-copyright-ai">Anthropic settles with authors in first-of-its-kind AI copyright infringement lawsuit</a>.</d-footnote>, but researchers have managed to learn some of the secrets of the data.

**Tokenizer Data:** Hayase et al. show how to infer properties of the data used to train a tokenizer, which underpins how LLMs process text<d-cite key="hayase2024data"></d-cite>.
Their approach, illustrated in Figure 2(a), is based on how Byte-Pair Encoding token merge lists are created, and the implications for data used to train the tokenizer.
This insight allowed them to accurately infer known properties about publicly disclosed models, and to predict the properties of private models.

**Pretraining Data:** Carlini et al. show how to extract verbatim sequences from the GPT-2 language model<d-cite key="carlini2021extracting"></d-cite>. 
Their work is <a href="https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html">primarily focused on understanding privacy attacks</a> on language models, in which a model may reveal personal identifying information.
More directly related to detecting copyright violations, Karamolegkou et al. prompted open and closed LLMs with prefixes of copyrighted material from books<d-cite key="karamolegkou2023copyright"></d-cite>.
Finally, Nasr et al. demonstrated a simple attack on ChatGPT that involved forcing it to generate the same token repeatedly, as shown in Figure 2(b)<d-cite key="nasr2025scalable"></d-cite>.
This eventually causes the model to "diverge" from its post-training objective and revert to its base model behaviour, in which it generates memorized training data.
It was possible to extract strings up-to 4,000 characters long using this method.

**Status**: it has never been confirmed if Hayase et al. inferred the true data distribution of the tokenizers for GPT-4 and Claude. 
It has never been confirmed if Carlini et al, Nasr et al, or Karamolegkou et al. succeeded in extracting the training data from the GPT or Claude models.

## Discovering Model Training Procedures

{% include figure.liquid path="assets/img/2026-04-27-genai-archaeology/model_training.png" class="img-fluid" %}
Figure 3: Parishad et al. stumbled upon evidence of how the Mistral-7B LLM was pretrained. These visualizations show the cosine similarity between encodings using causal attention and bidirection attention, across layers and token positions. The LLaMA2-7B model (a) shows low cosine similarities, especially at deeper layers, whereas Mistral-7B (b), shows the opposite, raising questions about how the model was pretrained (image source: <a href="https://openreview.net/forum?id=IW1PR7vEBf">Figure 5</a>.)

Exactly how LLMs are trained is becoming increasingly shrouded in mystery but some noteworthy explanations remain in the open-weight and open-science literature<d-cite key="biderman2023pythia"></d-cite><d-cite key="groeneveld2024olmo"></d-cite><d-cite key="langlais2025pleias"></d-cite>.
Nevertheless, given an LLM with unknown training process, researchers *can* sometimes discover behaviour that betrays tell-tale signatures of how a model was trained.

In trying to convert a LLM into a sentence embedding model, BenhamGhader et al.<d-cite key="behnamghaderllm2vec"></d-cite> stumbled upon evidence that the Mistral-7B LLM may have been pretrained with bidirectional attention.
Their approach was to enable bidirectional attention in the LLM, which would allow the model to create better sentence-level representations than unidirectional attention.
This new ability was trained using a combination of masked next token prediction and unsupervised contrastive learning.
However, it was discovered that the Mistral-7B model constructed nearly identical representations (Figure 3b) when using bidirectional attention or causal attention, compared to LLaMA2-7B (Figure 3a).
This *innate ability* in Mistral-7B seemed to rule out the possibility that it was only trained on next-token prediction.

In personal correspondence with Parishad BenhamGhader, she told me that the bidirectional attention finding came about as a result of a reviewer requesting additional models beyond LLaMa2-7B in the original manuscript.
The behaviour that was observed for the LLaMA-7B model did not appear in the Mistral-7B model, which sparked the additional analysis of the cosine similarities at different token positions before and after enabling bidirectional attention.
It was here that it became clear that the representations from the Mistral-7B model were extremely similar with or without bidirectional attention.
Further correspondence with Marius Mosbach, one of the collaborators on the project, revealed that he studied the Mistral inference code and found a flag in the `forward()` that could enable bidirectional attention.
This led to speculation that the model was trained using a PrefixLM-style objective<d-cite key="raffel2020exploring"></d-cite>, even though there are no details on this topic in the six-page preprint<d-cite key="jiang2023mistral7b"></d-cite>.

**Status**: it has never been confirmed whether this speculation is correct.
At the time of writing, the original Mistral inference code is no longer publicly available.

## Stealing Part of a Deployed Model

{% include figure.liquid path="assets/img/2026-04-27-genai-archaeology/stealing.png" class="img-fluid" %}
Figure 3: (a) Carlini et al. show that the difference between consecutive singular values in an SVD decomposition of a matrix of final token logits is aligned with the embedding size of the Pythia-1.4B LLM, and that this can be estimated efficiently (b). (image sources: <a href="https://proceedings.mlr.press/v235/carlini24a.html">Figure 1 and 2</a>.)

The size of most fully closed LLMs is not publicly known.
Furthermore, it is also not clear if they are dense models, mixtures-of-experts, or something else entirely.
Such is the lack of information that the GPT-4 Wikipedia article speculates that the model is somwhere between 1T--1.8T parameters<d-cite key="enwiki:1322967106"></d-cite>.
Researchers have turned to linear algebra to steal some of these details from LLMs.

**Hidden Dimension Size**: Finlayson et al.<d-cite key="finlayson2024logits"></d-cite> and Carlini et al.<d-cite key="carlini2024stealing"></d-cite> showed how to determine the hidden size of an LLM.
Carlini et al. relied on access to a model that returned the logits of the next token.
Their method works by initializing a matrix $$n \times l$$ matrix $$Q$$, where $$n$$ is much larger than the hypothesized dimensionality $$h$$ of the target model, and $$l$$ is the dimension of the returned logit vector. 
$$ Q $$ is populated with the logit vectors returned for each next token over a large set of random set of prefixes, based on the assumption that the $$l$$-dimension logits lie on the true $$h$$-dimension subspace of the model.
The largest difference in consecutive pairs of singular values aligns with the hidden dimension of the model, as shown in Figure 3(a).
Finlayson et al. used a similar technique to create an LLM image, from it was possible to estimate the embedding size of the API-based `gpt-3.5-turbo` model.

**Full Output Layer:** One can go further than inferring the hidden dimension side. 
Carlini et al. also showed how to extract the entire output layer of language models by observing that in the SVD decomposition of $$ Q = U \cdot \Sigma \cdot V^T $$, that $$U$$ is a linear transformation of the output layer.
They showed that this can accurately extract the output embedding layer of open-weights models, and they use the same attack to steal the output layer of OpenAI deployed models.

**Status**: it was acknowledged that the method of Carlini et al. correctly extract the size of the models. 
It was also confirmed that they could steal the output layer of the `ada` and `babbage` models with a root-mean squared error of $$ 5 \cdot 10^{-4} $$ and $$ 7 \cdot 10^{-4}$$, respectively, for just $4--12 dollars of API query credits.
Finlayson et al. reported that several fully closed models changed their APIs to prevent this information being stolen from their models<d-footnote>This was revealed in the <a href="https://openreview.net/forum?id=oRcYFm8vyB&noteId=aN0dV0Z7Od">COLM Author-Reviewer Discussion</a> of the Finlayson et al. article.</d-footnote>.

## Exfiltrating System Prompts

{% include figure.liquid path="assets/img/2026-04-27-genai-archaeology/prompt_exfiltration.png" class="img-fluid" %}
Figure 4: Zhang et al. showed that simply asking LLMs to translate inputs into a different language (a) can reveal the model system prompt (b).

This final finding is credited to Zhang et al. who show how to extract the system prompt of LLMs using a jailbreak attack<d-cite key="zhang2024effective"></d-cite>.
The attack involves prompting the model to translate everything into a different language, e.g. German, Korean, Portuguese, etc., as shown in Figure 4(a), which can cause the model to reply with its system prompts, as shown in Figure 4(b).
This technique can also be used to "unmask" the LLM used by a third-party service.

**Status**: The success of this technique can be directly confirmed for the <a href="https://docs.claude.com/en/release-notes/system-prompts">Claude models</a>.

## Outlook

The knowledge revealed in these studies has not been lost to time.
These models are not sitting at the bottom of the Mediterranean Sea, nor are they resting under the fields of Xi'an.
They are publicly distributed through the HuggingFace Models Hub or accessible through APIs, but they harbour deep secrets about how they were trained.
In the absence of this knowledge, researchers spend their time trying to decipher what has been secreted away inside startups and corporations.
This secretive behaviour is often justified with the claim that organizations need to maintain their competitive edge, but more openness could free up our time to focus on innovation instead of reproducing secrets. 