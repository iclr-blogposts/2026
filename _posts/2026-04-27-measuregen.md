---
layout: distill
title: On the Measure of a Model - From Intelligence to Generality
description: Benchmarks like ARC, Raven-style puzzles, and the Blackbird Task are often treated as measures of LLM intelligence. But intelligence is a moving target—hard to define and even harder to link to what we actually need models to do, like answer questions, summarize text, or write code. Optimizing for these abstract tests can pull evaluation away from real-world usefulness. We argue for a shift from chasing intelligence to measuring generality. This reframes how progress in AI should be assessed and proposes generality as a more stable foundation for evaluating capability across diverse and evolving tasks.

date: 2026-04-27
future: true
htmlwidgets: true
#hidden: true

# Mermaid diagrams
#mermaid:
  #enabled: true
  #zoomable: true

# Anonymize when submitting
authors:
- name: Anonymous

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
bibliography: 2026-04-27-measuregen.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Why Intelligence is Problematic
    subsections:
      - name: Conceptual Instability
      - name: Illusion of Competence
      - name: Rethinking our Ground
  - name: Unpacking Intelligence
    subsections: 
      - name: Generality is Independent
      - name: (Only) Generality is Necessary
  - name: What Generality Offers
  - name: Conclusion

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

Many people see large language models (LLMs) <d-cite key="radford2019language, brown2020language, burton2024large, steyvers2025large"></d-cite> as steps toward *artificial intelligence*—or even early hints of *artificial general intelligence* (AGI) <d-cite key="bubeck2023sparks, zhong2024evaluation, ilic2024evidence"></d-cite>. But if we want to know whether LLMs are actually becoming more capable, we first have to ask a  question: **what does it mean for an AI system to be "intelligent"?**

Despite its appeal, we discuss how *intelligence* is a vague and unstable concept, used differently across fields <d-cite key="howard1993intelligence, legg2007collection, wang2019defining, mitchell2024debates"></d-cite> and difficult to pin down. In empirical evaluation practice, the quest for intelligence is outsourced to benchmarks of intelligence-indicating tasks <d-cite key="horn1968organization, simon2024identifying"></d-cite> such as pattern abstraction <d-cite key="chollet2019measure"></d-cite>, reasoning <d-cite key="srivastava2023beyond, kazemi2025big"></d-cite>, or even general knowledge <d-cite key="phan2025humanity, hendrycks2020measuring, rein2024gpqa"></d-cite>. Yet these benchmarks frequently fall short of predicting the outcomes we actually care about, such as human preferences or real-world task performance (as we show in the next section). This raises a deeper question: are we evaluating the right thing at all?

We ask what commitments are implicitly bundled into the idea of evaluating models by their “intelligence.” We unpack these into three assumptions: **generality**, **stability**, and **realism**. We then ask which of these are actually needed for evaluation—and find that only **generality** is necessary. The other two, while often taken for granted, have little conceptual or empirical grounding.

Next, we ask whether generality itself provides a solid foundation. We show that it does: generality is conceptually stable (avoiding the ambiguity of intelligence) and it is formally grounded in multitask learning theory, which offers a principled way to measure reliable performance across tasks.

Ultimately, we ask the field to shift the core evaluative question. Instead of *“Is this model intelligent?”*, we propose a more concrete and actionable one: *"How general and reliable is this system across the tasks we care about?"* This reframing leads to a  more practical foundation for assessing progress in modern AI.

<!-- This theme supports rendering beautiful math in inline and display modes using [MathJax 3](https://www.mathjax.org/) engine.
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

## Why Intelligence is Problematic

The idea that some models are simply *more intelligent* than others has become central to how progress in language modeling is communicated <d-cite key="bubeck2023sparks, morris2023levels"></d-cite>. Benchmarks such as ARC <d-cite key="chollet2019measure"></d-cite>, Raven tests <d-cite key="abdelkarim2025evaluating"></d-cite> or the Blackbird Task <d-cite key="merlo-2023-blackbird"></d-cite> are often used to make such claims, implicitly treating benchmark performance as a proxy for general intelligence or capability. Yet, what these benchmarks actually measure is rarely interrogated, and our analysis suggests that the picture they paint might be incomplete and potentially misleading.

### Conceptual Instability

The discussion about general intelligence originates with the controversy between Spearman and Thomson <d-cite key="thomson1916"></d-cite>, but has since resurfaced within neuroscience <d-cite key="sims2013theory"></d-cite>,  cognitive science <d-cite key="sternberg1990metaphors, pfeifer2001understanding, sternberg2005cognition"></d-cite>, and  education <d-cite key="demetriou20051, ritchie2018much"></d-cite> — yet there is little consensus on what constitutes intelligence. Efforts to anchor intelligence in neuroscience have fallen short <d-cite key="mackintosh1986biology, sternberg2003biological, pietschnig2015meta, gignac2017brain"></d-cite>. Similarly, intelligence has long resisted a stable, unified definition in cognitive science. Intelligence may not be a unitary, well-specified capacity, but a family of loosely related, context-sensitive abilities, shaped by cultural, developmental, neural, and environmental factors <d-cite key="nisbett2001culture, sternberg2004culture, kan2013nature"></d-cite>.

In AI, we have largely inherited this ambiguity. The focus has been on defining intelligence in relation to the abilities of systems to perform certain tasks in environments in ways similar to humans <d-cite key="legg2007collection, poole2010artificial, russell2016artificial, kaplan2019siri, Bartneck2021"></d-cite>. Just as IQ Tests are incapable of indicating performance in significant cognitive capacities <d-cite key="schonemann1983iq, detterman1989correlations, stanovich2009intelligence, raven1989standard, gould1996mismeasure, henrich2010weirdest"></d-cite>, intelligence benchmarks have been argued to be 
problematic <d-cite key="bender2021dangers"></d-cite>.

### Illusion of Competence

Despite their widespread use, intelligence benchmarks often fail to reflect a model's effectiveness in real-world applications. We provide empirical evidence in Figure 1 and Figure 2 that strong performance on intelligence benchmarks exhibits limited correlation with human preference or task performance. As shown in Figure 1, the performance trends of different models for both ARC-AGI <d-cite key="chollet2019measure"></d-cite> (considered frontier intelligence benchmarks) and LMArena <d-cite key="chiang2024chatbot"></d-cite> differ significantly, i.e., a model that scores higher in ARC is not necessarily also better in LMArena (considered a human preference benchmark).

{% include figure.liquid path="assets/img/2026-04-27-measuregen/figure1.png" class="img-fluid" %}
<div class="caption">
    Figure 1: Performance comparison across ARC-AGI and LMArena benchmarks.
</div>

Also, as seen in Figure 2, these performance trends do not translate into universal capability and into reliable performance across the kinds of real-world tasks language models are often used for. This undermines the assumption that intelligence benchmark scores are good predictors of general-purpose competence.

{% include figure.liquid path="assets/img/2026-04-27-measuregen/figure2.png" class="img-fluid" %}
<div class="caption">
    Figure 2: Performance comparison across ARC-AGI and real-world task like OpenBookQA, Entity Extraction, and StackUnseen.
</div>

### Rethinking our Ground

The issues point to a deeper problem with how intelligence has been framed in AI evaluation. The continued reliance on intelligence-style benchmarks reflects an implicit belief that we are measuring something stable and meaningful, even as evidence suggests otherwise. Benchmarks are treated as if they reflect an underlying cognitive trait, and success on some fixed tasks is seen as progress toward intelligence (but never quite reaching intelligence). These patterns persist because the ambiguity surrounding intelligence allows these assumptions to remain unchallenged. What, then, are we actually assuming when we say a model is becoming more intelligent? The next section will detail assumptions through evaluation studies.


## Unpacking Intelligence

One of the earliest definitions of intelligence came from Marvin Minsky, a founding father of AI, and was as follows: *Artificial Intelligence is the science of making machines do things that would require intelligence if done by people* <d-cite key="bolter1984turing, fjelland2020general"></d-cite>.  However, this leaves open the question of what intelligence is, and how we best make machines do tasks that require it. In the literature, we have two divergent commitments.

- **i)** The first view emphasizes **Generality**. A system is intelligent because it is able to achieve a wide range of goals. Intelligence is demonstrated directly by the scope of performance.<d-footnote>Here, generality refers to the empirical manifestation of task performance across environments, not an abstract property of "generalization" in the learning-theoretic sense.</d-footnote>
- **ii)** The second view emphasizes **Realism**. A system is intelligent because it possesses some latent property that allows it to achieve a wide range of goals. Intelligence is not only observed but posited as an explanatory trait.

Once we take the realist route, however, further commitments follow. Realism, i.e., the idea that intelligence refers to a fixed, real property, implies that the capacities unlocked by intelligence are fixed. This means that if we can enumerate these capacities, we can design representative task suites and treat performance there as evidence, i.e, a system is intelligent if it does well on a fixed representative set of tasks that embody intelligence <d-cite key="chollet2019measureintelligence"></d-cite>. If not,  benchmarks only approximate intelligence, i.e, 
a system is intelligent by possessing intelligence <d-cite key="ilievski2025aligning"></d-cite>, and the benchmark performance of a system is only a direct measure of its intelligence.

Synthesizing these positions, we can see that the primary approaches to evaluating intelligence collapse onto three fundamental commitments:

| Assumption | Description |
|------------|-------------|
| **Generality** | We should develop an *all-purpose* or *multi-task* system. |
| **Stability** | A system should solve a *fixed* set of high-value intelligence indicating tasks. |
| **Realism** | We should focus on modelling *intelligence* itself. |

**Generality:** This assumption is also common in both the AGI literature <d-cite key="morris2023levels"></d-cite> and in practical NLP benchmarking initiatives, like Big Bench <d-cite key="srivastava2023beyond, kazemi2025big"></d-cite> or HELM <d-cite key="bommasani2023holistic"></d-cite>, that prioritize cross-domain competence on multiple language tasks. Even the instruction-tuning of models like T0 <d-cite key="sanh2022multitask"></d-cite>, FLAN-T5 <d-cite key="longpre2023flan"></d-cite>, or OPT-IML <d-cite key="iyer2022opt"></d-cite> all focus training on diverse prompts and task formulations with the explicit aim of cross-task generalization. These efforts reflect a growing recognition that narrow task performance is insufficient, especially as LLMs are increasingly expected to act as generalist agents. <d-cite key="hernandez2021general, moor2023foundation, zhang2024generalist"></d-cite>

**Stability:** Stability assumes that there exists a fixed set of tasks on which evaluation can reliably represent intelligence or capability. Recent research has often tried to identify the fixed set of such tasks <d-cite key="hendrycks2025definition"></d-cite> with explicit focus on few such tasks as reasoning <d-cite key="ilic2024evidence, 10.5555/3600270.3602070, kojima2022large, morishita2024enhancing"></d-cite> or planning <d-cite key="valmeekam2022large, valmeekam2023planning, valmeekam2023planbench"></d-cite> as important indicators of LLM performance.

**Realism:** Perhaps the most pervasive assumption is that benchmark success reflects a singular underlying cognitive trait called "intelligence". This can be seen in general tests of intelligence <d-cite key="chollet2019measure, phan2025humanity, cai2025mm"></d-cite> or IQ-style comparisons. <d-cite key="pellert2024ai, huang2024measuring, abdelkarim2025evaluating"></d-cite>

Next, we show why generality is independent and also necessary and sufficient for evaluating models.


### Generality is Independent

We want to argue that accepting **Generality** should not automatically lead us also to accept **Stability** and **Realism**.

**Formal setup.** Let $ T $ be a (possibly infinite) set of tasks endowed with a probability measure $ Q $ (the *task environment*). Each model $ M $ induces a measurable performance function

$$
f_M : \mathcal{T} \to [0,1], \qquad t \mapsto f_M(t),
$$

where $ f_M(t) $ denotes the normalized performance of model $ M $ on task $ t $. We want to define what "evaluating M" means under the three alternative assumptions.

**Definition (Generality).** The *generality* of a model is its expected performance across the task environment:

$$
E_G(M) = \mathbb{E}_{t \sim Q}\big[ f_M(t) \big].
$$

It assumes no fixed task set or latent variable, only performance averaged over the environment $ Q $.

**Definition (Stability).** The *stability* of a model is its aggregated performance on a fixed benchmark subset $ S \subset \mathcal{T} $:

$$
E_S(M) = F\big( ( f_M(t) )_{t \in S} \big),
$$

where $ F $ is a predetermined aggregation functional. It assumes that the same benchmark tasks remain representative.

**Definition (Realism).** The *realism* assumption posits a latent cognitive representation $ I(M) \in \mathbb{R}^k $ and task-specific decoding functions $ g_t : \mathbb{R}^k \to [0,1] $, such that performance derives from this shared latent space:

$$
E_R(M) = \mathbb{E}_{t \sim Q}\big[ g_t(I(M)) \big].
$$

It assumes that observable task success reflects an underlying property, interpreted as "intelligence". Now consider the following thought experiment!

{% include figure.liquid path="assets/img/2026-04-27-measuregen/genthought.jpg" class="img-fluid" %}
<div class="caption">
    Thought experiment
</div>

This thought experiment illustrates a key insight: **generality can be pursued without assuming either stability or realism**. One can seek broad capability without committing to a fixed task set or a unified latent construct. 


<!-- Citations are then used in the article body with the `<d-cite>` tag.
The key attribute is a reference to the id provided in the bibliography.
The key attribute can take multiple ids, separated by commas.

The citation is presented inline like this: <d-cite key="gregor2015draw"></d-cite> (a number that displays more information on hover).
If you have an appendix, a bibliography is automatically created and populated in it.

Distill chose a numerical inline citation style to improve readability of citation dense articles and because many of the benefits of longer citations are obviated by displaying more information on hover.
However, we consider it good style to mention author last names if you discuss something at length and it fits into the flow well — the authors are human and it’s nice for them to have the community associate them with their work. -->

---

### (Only) Generality is Necessary

The necessity of generality becomes clear when we ask what evaluation aims to uncover. Following Hernández-Orallo et al. (2021) <d-cite key="hernandez2021general"></d-cite>, evaluation can be expressed as a mapping from task to expected success, yielding an *agent-characteristic curve* (ACC). Each model $ M $ induces a performance function $ f_M : \mathcal{T} \to [0,1] $. Let $ h(t) $ denote the *difficulty* of task $ t $. The ACC captures how success declines as difficulty increases:

$$
\psi_M(h) = \mathbb{E}_{t\sim Q \,|\, h(t)=h}\big[f_M(t)\big]
$$


**Why evaluation requires generality.** If two systems $ M_1 $ and $ M_2 $ yield the same mean performance but exhibit different $ \psi(h) $ curves, they cannot be regarded as equivalent: one may fail on trivial tasks while the other fails only on hard ones. A meaningful evaluation must depend on the *shape* of $ \psi_M(h) $, not merely its average.


We define the *spread* of the ACC as:

$$
S_M^2 = \int_0^\infty (h - \bar{h}_M)^2 \psi_M(h)\,dh
$$

A small $ S_M $ means performance is concentrated before a clean decline—predictable and *general* behavior. A large $ S_M $ indicates scattered or specialized success. The reciprocal quantity

$$
\Gamma_M = \frac{1}{S_M}
$$

is thus a direct measure of generality.<d-footnote>This formulation follows the generality-spread relation introduced by Hernández-Orallo et al. (2021).</d-footnote>

**Why only generality.** Without accounting for $ \Gamma_M $, evaluation collapses into an unanchored average that changes arbitrarily with task sampling. Once tasks are organized along a difficulty axis, the measure of how tightly performance is distributed becomes the only element ensuring comparability across environments. Therefore, *generality is the necessary condition for evaluation to be coherent and transferable*. Neither *Stability* nor *Realism* is required: fixing a task subset removes the difficulty structure, and positing a latent "intelligence" adds nothing to the observable shape of $ \psi_M(h) $.

<!-- Just wrap the text you would like to show up in a footnote in a `<d-footnote>` tag.
The number of the footnote will be automatically generated.<d-footnote>This will become a hoverable footnote.</d-footnote> -->

---

## What Generality Offers

We identify two factors here: generality is conceptually stable and it is theoretically grounded in multitask learning.

**Conceptual Stability:** Generality offers a conceptually stable and empirically grounded alternative i.e one that aligns more directly with how models are used and deployed. Prior work in cognitive science and AI emphasizes generalization as the hallmark of intelligent behavior <d-cite key="lake2017building, tenenbaum2001generalization, yu2020meta, tomov2021multi, ilievski2025aligning"></d-cite>. Moreover, generality is flexible to task drift and evolving use cases. Unlike static benchmarks, which quickly lose relevance as models saturate their task sets or learn test-specific heuristics, generality-based evaluation can accommodate new tasks as they emerge. It requires only that we specify a diverse and representative sample of tasks at evaluation time - not that we define a canonical set of "core" challenges in advance. In short, generality is not only the most practically relevant evaluation principle, but it is also the most conceptually resilient.

**Theoretical Grounding.** Generality is supported by learning theory and empirical evidence—especially multitask learning (MTL) <d-cite key="caruana1993multitask,caruana1997multitask"></d-cite> and "learning to learn" in humans <d-cite key="thrun1998learning,ilievski2025aligning"></d-cite>. 

**Theorem:** Consider an environment $\mathcal{E}$ consisting of a distribution $Q$ over tasks, where each task $P \sim Q$ is a distribution over data in a learning problem. Let $\mathcal{H}$ be a hypothesis class, $L_P(h)$ be the loss on task $P$, and $L_Q(h)$ be the model's environment average error. Then, for any $\delta>0$ (where $\delta$ is the confidence parameter), with probability at least $1-\delta$, the generalization bound is reduced by approximately a factor of $\sqrt{n}$ in the multi-task case.

**Proof:** We proceed in three steps: 

**Step 1: Generalization Bound for Single-Task Environment (STE)** Let $\mathbb{E}$ be an environment consisting of a distribution $Q$ over tasks. For each task $P \sim Q$, let true loss of $h \in \mathcal{H}$ be given by $L_P(h) = \mathbb{E}_{(x,y)\sim P}[\ell(h(x),y)]$ where $\ell$ is a loss function. The empirical loss computed from $m$ i.i.d. samples drawn from $P$: 

$$
\hat{L}_P(h) = \frac{1}{m} \sum_{i=1}^{m} \ell\bigl(h(x_i), y_i\bigr)
$$

and the environment-average loss is: $L_Q(h) = \mathbb{E}_{P\sim Q}[L_P(h)].$ By standard PAC-learning results (see <d-cite key="baxter2000model"></d-cite>), with probability at least $1-\delta$:

$$
\sup_{h\in \mathcal{H}} |L_P(h) - \hat{L}_P(h)| = O\Bigg(\sqrt{\frac{C + \ln(1/\delta)}{m}}\Bigg).
$$

**Step 2: Generalization Bound for Multi-Task Environment (MTE)** Here we evaluate $h$ on $n$ tasks $P_1,\dots,P_n \sim Q$, each with $m$ samples, yielding the average empirical error as an estimate of $L_Q(h)$: $\frac{1}{n}\sum_{i=1}^n \hat{L}_{P_i}(h)$. 

There are two sources of generalization error:

1. *Within-task generalization:* By the same PAC bound as in STE, for each $P_i$, 
   $$
   \sup_{h\in \mathcal{H}} |L_{P_i}(h) - \hat{L}_{P_i}(h)| = O\Bigg(\sqrt{\frac{C + \ln(n/\delta)}{m}}\Bigg).
   $$

2. *Across-task generalization:* Since tasks are drawn i.i.d. from $Q$, by Hoeffding's inequality:
   $$
   \sup_{h\in\mathcal{H}} |L_Q(h) - M(h)| = O\Bigg(\sqrt{\frac{C + \ln(1/\delta)}{n}}\Bigg),
   $$
   where $M(h) = \frac{1}{n} \sum_{i=1}^{n} L_{P_i}(h)$ is the average true error across task.

Combining these bounds via the triangle inequality,

$$
\sup_{h\in\mathcal{H}} \Big|L_Q(h) - \frac{1}{n} \sum_{i=1}^n \hat{L}_{P_i}(h)\Big| \leq O\Bigg(\sqrt{\frac{C + \ln(n/\delta)}{m}} + \sqrt{\frac{C + \ln(1/\delta)}{n}}\Bigg).
$$

**Step 3: Comparison of Bounds.** 

We derive the STE bound:

$$
L_Q(h) \leq \hat{L}_{P}(h) + O\Bigg(\sqrt{\frac{C + \ln(1/\delta)}{m}}\Bigg),
$$

We derive the MTE bound as:

$$
L_Q(h) \leq \frac{1}{n}\sum_{i=1}^n \hat{L}_{P_i}(h) + O\Bigg(\sqrt{\frac{C + \ln(1/\delta)}{n m}}\Bigg).
$$


Thus, we see that in a learning environment where tasks are drawn i.i.d. from a distribution Q, the single–task generalization bound decays at a rate inversely proportional to $m$ (the number of samples in the task) while for multi-task environments, the error decays much faster at a rate of $1/\sqrt{mn}$ where $n$ is the number of tasks evaluated in our environment. This $\sqrt{n}$ reduction results from combining within-task PAC bounds with an across-task concentration (via Hoeffding’s inequality), thereby demonstrating that multi–task evaluation (or learning) effectively reduces the estimation variance. 

<!-- **Theoretical Grounding:** Generality is supported by learning theory and empirical evidence—especially multitask learning (MTL) <d-cite key="caruana1993multitask,caruana1997multitask"></d-cite> and "learning to learn" in humans <d-cite key="thrun1998learning,ilievski2025aligning"></d-cite>. When tasks are sampled from an environment distribution $Q$, evaluating a hypothesis $h$ across diverse tasks yields tighter, more predictive estimates than any single-task score.

Define the empirical loss on a task $P$ with $m$ samples:
$$
\hat{L}_P(h) = \frac{1}{m} \sum_{i=1}^{m} \ell\big(h(x_i), y_i\big),\qquad L_Q(h)=\mathbb{E}_{P\sim Q}[L_P(h)].
$$

Single-task generalization concentrates as
$$
\sup_{h\in\mathcal H}\big|L_P(h)-\hat L_P(h)\big|=O\!\left(\sqrt{\frac{C+\ln(1/\delta)}{m}}\right).
$$

Averaging over $n$ independent tasks $P_1,\dots,P_n\sim Q$ with $m$ samples each gives the estimator
$$
\hat M(h)=\frac{1}{n}\sum_{i=1}^n \hat L_{P_i}(h),
$$
which satisfies the tighter environment bound
$$
\sup_{h\in\mathcal H}\Big|L_Q(h)-\hat M(h)\Big|=O\!\left(\sqrt{\frac{C+\ln(1/\delta)}{nm}}\right).
$$

In words: averaging across $n$ diverse tasks reduces estimation error by about $\sqrt{n}$ compared to a single task. This formalizes why generality-based evaluation—measuring performance across multiple, varied tasks—provides a more stable and forward-looking assessment of capability. -->

## Conclusion

In this work, we have put forth the perspective that model evaluation should be grounded in *generality*—the breadth and consistency of performance across tasks—rather than in abstract notions of *intelligence*. Unlike intelligence, which rests on unstable conceptual and empirical foundations, generality offers a measurable, theoretically grounded, and operationally meaningful principle. It captures what truly matters for deployment: how reliably a system performs when tasks vary or evolve.

By showing that generality is both independent from, and sufficient for, coherent evaluation, we provide a framework that unifies conceptual clarity with formal rigor. This reframing is increasingly necessary as language models are applied in open and shifting environments, where success cannot be defined by static benchmarks <d-cite key="kiela2021dynabench,hofmann2025fluid,kim2025benchmark"></d-cite> or latent cognitive claims <d-cite key="gignac2015raven,blili2025stop"></d-cite>. Future progress in AI should therefore be assessed not by how "intelligent'' a model appears, but by how *generally and dependably* it performs across the diverse tasks we ask of it.


---
