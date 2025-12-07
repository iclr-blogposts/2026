---
layout: distill
title: The Information Shadow - Why There Are Things LLMs Can Never, Ever Know

description: This blogpost introduces the idea of the information shadow - the region of reality that large language models can never fully know, no matter their scale or training data. We argue that some truths are fundamentally inaccessible because they cannot be expressed in language, cannot be inferred from observable data, or cannot be reached by gradient-based training dynamics. Through thought experiments and concrete probes, the post shows how representational limits, identifiability barriers, and optimization constraints create predictable blind spots in model behavior. Understanding this shadow is essential for building safer systems, designing better benchmarks, and avoiding misplaced confidence in model outputs. The goal is not to diminish LLMs, but to map the structural edges of their intelligence.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous

authors:
  - name: Anonymous
    url: ""
    affiliations:
      name: ICLR 2026 Submission
  
# must be the exact same name as your blogpost
bibliography: 2026-04-27-the-information-shadow.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: The opening riddle
  - name: Three kinds of impossibility — but not the obvious ones
  - name: Why these shadows matter — three surprises
  - name: How to see the shadow - three operational probes (doable experiments)
  - name: A sharper taxonomy (and a few wild conjectures)
  - name: Practical consequences (for researchers and product teams)
  - name: The research program — five crisp projects
  - name: Closing - an ethic of humility

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

## The opening riddle

Ask an LLM for a limerick, and it will gladly provide one. Ask it for the exact path of every raindrop from a storm ten years ago, and it will not be able to help. This isn’t because it is “stupid,” but because the question goes beyond its knowledge limits.
However, there are weirder blind spots than these obvious ones. Some limitations won’t disappear with more data, computing power, or smarter algorithms <d-cite key="raghu2017expressive"></d-cite>. They are structural; they exist in the space between what can be described, what can be inferred, and what can be found through the training process.
We can call that boundary the information shadow.

This post makes a bold claim: 
```
There are entire kinds of truths that a text-based learner cannot grasp under any practical training conditions. These truths show systematic, measurable signs.
```

For a glimpse of the strange: think about truths that can be represented and exist in the data, but are unreachable by gradient descent because the method never finds the area that contains them. That’s not a flaw; it’s a different type of knowledge limit.

## Three kinds of impossibility — but not the obvious ones 

Most people would agree with statements like “you can’t know private experiences” or “you can’t predict coin tosses.” Those are boring. The interesting impossibilities are subtler and can be acted on. I divide them into three less obvious categories, each illustrated with a thought experiment.

### A. Inexpressible structures — the limits of what language can convey 

Thought experiment: Imagine a civilization communicates a natural law not through statements but as a complete change in perception. This pattern only exists in multiple sensory experiences and cannot be expressed succinctly in symbols <d-cite key="tishby2015deep"></d-cite>. Even with infinite text about that civilization, an LLM cannot recreate the law because it does not have a concise representation in written form <d-cite key="beck2020deep"></d-cite>.

**Key point:** The impossibility of representation goes beyond just personal experiences or needing infinite details; it’s about structure. If the way a phenomenon is generated cannot be neatly described in symbols, language loses the pattern.

### B. Non-identifiable functions — rules hidden in the data 

Thought experiment: Two different processes produce identical distributions of all observable texts. One uses a simple rule R, while the other relies on a look-up table based on hidden variables not present in the text. These identical datasets cannot distinguish between them. No amount of data reveals the underlying rule <d-cite key="xu2012identifiability"></d-cite>
<d-cite key="valiant1984theory"></d-cite>.

**Key point:** Similarity in statistics prevents learning. When several hidden mechanisms lead to the same text distribution, inference cannot determine which mechanism created the data. LLMs can fit either but can’t tell which one is correct.

### C. Optimization shadows — knowledge that can be represented but is unreachable 

Thought experiment: There exists a function f that, if included in the model, would make it act more “logically.” However, f’s area lies through a narrow, high-loss region in parameter space, and SGD skips over this region. The design of the model and its training process keep this area hidden. The function can be represented (a network can compute it) but cannot be reached in practice.

**Key point:** The space of functions that can be represented does not equal the space of functions that can be reached through typical training methods. SGD, the initial setup, and dataset biases create pathways with dead ends.

## Why these shadows matter — three surprises 

These are not just academic concepts. The shadows help explain issues that practitioners often encounter:

- **Confident fabrications that persist.** Some illusions remain despite more data because the “better” representation lies in an unreachable area.
- **Misleading strength of benchmarks.** A model may pass a test not because it has learned the rule, but because the test aligns with an easier, reachable shortcut <d-cite key="zhou2023datasetblindspots"></d-cite>.
- **The illusion of knowing everything.** While scaling reduces some shadows, new, qualitatively different shadows appear as models adopt increasingly complex but fragile versions of real knowledge.

So, the shadows shift and divide as models grow. The edge of knowledge moves, but it never disappears.

## How to see the shadow: three practical tests 

I promise this will be useful. Here are three tests you can apply to examine different aspects of these shadows.

### Probe 1 — Counterfactual Distinction Test (CDT)

Goal: find non-identifiable functions.

Procedure: Create two synthetic processes that are the same based on training data but differ under a counterfactual situation (for example, a hidden key alters outputs in unknown contexts). Train separate models on identical datasets. Then test the models on the counterfactual context: do their outputs differ? If they do, the training process shows a specific bias; if they do not, the task is non-identifiable. 

Why it reveals a shadow: if you cannot get the model to favor the “true” mechanism, then the function cannot be learned from plain text.

### Probe 2 — Basin Escape Mapping (BEM)

Goal: find unreachable but representable areas <d-cite key="du2019gradient"></d-cite>.

Procedure: Identify a target function f that you know can be implemented (for instance, a small hand-crafted circuit within a larger architecture). Use targeted, strong interventions (like evolutionary search or synthetic curricula) to try to reach the parameters that realize f, starting from standard initial setups. Measure how often the optimization attempts achieve f. If standard SGD rarely finds it, but targeted searches do, then f is in an optimization shadow <d-cite key="du2019gradient"></d-cite> <d-cite key="anonymous2023sgdtraps"></d-cite>.

Why useful: it distinguishes representation ability from optimization access.

### Probe 3 — Language Compression Residuals (LCR)

Goal: find genuinely inexpressible structures <d-cite key="beck2020deep"></d-cite>.

Procedure: Check whether adding non-textual elements (like images or human annotations) reduces predictive confusion more than the size of the text can. If there’s hidden predictive power from non-linguistic sources even when large text datasets are available, that shows inexpressible content.

Why neat: it makes the “embodied” part of the shadow actionable.

## A clearer classification (and some wild guesses) 

The three shadows give us a refined classification of “unknowns” for LLMs:

- **Type I** — Linguistic ceiling: phenomena that cannot be represented perfectly in language.
- **Type II** — Identifiability ceiling: phenomena that produce the same observable patterns under different mechanisms.
- **Type III** — Dynamical ceiling: phenomena that can be represented but are not dynamically accessible.

Wild conjectures (stated openly as conjectures):

- **Conjecture 1 (Persistence):** For any scale S and type A of architecture, there are Type II functions that stay non-identifiable at scale S but can become identifiable by introducing special signals (like sensors).

- **Conjecture 2 (Dynamical bottleneck):** There are representable cognitive functions whose areas are extremely rare under SGD, so just increasing the model size or dataset size won’t help to find them easily <d-cite key="bottou2018optimization"></d-cite>.

- **Conjecture 3 (Shadow tradeoff):** Trying to reduce Type I shadows (like through using multiple modalities) can increase Type III shadows (optimization gets tougher because richer representations create tighter areas).

These guesses are intriguing because they suggest distinct research strategies: add sensors, modify training processes, or change the learning challenges.

## Practical implications (for researchers and product teams)

If you accept the existence of shadows, certain practices should change:

- **Benchmark design:** include counterfactuals and adversarial generative methods to uncover limits of identifiability.

- **Model auditing:** assess access to areas through basin-hopping tests before trusting a learned ability.

- **System design:** don’t expect models that rely only on language to make decisions backed by private or physical evidence — connect those processes to sensors or human input.

- **Safety & alignment:** assess not just “can the model answer X,” but also “can the model know if it can answer X?” (this is about calibrating awareness of what is unknown). 

## The research agenda — five clear projects 

If you want to turn these ideas into papers, here are some immediate projects:

- **Synthetic identifiability benchmark suite —** pairs of generative models that are observationally identical in training but diverge on controlled counterfactuals.

- **Basin accessibility atlas —** a catalog of target functions and measured discovery rates under SGD, evolutionary search, and training methods.

- **Multimodal residual maps —** measure how much additional predictive power other modalities provide compared to text across various tasks.

- **Shadow-aware alignment protocols —** methods that encourage models to express uncertainty related to their shadow presence.

- **Optimization-aware architecture search —** identify architectures that minimize Type III shadows for specific cognitive functions.

## Closing: an ethic of humility 

The big temptation in modern ML is to believe that we can uncover any truth if we just scale up and optimize effectively. The information shadow tells us that some truths are not only difficult but are obscured by the nature of our data and how we learn.

This should make us more humble. Not pessimistic, but strategic: to push back the shadow, we need to change the approach — add sensors, redesign objectives, and create training methods that explore the unknown areas <d-cite key="raghu2017expressive"></d-cite>.

Finally, here’s a simple guideline for product teams and reviewers:
**If your model confidently answers about an area that might be affected by Type I, II, or III shadows, treat the answer as a hypothesis, not a certainty.**

The shadow isn’t a barrier; it’s a map. Let’s start exploring it.
