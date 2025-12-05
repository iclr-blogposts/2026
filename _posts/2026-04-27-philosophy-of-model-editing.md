---
layout: distill
title: A Philosophy of Model Editing - What Does It Mean to “Change Knowledge” in a Neural Network?

description: This blogpost explores what it truly means to change knowledge inside a neural network. Unlike symbolic systems, large language models do not store facts in explicit locations; they implement them through distributed geometric transformations. Editing a model therefore reshapes regions of its activation space, alters relational structures, and sometimes shifts broader behavioral tendencies. We examine how local edits differ from global ones, why forgetting resembles suppression rather than deletion, and how repeated modifications can change a model’s identity. By framing model editing as a philosophical and structural question rather than a purely technical procedure, this piece highlights the need to evaluate edits not only for local correctness but also for their impact on coherence, ontology, and long-term behavior.
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

# Authors
authors:
  - name: Anonymous
    url: ""
    affiliations:
      name: ICLR 2026 Submission
  
# must be the exact same name as your blogpost
bibliography: 2026-04-27-philosophy-of-model-editing.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: What Is “Knowledge” Inside a Neural Network?
  - name: What Actually Changes During a Model Edit?
      subsections:
      - name: Local Micro-Manifold Rewrites
      - name: Distributed Relational Shifts
  - name: Does Editing Introduce Inconsistency?
  - name: Can a Neural Network Ever Truly “Forget”?
  - name: The Ontology of Edited Models - Does Editing  Change Identity?
  - name: Why This Matters - Beyond Bug-Fixes
      subsections:
      - name: Safety
      - name: Alignment
      - name: Interpretability
      - name: Delegated Autonomy
      - name: Scientific Understanding
  - name: A Research-Aware Conceptual Taxonomy of Model Edits
  - name: Conclusion — Editing Knowledge ≠ Editing Memory

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

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## Introduction

Large language models can now be adjusted. They can be corrected after deployment, fixed for safety, or guided towards new behaviors without needing complete retraining. This raises a fundamental question that spans machine learning, knowledge theory, and philosophy:

What does it mean to change knowledge within a neural network?

Traditional software updates a record in a database. Humans change beliefs through reasoning, feelings, and contradictions. Neural networks do neither.

They do not store symbols, explicit beliefs, or lookup tables. Instead, knowledge is spread out, intertwined, and geometric.

So when we edit a model : 

- Are we rewriting memory? 

- Are we distorting the shape of meaning itself? 

- Is a fact a localized change or a global guideline? 

- Can a neural network truly forget?

- And if we reshape enough knowledge, does the model’s identity shift?

This blog post provides a framework for these questions. It is based on the technical features of neural networks but also invites philosophical reflection.


## What Is “Knowledge” Inside a Neural Network?

Before exploring how knowledge changes, we need to understand what it is.

In symbolic systems, knowledge exists as:

- entries
- axioms
- pointers
- definitions

In a neural network, there is no specific spot for “The Eiffel Tower is in Paris.” Instead, knowledge arises from a transformation represented in <d-cite key="bengio2013representation"></d-cite>:

- embedding geometry
- attention pathways
- MLP activations 
- token transition statistics

A useful summary:
```
A fact is not stored; it is enacted. A network knows something because its transformations consistently bring it about.
```

To illustrate this, consider a straightforward example <d-cite key="geva2021transformerfeedforward"></d-cite>
:

```
When a model states, “Rome is the capital of Italy,” the fact is not retrieved; it emerges from a path through activation space that consistently lands in the “Rome” area.
```

Thus:

- A fact is a stable attractor. 
- A belief is a directional bias in that space. 
- A concept is a group of reachable states.

This means editing is essentially a geometric process.


## What Actually Changes During a Model Edit?

Editing a model—whether through fine-tuning, ROME-style updates, MEMIT, soft prompts, or manual weight adjustments—changes the underlying geometry of these attractors.

Two main types of changes typically occur:

### Local Micro-Manifold Rewrites

Imagine a micro-manifold: a small area of activation space where similar prompts converge. For example, questions like:

- “What is the capital of Italy?” 
- “Italy’s capital city is…” 
- “The city that serves as Italy’s seat of government is…”

All fall into a similar neighborhood. 

A local edit only alters this specific area:

- the boundary shifts, 
- an attractor nudges, 
- nearby semantic neighbors reorganize slightly. 

Methods like ROME and MEMIT aim to work here <d-cite key="meng2022rome"></d-cite> <d-cite key="meng2023memit"></d-cite>, making precise and minimally invasive modifications.

### Distributed Relational Shifts

Some relationships are globally structured. 

Editing a relation like (Italy, capital, X) can have broader impacts:

- “Italian government is located in…” 
- “Italian culture in ___ city” 
- analogies like “France ↔ Paris :: Italy ↔ ___”

This constitutes a relational edit, rather than a local one. You reshape an entire conceptual subspace.

This is similar to full relation editing or changing an embedding direction like “country→capital.”

In summary:

```
Local rewrites fix a pocket of meaning. 
Distributed edits alter the semantic landscape.
```

Both are important and can create contradictions.



## Does Editing Introduce Inconsistency?

Human beliefs can be modular and inconsistent, while neural networks are more globally connected.

This creates a challenge:

- Overly local edits cause the model to revert to the old fact under rephrasing. 
- Overly global edits distort unrelated knowledge (identity drift).

For example:

- Correcting a wrong fact might unintentionally alter analogy patterns.
- Changing a persona (like making the assistant sarcastic) could affect unrelated topics.

A helpful distinction:

- Epistemic content refers to what the model believes (facts). 
- Ontological structure refers to the types of concepts and relationships the model recognizes.

A good edit should change epistemic content without harming ontology.

In simpler terms:
```
Edit the belief, not the overall worldview.
```

Current methods often struggle to maintain this balance.





## Can a Neural Network Ever Truly “Forget”?

For humans, forgetting can take various forms:

- deletion
- suppression
- reconsolidation
- interference

In neural networks, forgetting is a strange idea. 

There is no slot to erase or pointer to zero out. Knowledge is redundantly encoded <d-cite key="frankle2019lottery"></d-cite> across many layers and modules.

To forget the fact:
“The Eiffel Tower is in Paris,” 
you would need to disrupt all the attractor pathways leading to “Paris.”

But because:

- representations are redundant, 
- associations are intertwined, 
- learned structures are densely distributed, 

a single update rarely wipes out all paths. 

This explains why edited facts often come back <d-cite key="goodfellow2013empirical"></d-cite>:

- under rephrasing, 
- in lengthy contexts, 
- through analogy paths, 
- under tricky prompts.

So:
```
Forgetting in neural networks isn’t about destruction; it’s about reducing the chance of a belief coming back.
```
Philosophically, this is more like suppression than deletion.

## The Ontology of Edited Models: Does Editing Change Identity?

Model identity isn’t defined by the dataset but by:

- behavior
- coherence
- decision tendencies
- inductive biases

Editing can change these.

Small edits (like a single factual relationship) usually do not alter identity. However, large or repeated edits such as changing tone, worldview, or moral views can create a distinctly different agent.

Example:

- turning a friendly assistant into a sarcastic persona, 
- shifting political leanings <d-cite key="olah2020circuits"></d-cite>, 
- imposing a strict safety rule that leads to reasoning changes.

This relates to the Ship of Theseus problem:
```
After many edits, is it still the same model?
```
From an engineering perspective:

- the model is “the same” if its functional behavior remains stable
- “different” if the relational structure or persona changes.

From a philosophical perspective:

- identity ties to the model’s internal ontology. 
- Changing high-level relational geometry equals changing identity.

## Why This Matters: Beyond Bug-Fixes

A theory of editing matters because it relates to:

### Safety

Local edits may unintentionally create global contradictions.

### Alignment

Changing epistemic content could distort ontological foundations, altering what the model values.

### Interpretability

Editing enables us to examine the model’s structure: we find out which relationships are fragile or deep.

### Delegated Autonomy

If a model is part of a workflow or agent system, shifts in identity can undermine trust.

### Scientific Understanding

Editing provides insight into how neural networks represent meaning and change knowledge.

## A Research-Aware Conceptual Taxonomy of Model Edits

Here is an improved taxonomy, now enhanced with examples and basic connections to known methods:

**Type-1. Local Semantic Rewrites**

Small, targeted changes to micro-manifolds. This roughly corresponds to methods like ROME or MEMIT that aim for precise, fact-level edits.

**Type-2. Distributed Relational Shifts**

Edits that affect entire relational subspaces (for example, altering all country→capital pairs). These can be seen in behavior changes after broader fine-tuning or multi-example edits.

**Type-3. Ontological Reorientations**

Changes that affect how the model organizes concepts (for instance, safety-alignment fine-tunes that reshape moral or intentional concepts). These often occur as a side effect of broad, domain-specific training.

**Type-4. Identity-Level Modifications**

Edits that change persona, reasoning style, or core behavioral tendencies. Example: turning a neutral assistant into one that is consistently sarcastic. This is typical in instruction tuning, reinforcement learning from human feedback, or safety training.

This taxonomy is conceptual, but it connects to real methods <d-cite key="hartford2024ke-survey"></d-cite>, grounding philosophy in current practice.

## Conclusion — Editing Knowledge ≠ Editing Memory

We circle back to the question:

What does it mean to change knowledge in a neural network?

The refined answer:

- Facts are not stored; they are acted upon as transformations.
- Editing involves changing geometry, not memory.
- Forgetting is about reducing probability, not deleting.
- Local changes can result in global contradictions.
- Edits can shift not only beliefs but also identity.
- Maintaining ontological stability is just as important as epistemic correctness.

A practical implication is:
```
Evaluating model edits should assess not only factual accuracy but also identity shifts and ontological changes.
```
Ultimately:
```
Model editing is not just a technical tool. It is a philosophical act involving decisions about how an artificial mind should evolve.
```