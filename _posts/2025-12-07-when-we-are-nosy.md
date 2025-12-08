---
layout: distill
title: When We are Nosy
description: Machine learning systems are defined for many people, and for the design of in particular language models, calls for “social choice–based’’ methods are increasing. This seems to run counter to the practice in machine learning to “personalize’’ models. This blogpost clarifies when personalization and when social choice has its place, using the Impossibility of a Paretian Liberal by Amartya Sen.
date: 2025-12-07
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2025-12-07-when-we-are-nosy.bib

toc:
  - name: 1. Disagreement All the Way Down
  - name: 2. Personalization as the Default Story
  - name: "3. Enter Social Choice – When Your Preferences Care About Mine"
  - name: 4. The Impossibility of a Paretian Liberal, AI version
  - name: 5. When Liberalism, when Social Choice?
  - name: 6. A Call to Action
---

## 1. Disagreement All the Way Down

Imagine three colleagues sharing the same large language model.

 - Ada wants the model to show her every spicy take and half baked hypothesis it can find.
 - Ben wants only peer reviewed, citation laden answers with confidence intervals.
 - Chris not only wants a very “safe’’ model for themselves, but also insists that no one in the lab should be able to prompt the model into generating certain topics at all.

All three log into the same interface. It is the same base model, same weights, same provider, different users.

Ada: “Please do not nerf my model just because Chris does not like memes about cosmology.’’

Chris: “I do not care what you want. I do not want this system in our lab producing that stuff for anyone.’’

Ben: “…can I just get a proof assistant?’’

If you only listen to Ada and Ben, the story sounds simple:

> Alignment is "just personalization." Give each user their own fine tuned slice of behavior.

If you only listen to Chris, a different story appears:

> Alignment is "shared standards." We need institutional policies about what models are allowed to do for anyone.

These are both recognizably alignment stories:

- **Personalization:** fit models to per-user preferences inferred from their data and feedback.
- **Social choice:** treat alignment as an aggregation problem over many people's preferences, mediated by institutions <d-cite key="ge2024axiomsaialignmenthuman"></d-cite><d-cite key="sorensen2024roadmappluralisticalignment"></d-cite>.

The tension is not new. Economists, philosophers, and political theorists have been poking it for decades. But in ML, we reinvent it every time someone says:

> "Let us just give everyone their own model; then there is no conflict."

This post tries to make that intuition precise and then break it, in a controlled way.

The central question:

> **When can alignment be "just personalization," and when is it inherently a social choice problem?**

We will get there via a detour through , a result that sounds like political philosophy but is uncomfortably relevant for recommender systems, RLHF, and foundation model design.

## 2. Personalization as the Default Story

Let us start with the story that ML people naturally tell. In a recommender or chatbot setting, personalization usually means:

- There is a shared model architecture and training pipeline.
- For each user $i$, the system maintains some representation $\theta_i$ (parameters, adapters, preference vectors, etc.).
- The system optimizes an objective that depends on user specific data:

$$\theta_i^\star = \arg\max_{\theta} \mathbb{E}_{(x,r)\sim D_i}[u_i(r, f_\theta(x))],$$

where $D_i$ is the logged interaction data, $r$ is reward or feedback, and $u_i$ is some proxy utility (clicks, watch time, RLHF scores).

In the limit of abundant data and expressive models, you get the slogan:

> "Each user gets their model."

This is desirable for several reasons.

**Autonomy.** Each user has preferences $\succ_i$ over outputs. If the system is "aligned" with user $i$, it produces outputs in their local top set.

**Epistemic humility.** Providers generally do not know what is best for users, so they infer it from behavior. Clicks, preferences, and comparisons steer model behavior.

**Conflict avoidance.** If every user gets their own slice, we avoid fights over a single global value system.

**Example: Personalized news vs. a global editorial line**

At one extreme: a global front page with one editorial policy.
At the other: a fully personalized newsfeed.

If what I read does not affect you, personalization seems like a Pareto improvement:

- more relevance,
- less complaining about "bias,"
- outcomes tailored to each user.

But that "if" matters a lot.

## 3. When We are Nosy

The neat personalization story breaks the moment we admit the obvious: people often care about what other people see, do, or get from a system. Instead of aligning a system to a single user's preferences, we now have:

- many users,
- who may be nosy about each other's experience.

This is where calls for "social choice–based alignment" come from.

### 3.1 What it means to be nosy

Suppose the system allocates each user $i$ a personalized output $x_i$, so a social outcome is

$$x = (x_1, x_2, \dots, x_n) \in X_1 \times \cdots \times X_n.$$

**Definition (Nosy preferences).**
User $i$ has nosy preferences if there exist $x, y$ such that:

- $x_i = y_i$,
- yet $x \succ_i y$ or $y \succ_i x$.

That is, user $i$ cares about what happens to others, even when nothing changes for themselves.

### 3.2 Examples

Nosiness is common:

1. “This content should not exist for anyone.” Chris refuses to see misinformation—and also prefers that Ada never sees it.
2. Externalities of attention. Ada enjoys spicy political takes. Ben prefers that no one in the lab sees them because he thinks they damage productivity.
3. Harmlessness/Personalization within Bounds. A safety officer wants a model that never outputs step-by-step harmful instructions, not even to consenting experts.
4. Paternalism. A parent cares about what their child and the children in her preschool clas see.
5. Epistemic environment concerns. Researchers may say: “Even if I can handle low-quality medical advice, I don’t want the model producing it for society.”

These reveal a key point: nosy preferences disrupt the logic of pure personalization.

---

## 4. The Impossibility of a Paretian Liberal, AI Version

What we are approaching in this is a conflict between users being able to control what they see (we could call this "liberalism") and societal benefit (we could call this "Paretism" in the sense of it being Pareto-optimal). Amartya Sen’s <d-cite key="sen1970impossibility"></d-cite> Impossibility of a Paretian Liberal  shows that two mild normative principles—minimal personal freedom and Pareto efficiency—cannot coexist when preferences are nosy. The result is tiny, elegant, and unavoidable.

### 4.1 Principle 1: Minimal Liberalism

Each person should have at least one personal sphere in which their strict preference must determine what they see.

Formally: for each user $i$, there exist distinct states $x, y$ such that:

1. $x$ and $y$ differ only in what happens to $i$;
2. if $i$ prefers $x$ to $y$, then society must prefer $x$ to $y$.

This is much less restrictive than full liberalism.

### 4.2 Principle 2: Pareto

If everyone strictly prefers $x$ to $y$, society must prefer $x$ to $y$:

$$\forall i:\; x \succ_i y \;\Rightarrow\; x \succ^* y.$$

A minimal idea of collective rationality.

### 4.3 Sen's theorem

With at least two agents and nosy preferences, there exists no social preference ordering that satisfies minimal liberalism and Pareto.

> This is a logical impossibility, not a political statement.

### 4.4 AI-flavored toy example

Consider whether the model shows topic $T$ to:

- Ada (A),
- Chris (C).

A social state is

$$s_{ij} = (\text{A: show/not},\; \text{C: show/not}).$$

The four states: $s_{00}$ (none), $s_{10}$ (Ada only), $s_{01}$ (Chris only), $s_{11}$ (both).

Nosy preferences:

- **Ada:** $s_{00} \succ_A s_{10} \succ_A s_{01} \succ_A s_{11}$
- **Chris:** $s_{10} \succ_C s_{01} \succ_C s_{00} \succ_C s_{11}$

Now impose minimal liberalism:

- Ada controls her own exposure when Chris is off: $s_{00} \succ^* s_{10}$
- Chris controls his exposure when Ada is off: $s_{01} \succ^* s_{00}$

Chaining gives $s_{01} \succ^* s_{10}$.

But both Ada and Chris strictly prefer $s_{10} \succ s_{01}$.

Pareto then requires $s_{10} \succ^* s_{01}$.

**Contradiction.** Hence, when preferences are nosy, you cannot guarantee (even minimal) per-user control and respect unanimous improvements. Personalization cannot solve all alignment problems, and social choice cannoot guarantee even a minimal personal freedom.

## 5. When Liberalism, when Social Choice?

When preferences are non-nosy—your utility depends only on your slice $x_i$—Sen's contradiction disappears. Personalization is coherent.

Examples:

- developer tools
- writing-style settings
- reading preferences
- private tutoring or coaching systems
- verbosity, tone, formatting controls

Here, "just personalize" is normatively attractive.

In nosy domains, personalization cannot satisfy all relevant preferences:

- misinformation or public-sphere content
- safety concerns with externalities
- fairness and group-norm constraints
- content shaping public norms or collective outcomes

In such domains, alignment requires institutional design:

- shared floors (global constraints on model behavior),
- customizable ceilings (user-level control within safe bounds),
- explicit rules about which nosy preferences count

and techniques from social choice that will shape the research.

---

## 6. A Call to Action

Modern ML systems already combine personalization with social choice. Alignment decisions appear at multiple layers:

1. **Base model training:** Data curation, capability shaping, and architectural choices create global defaults.
2. **Safety and alignment fine-tuning:** RLHF, constitutional training, and policy filters impose shared constraints.
3. **Deployment governance:** Providers enforce global rules: disallowed content, auditing, rate limits, logging, and oversight.
4. **User-level personalization:** Style settings, topical preferences, writing tone, tutoring paths, etc.

A helpful mental model distinguishes:

- a **shared floor** (what models cannot do for anyone),
- a **personal ceiling** (what users can customize above that floor).

Sen's theorem explains why we cannot simply expand both floor and ceiling simultaneously in nosy domains. Giving users more decisive control while also trying to satisfy more global consensus constraints inevitably creates contradictions.

Many alignment debates are ultimately about:

- Where the floor should be.
- How high the ceiling can go.
- Which nosy preferences are normatively relevant.
- Which ones we deliberately do not encode.

Once we see the structure, personalization and social choice stop being opponents and instead become complementary design tools for ML systems that serve many people with, even when we are nosy.