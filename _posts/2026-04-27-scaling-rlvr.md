---
layout: distill
title: Scaling Online RLVR Done Right with Decoupled Generation & Optimization
description: Reinforcement Learning with Verifiable Rewards (RLVR) optimizes large language models on tasks with objective correctness criteria by directly leveraging deterministic reward signals rather than learned preferences. While theoretically principled, online RLVR remains computationally prohibitive due to tight coupling of generation and optimization, which inflates memory and severely limits training throughput. We prove this gap is architectural, not fundamental. Online RLVR can be reformulated exactly as offline supervised fine-tuning with importance-weighted samples. We introduce Decoupled Generation & Optimization (DGO), a two-phase paradigm that separates generation from optimization, reducing peak memory by ~18-31% and training time by ~75-85% while enabling multi-epoch training. Our framework unifies existing offline methods, exposes systematic theory-practice mismatches, and establishes DGO as the first method where theoretical optimal weights align perfectly with implementation. We show scaling online RLVR is achievable when done right, through principled decoupling and theoretically-grounded design.
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
bibliography: 2026-04-27-scaling-rlvr.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
    subsections:
      - name: Supervised Fine-tuning
      - name: Reinforcement Learning with Verifiable Rewards
  - name: "From Online RLVR to Offline SFT: A Theoretical Bridge"
    subsections:
      - name: "Reverse KL: Equivalence to Original RLVR Objective"
      - name: "Forward KL: Reformulation to Weighted Supervised Learning"
      - name: Connecting Forward and Reverse KL
  - name: The Decoupled Generation & Optimization (DGO) Paradigm
    subsections:
      - name: From Theory to Paradigm
      - name: Algorithm Overview
      - name: Scaling Advantages
      - name: Analytical Comparison
      - name: Empirical Comparison
  - name: Conclusion
---

# Introduction

Large language models (LLMs) are typically trained in two stages: pre-training on massive corpora to learn general language understanding, followed by fine-tuning to align the model with specific tasks or outcomes. Fine-tuning can be broadly categorized into *supervised fine-tuning (SFT)* <d-cite key="wei2021finetuned"></d-cite><d-cite key="ouyang2022training"></d-cite> and *reinforcement learning with verifiable rewards (RLVR)* <d-cite key="lightman2023let"></d-cite><d-cite key="wang2025reinforcement"></d-cite><d-cite key="shao2024deepseekmath"></d-cite><d-cite key="yu2025dapo"></d-cite>, each with distinct objectives and optimization strategies. RLVR is particularly powerful for tasks where correctness can be objectively verified, such as mathematical problem-solving, code generation, and logical reasoning, where the reward signal comes from deterministic verification rather than learned preference models.

## Supervised Fine-tuning

The first approach, supervised fine-tuning (SFT), constitutes a fundamental paradigm for adapting pre-trained language models to downstream tasks through maximum likelihood estimation over curated datasets. Given a pre-collected dataset $$\mathcal{D}$$ comprising prompt-response pairs $$(x, y)$$, the standard SFT objective seeks to minimize the negative log-likelihood of target sequences under the parameterized policy $$\pi_{\theta}$$. Leveraging the autoregressive factorization inherent to transformer-based language models, this objective decomposes into a sum of token-level cross-entropy losses, where each token $$y_l$$ is predicted conditioned on the preceding context $$y_{< l}$$ and the input prompt $$x$$. This formulation can be viewed as a special case of behavioral cloning from imitation learning, where the model learns to replicate expert demonstrations encoded in the training corpus. Formally, the objective is:

$$
\begin{aligned}
\min_{\theta}\mathcal{J}_{\mathrm{SFT}}(\theta) &\triangleq \mathbb{E}_{(x,y) \sim \mathcal{D}}\left[-\log\pi_{\theta}(y \mid x)\right]\\
&=\mathbb{E}_{(x,y) \sim \mathcal{D}}\left[\sum_{l=1}^L-\log\pi_{\theta}(y_l\mid y_{< l},x)\right].
\end{aligned}
$$

## Reinforcement Learning with Verifiable Rewards

In contrast to the imitation-based approach of SFT, RLVR shifts the objective from imitating fixed demonstrations to optimizing a verifiable reward function $$r(x, y)$$ that objectively measures correctness or task success. Unlike learned reward models that approximate human preferences, verifiable rewards provide ground-truth signals—such as whether a mathematical solution is correct, code passes unit tests, or a logical proof is valid. The KL-regularized RLVR objective balances maximizing expected reward against staying close to a reference policy $$\pi_{\mathrm{ref}}$$, preventing the model from deviating too far and producing degenerate outputs. Formally, the objective is:

$$
\max_{\theta} \mathcal{J}_{\mathrm{RL}}(\theta) \triangleq \mathbb{E}_{x\sim \mathcal{X}, y \sim \pi_{\theta}(\cdot \mid x)} [r(x, y)] - \beta \cdot \mathrm{KL}(\pi_\theta(\cdot \mid x) \parallel \pi_{\mathrm{ref}}(\cdot \mid x))
$$

where $$\beta > 0$$ is a temperature parameter that controls the trade-off between reward maximization and policy regularization, and $$r(x, y)$$ is a verifiable reward that can be computed deterministically (e.g., $$r(x,y) = \mathbb{1}[\mathrm{answer}(y) = \mathrm{ground\_truth}(x)]$$ for mathematical reasoning).

**Drawbacks of online RLVR.** While the KL-regularized RLVR objective is theoretically elegant and eliminates reward hacking concerns through verifiable signals, its practical implementation faces critical challenges that hinder scalability. Specifically, online RLVR methods like PPO <d-cite key="schulman2017proximal"></d-cite> and GRPO <d-cite key="shao2024deepseekmath"></d-cite> tightly couple sample generation with policy optimization, requiring simultaneous loading of policy and reference models, discarding samples after minimal reuse, and performing gradient updates in lockstep with generation. This coupling not only slows training but also inflates memory consumption, severely limiting the scale of models and batch sizes that can be trained. In addition, importance sampling from an evolving policy $$\pi_{\theta}$$ becomes compute-inefficient for off-policy update: as the policy shifts during training, earlier samples drift off-policy, leading to high-variance gradient estimates and poor sample efficiency. These bottlenecks motivate a reformulation that decouples generation from optimization while maintaining theoretical soundness.

**Closed-form optimal policy.** To address these challenges, we begin by analyzing the theoretical solution to the KL-regularized RLVR objective. It turns out that the optimal policy $$\pi^*(y \mid x)$$ has a closed-form solution that reweights the reference policy by the exponentiated reward, normalized by the partition function $$Z(x)$$:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\exp(r(x, y) / \beta)
$$

where $$Z(x) = \mathbb{E}_{y \sim \pi_{\mathrm{ref}}(\cdot \mid x)}\left[\exp(r(x, y) / \beta)\right]=\sum_{y}\pi_{\mathrm{ref}}(y\mid x)\exp(r(x, y) / \beta)$$ is the partition function ensuring $$\sum_y\pi^*(y \mid x)=1$$.

<details>
<summary>Proof of Close-Form Optimal Policy.</summary>
    
    We want to find the policy $\pi_{\theta}$ that maximizes the KL-regularized RL objective:
    
    $$
    \max_{\theta} \mathcal{J}_{\mathrm{RL}}(\theta) = \mathbb{E}_{x\sim \mathcal{X}, y \sim \pi_{\theta}(\cdot \mid x)} [r(x, y)] - \beta \cdot \mathrm{KL}(\pi_\theta(\cdot \mid x) \parallel \pi_{\mathrm{ref}}(\cdot \mid x))
    $$
    
    Step 1: Expand the KL divergence term
    
    The KL divergence can be written as:
    
    $$
    \mathrm{KL}(\pi_\theta(\cdot \mid x) \parallel \pi_{\mathrm{ref}}(\cdot \mid x)) = \mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}\left[\log \pi_{\theta}(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x)\right]
    $$
    
    Substituting this into the objective:
    
    $$
    \begin{aligned}
    \mathcal{J}_{\mathrm{RL}}(\theta) &= \mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)} [r(x, y)] - \beta \mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}\left[\log \pi_{\theta}(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x)\right] \\
    &= \mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)} \left[r(x, y) - \beta \log \pi_{\theta}(y \mid x) + \beta \log \pi_{\mathrm{ref}}(y \mid x)\right]
    \end{aligned}
    $$
    
    Step 2: Change from expectation to summation
    
    Since $\mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}[f(y)] = \sum_y \pi_{\theta}(y \mid x) f(y)$, we can write:
    
    $$
    \mathcal{J}_{\mathrm{RL}}(\theta) = \sum_y \pi_{\theta}(y \mid x) \left[r(x, y) + \beta \log \pi_{\mathrm{ref}}(y \mid x) - \beta \log \pi_{\theta}(y \mid x)\right]
    $$
    
    Step 3: Apply the calculus of variations
    
    To find the optimal policy, we use the method of Lagrange multipliers with the constraint $\sum_y \pi_{\theta}(y \mid x) = 1$. The Lagrangian is:
    
    $$
    \mathcal{L} = \sum_y \pi_{\theta}(y \mid x) \left[r(x, y) + \beta \log \pi_{\mathrm{ref}}(y \mid x) - \beta \log \pi_{\theta}(y \mid x)\right] - \lambda\left(\sum_y \pi_{\theta}(y \mid x) - 1\right)
    $$
    
    Taking the derivative with respect to $\pi_{\theta}(y \mid x)$ and setting it to zero:
    
    $$
    \frac{\partial \mathcal{L}}{\partial \pi_{\theta}(y \mid x)} = r(x, y) + \beta \log \pi_{\mathrm{ref}}(y \mid x) - \beta \log \pi_{\theta}(y \mid x) - \beta - \lambda = 0
    $$
    
    Solving for $\pi_{\theta}(y \mid x)$:
    
    $$
    \begin{aligned}
    \beta \log \pi_{\theta}(y \mid x) &= r(x, y) + \beta \log \pi_{\mathrm{ref}}(y \mid x) - \beta - \lambda \\
    \log \pi_{\theta}(y \mid x) &= \frac{r(x, y)}{\beta} + \log \pi_{\mathrm{ref}}(y \mid x) - 1 - \frac{\lambda}{\beta} \\
    \pi_{\theta}(y \mid x) &= \pi_{\mathrm{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta} - 1 - \frac{\lambda}{\beta}\right)
    \end{aligned}
    $$
    
    Step 4: Determine the normalization constant
    
    Using the constraint $\sum_y \pi_{\theta}(y \mid x) = 1$:
    
    $$
    \sum_y \pi_{\mathrm{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta} - 1 - \frac{\lambda}{\beta}\right) = 1
    $$
    
    Let $C = \exp\left(-1 - \frac{\lambda}{\beta}\right)$. Then:
    
    $$
    C \sum_y \pi_{\mathrm{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right) = 1
    $$
    
    Therefore:
    
    $$
    C = \frac{1}{\sum_y \pi_{\mathrm{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)} = \frac{1}{Z(x)}
    $$
    
    where $Z(x) = \sum_y \pi_{\mathrm{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)$ is the partition function.
    
    Step 5: Final optimal policy
    
    Substituting back, we obtain the optimal policy:
    
    $$
    \pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y \mid x) \exp\left(\frac{r(x, y)}{\beta}\right)
    $$
    
    This completes the proof. $\square$
</details>

This theoretical result suggests a natural question: *Can we directly learn the optimal policy through offline supervised learning?* The answer is yes, and this insight forms the foundation of our **Decoupled Generation & Optimization (DGO)** paradigm.

---

# From Online RLVR to Offline SFT: A Theoretical Bridge

The key insight is that *minimizing the Kullback-Leibler (KL) divergence from the optimal policy* is equivalent to the original RLVR objective. We establish this through two complementary perspectives: reverse KL and forward KL. While some of the underlying theoretical connections have been explored in prior work, e.g., the forward KL formulation in RAML <d-cite key="NIPS2016_2f885d0f"></d-cite> and the weighted policy learning framework in AWR <d-cite key="peng2019advantage"></d-cite>, *our contribution lies in providing a more comprehensive theoretical framework to reveal how these principles unlock scalable online RLVR for LLMs*. Crucially, the verifiable nature of RLVR rewards means we can compute exact reward values without approximation error, unlike learned reward models.

## Reverse KL: Equivalence to Original RLVR Objective

The reverse KL perspective $\mathrm{KL}(\pi_{\theta}(\cdot \mid x) \parallel \pi^*(\cdot \mid x))$ in fact has a deep connection to the original RLVR objective. Expanding the reverse KL:

$$
\begin{aligned}
\mathrm{KL}\left(\pi_{\theta}(\cdot \mid x) \parallel \pi^*(\cdot \mid x)\right)
&= \mathbb{E}_{y \sim \pi_{\theta}(\cdot\mid x) }\left[\log \pi_{\theta}(y \mid x) - \log \pi^*(y \mid x)\right] \\
&= \mathbb{E}_{y \sim \pi_{\theta}(\cdot\mid x) }\left[\log \pi_{\theta}(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x) - \tfrac{1}{\beta} r(x,y) + \log Z(x)\right] \\
&= \mathbb{E}_{y \sim \pi_{\theta}(\cdot\mid x) }\left[\log \pi_{\theta}(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x)\right] - \tfrac{1}{\beta}\mathbb{E}_{y \sim \pi_{\theta}(\cdot\mid x) }[r(x,y)] + \log Z(x) \\
&= \mathrm{KL}(\pi_{\theta}(\cdot \mid x)\parallel\pi_{\mathrm{ref}}(\cdot \mid x)) - \tfrac{1}{\beta}\mathbb{E}_{y \sim \pi_{\theta}(\cdot\mid x) }[r(x,y)] + \log Z(x).
\end{aligned}
$$

Rearranging and multiplying by $-\beta$:

$$
\begin{aligned}
\mathcal{J}_{\mathrm{RL}}(\theta) &= \mathbb{E}_{x \sim \mathcal{X}}\left[\mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}[r(x,y)] - \beta\,\mathrm{KL}(\pi_{\theta}(\cdot \mid x)\parallel\pi_{\mathrm{ref}}(\cdot \mid x))\right]  \\
&= \mathbb{E}_{x \sim \mathcal{X}}\left[\beta\log Z(x)\right] - \beta\,\mathbb{E}_{x \sim \mathcal{X}}\left[\mathrm{KL}\left(\pi_{\theta}(\cdot \mid x) \parallel \pi^*(\cdot \mid x)\right)\right].
\end{aligned}
$$

Since $\log Z(x)$ is independent of $\theta$, *maximizing the original RLVR objective is equivalent to minimizing the reverse KL to the optimal policy, i.e.,*

$$
\min_{\theta} \, \mathcal{J}_{\mathrm{RL}}(\theta) \iff \min_{\theta} \, \mathbb{E}_{x \sim \mathcal{X}}\left[\mathrm{KL}(\pi_{\theta}(\cdot \mid x) \parallel \pi^*(\cdot \mid x))\right].
$$

While the reverse KL perspective provides an exact equivalence to the original RL objective, it still requires sampling from the current policy $\pi_{\theta}$ during optimization. This means we must perform online rollouts at each training step, which remains computationally expensive and memory-intensive. This limitation motivates us to consider the forward KL perspective, which enables a fully offline approach by sampling from a fixed reference policy $\pi_{\mathrm{ref}}$ instead.

## Forward KL: Reformulation to Weighted Supervised Learning

Starting from the forward KL divergence $$\mathrm{KL}(\pi^*(\cdot\mid x) \parallel \pi_{\theta}(\cdot \mid x))$$, we can derive an equivalent weighted SFT objective in an offline manner. The forward KL measures how well our learned policy $\pi_{\theta}$ approximates the optimal policy $$\pi^*$$:

$$
\min_{\theta} \mathbb{E}_{x \sim \mathcal{X}}\left[\mathrm{KL}(\pi^*(\cdot\mid x) \parallel \pi_{\theta}(\cdot \mid x))\right].
$$

Expanding the KL divergence:

$$
\begin{aligned}
\mathrm{KL}(\pi^*(\cdot\mid x) \parallel \pi_{\theta}(\cdot\mid x)) &= \mathbb{E}_{y \sim \pi^*(\cdot \mid x)}\left[\log \pi^*(y\mid x) - \log\pi_{\theta}(y\mid x)\right] \\[8pt]
&= -H(\pi^*(\cdot \mid x)) - \mathbb{E}_{y \sim \pi^*(\cdot\mid x) }\left[\log\pi_{\theta}(y\mid x)\right].
\end{aligned}
$$

where $H(\pi^*(\cdot \mid x))$ is the entropy of the optimal policy. Since the entropy does not depend on $\theta$, we can drop it from the optimization objective.

<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key insight:</strong> This formulation reveals a theoretical foundation behind standard SFT. Standard SFT can be viewed as a special case where we minimize $\mathrm{KL}(\pi^*(\cdot\mid x) \parallel \pi_{\theta}(\cdot \mid x))$ with the training data $\mathcal{D}$ being samples curated or selected to follow an implicit optimal policy $\pi^*$. In other words, <em>when we perform SFT on carefully curated demonstrations, we are implicitly learning to match an optimal policy defined by those demonstrations</em>.
</div>

Substituting the closed-form expression for $$\pi^*$$:

$$
\begin{aligned}
\mathbb{E}_{y \sim \pi^*(\cdot \mid x)}\left[\log\pi_{\theta}(y\mid x)\right]
&= \sum_{y} \pi^*(y\mid x)\log \pi_{\theta}(y\mid x) \\
&= \sum_{y} \frac{\exp(r(x, y) / \beta)}{Z(x)}\pi_{\mathrm{ref}}(y\mid x)\log \pi_{\theta}(y\mid x) \\
&= \mathbb{E}_{y \sim \pi_{\mathrm{ref}}(\cdot \mid x)}\left[\frac{\exp(r(x, y) / \beta)}{Z(x)}\log \pi_{\theta}(y\mid x)\right].
\end{aligned}
$$

This leads to the *weighted SFT objective*:

$$
\min_{\theta} \mathcal{J}_{\mathrm{W-SFT}}(\theta) \triangleq \mathbb{E}_{x \sim \mathcal{X}}\left[\mathbb{E}_{y \sim \pi_{\mathrm{ref}}(\cdot \mid x)}\left[-w(x,y)\log \pi_{\theta}(y\mid x)\right]\right].
$$

where $w(x,y) = \frac{\exp(r(x, y) / \beta)}{Z(x)}$ is the sample weight.

<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key insight:</strong> This shows that optimal RLVR can be solved by sampling responses from the reference policy $\pi_{\mathrm{ref}}$, computing sample weights based on verifiable rewards (which can be computed efficiently and exactly), and performing weighted supervised learning—<em>no online policy rollouts needed!</em> The verifiable nature of rewards means no reward model needs to be loaded during training, further reducing memory requirements.
</div>

## Connecting Forward and Reverse KL

### Optimal Connection

Both forward and reverse KL perspectives lead to the same optimal policy $$\pi^*$$, but they connect to the original RL objective in different ways.

**Reverse KL is exactly equivalent to the original RL objective.** As shown in the previous section, the reverse KL derivation reveals:

$$
\begin{aligned}
\mathcal{J}_{\mathrm{RL}}(\theta) &= \mathbb{E}_{x \sim \mathcal{X}}\left[\mathbb{E}_{y \sim \pi_{\theta}(\cdot \mid x)}[r(x,y)] - \beta\,\mathrm{KL}(\pi_{\theta}(\cdot \mid x)\parallel\pi_{\mathrm{ref}}(\cdot \mid x))\right]  \\
&= \mathbb{E}_{x \sim \mathcal{X}}\left[\beta\log Z(x)\right] - \beta\,\mathbb{E}_{x \sim \mathcal{X}}\left[\mathrm{KL}\left(\pi_{\theta}(\cdot \mid x) \parallel \pi^*(\cdot \mid x)\right)\right].
\end{aligned}
$$

Since $\log Z(x)$ is independent of $\theta$, this establishes a direct equivalence: *maximizing the RL objective* $$\mathcal{J}_{\mathrm{RFT}}(\theta)$$ *is exactly equivalent to minimizing* $$\mathrm{KL}(\pi_{\theta} \parallel \pi^*)$$. This is not an approximation—it's an algebraic identity.

**Forward KL is closely related through the shared optimal policy.** While forward KL $$\mathrm{KL}(\pi^* \parallel \pi_{\theta})$$ does not directly equal the RL objective, it shares the same global optimum. Both KL divergences achieve their minimum value of zero at $\pi_{\theta} = \pi^*$:

$$
\min_{\theta} \, \mathrm{KL}(\pi^* \parallel \pi_{\theta}) = 0 \iff \pi_{\theta} = \pi^* \iff \min_{\theta} \, \mathrm{KL}(\pi_{\theta} \parallel \pi^*) = 0.
$$

Therefore, minimizing forward KL optimizes toward the same policy $$\pi^*$$ that maximizes the RL objective. The key insight is that *forward KL enables offline optimization*: by sampling from a fixed reference policy $$\pi_{\mathrm{ref}}$$ with importance weights $w(x,y) = \exp(r/\beta)/Z(x)$, we can approximate samples from $$\pi^*$$ and perform standard supervised learning. This transforms the online RL problem into an offline weighted SFT problem.

### Asymptotic Equivalence

While forward and reverse KL lead to different optimization procedures, a remarkable result shows they become nearly indistinguishable as the learned policy approaches the optimum. Specifically, the difference between the two KL directions vanishes quadratically as $\Delta_x \to 0$:

$$
\big|\mathrm{KL}(\pi^{*}(\cdot\mid x)\Vert \pi_{\theta}(\cdot\mid x)) - \mathrm{KL}(\pi_{\theta}(\cdot\mid x)\Vert \pi^{*}(\cdot\mid x))\big| = \mathcal{O}(\Delta_x^{2}),
$$

where $$\Delta_x := \mathrm{TV}(\pi^{*}(\cdot\mid x),\,\pi_{\theta}(\cdot\mid x))$$ measures the total variation distance between $\pi_{\theta}$ and $$\pi^*$$.

<details>
<summary>Proof of Quadratic Convergence.</summary>
  Fix a reference distribution $Q$ on a finite support with $Q(y) \ge c > 0$ for all $y$, and let $P$ be any distribution on the same support. We have the identity:
  $$
  \mathrm{KL}(P\Vert Q) - \mathrm{KL}(Q\Vert P) = \sum_y \big(P(y)-Q(y)\big)\log\Big(\tfrac{P(y)}{Q(y)}\Big).
  $$

  Let $\Delta := \mathrm{TV}(P,Q) = \tfrac{1}{2}\sum_y |P(y)-Q(y)|$. In the small-distance regime where $\Delta \ll 1$, write $P = Q + \delta$ with $\sum_y \delta(y)=0$ and $\tfrac{1}{2}\sum_y |\delta(y)|=\Delta$.

  Using a Taylor expansion of the logarithm for small $\delta(y)/Q(y)$, we have for each $y$:
  $$
  \log\Big(\tfrac{Q(y)+\delta(y)}{Q(y)}\Big)
  = \tfrac{\delta(y)}{Q(y)} - \tfrac{\delta(y)^2}{2Q(y)^2} + \mathcal{O}\Big(\tfrac{|\delta(y)|^3}{Q(y)^3}\Big),
  $$

  where the big-$\mathcal O$ constant is universal (independent of $P$) and the dependence on $Q$ is controlled by the lower bound $Q(y)\ge c$.

  Substituting into the KL difference:

  $$
  \begin{aligned}
  \mathrm{KL}(P\Vert Q) - \mathrm{KL}(Q\Vert P)
  &= \sum_y \delta(y)\Big[\tfrac{\delta(y)}{Q(y)} - \tfrac{\delta(y)^2}{2Q(y)^2} + \mathcal{O}\Big(\tfrac{|\delta(y)|^3}{Q(y)^3}\Big)\Big] \\
  &= \sum_y \tfrac{\delta(y)^2}{Q(y)} - \sum_y \tfrac{\delta(y)^3}{2Q(y)^2} + \mathcal{O}\Big(\sum_y \tfrac{|\delta(y)|^4}{Q(y)^3}\Big).
  \end{aligned}
  $$

  Since $Q(y)\ge c>0$, all denominators are bounded by constants. Moreover, $|\delta|_1 = 2\Delta$ and $|\delta|_2 \le |\delta|_1$, so
  $$
  \sum_y \delta(y)^2 = \mathcal{O}(\Delta^{2}),\qquad
  \sum_y |\delta(y)|^3 = \mathcal{O}(\Delta^{3}),\qquad
  \sum_y |\delta(y)|^4 = \mathcal{O}(\Delta^{4}).
  $$

  As a result,
  $$
  \big|\mathrm{KL}(P\Vert Q) - \mathrm{KL}(Q\Vert P)\big| = \mathcal{O}(\Delta^2)\quad\text{as }\Delta\to 0.
  $$

  Applying this to $P=\pi_{\theta}$ and $Q=\pi^{*}$ yields the claimed result with $\Delta = \Delta_x$. $\square$
</details>

As the learned policy $\pi_{\theta}$ approaches the optimal policy $$\pi^*$$ (i.e., $\Delta_x \to 0$), the difference between forward and reverse KL objectives diminishes quadratically, meaning both converge to the same optimum. Crucially, the KL-regularized RLVR objective constrains $$\pi^*$$ to remain close to $\pi_{\mathrm{ref}}$ by design: the closed-form solution $$\pi^*(y\mid x) = \pi_{\mathrm{ref}}(y\mid x)\exp(r(x,y)/\beta)/Z(x)$$ shows that $$\pi^*$$ is merely a reweighted version of $\pi_{\mathrm{ref}}$, with the temperature $\beta$ controlling the deviation. Since $\pi_{\theta}$ starts from (or near) $\pi_{\mathrm{ref}}$ and optimizes toward $$\pi^*$$, both remain in a neighborhood of $\pi_{\mathrm{ref}}$ throughout training, ensuring $\Delta_x = \mathrm{TV}(\pi^*(\cdot\mid x), \pi_{\theta}(\cdot\mid x))$ is naturally small and validating the quadratic bound.

### Comprehensive Comparison
To systematically understand the relationship between forward and reverse KL, we provide a detailed comparison across four key dimensions: their connection to the optimal policy $$\pi^*$$, sampling requirements, optimization behavior, and practical implementation.

<table style="width:100%; border-collapse: collapse; margin: 1.2em 0 1.5em 0; font-size: 0.95rem; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">
  <caption style="caption-side: top; text-align: left; font-weight: 600; margin-bottom: 0.4em; text-align: center;">
    Table 1. A Comprehensive Comparison between Forward and Reverse KL.
  </caption>
  <thead>
    <tr style="background-color: rgba(0, 128, 0, 0.05); font-weight: 600; text-align: left;">
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 22%;">Aspect</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 39%;">Forward KL: \(\mathrm{KL}(\pi^{*} \parallel \pi_{\theta})\)</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 39%;">Reverse KL: \(\mathrm{KL}(\pi_{\theta} \parallel \pi^{*})\)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: 600;">Connection to \(\pi^{*}\)</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        Equivalent to maximum-likelihood estimation of \(\pi_{\theta}\) from \(\pi^{*}\).
      </td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        Equivalent to original RL objective:<br>
        \(\max \mathcal{J}_{\mathrm{RL}} \iff \min \mathbb{E}\big[\mathrm{KL}(\pi_{\theta} \parallel \pi^{*})\big]\).
      </td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: 600;">Sampling Strategy</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        Samples from fixed \(\pi_{\mathrm{ref}}\) (offline).
      </td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        Samples from current policy \(\pi_{\theta}\) (online).
      </td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: 600;">Optimization Behavior</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        Mode-covering: encourages \(\pi_{\theta}\) to place mass on all modes of \(\pi^{*}\).<br>
        More diverse, explores broadly.
      </td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        Mode-seeking: encourages \(\pi_{\theta}\) to focus on dominant modes of \(\pi^{*}\).<br>
        More concentrated, focuses narrowly.
      </td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: 600;">Practical Implementation</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        ✓ Offline optimization<br>
        ✓ Fixed reference policy<br>
        ✓ Decoupled generation-optimization<br>
        ✓ Multi-epoch training
      </td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">
        ✗ Online rollouts required<br>
        ✗ Moving policy \(\pi_{\theta}\)<br>
        ✗ Coupled generation-optimization<br>
        ✗ Limited data reuse
      </td>
    </tr>
  </tbody>
</table>


<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key takeaway:</strong> Forward KL $\mathrm{KL}(\pi^* \mid \pi_{\theta})$ and reverse KL $\mathrm{KL}(\pi_{\theta} \mid \pi^*)$ both optimize toward $\pi^*$ and are equivalent to the original RLVR objective in different perspectives. Forward KL enables offline optimization through importance-weighted sampling from a fixed reference, making it ideal for resource-efficient RLVR. For verifiable rewards, sample weights can be computed efficiently without a reward model, and the deterministic nature of verification eliminates reward hacking concerns. The choice determines sampling strategy (offline vs. online) and exploration behavior (broad vs. focused), but not the destination.
</div>

---

# The Decoupled Generation & Optimization (DGO) Paradigm

Having established the theoretical equivalence between online RLVR and offline weighted SFT, we now present **Decoupling Generation & Optimization (DGO)**, a practical paradigm that implements the forward KL objective correctly while addressing the scaling challenges of current approaches. DGO is particularly well-suited for RLVR because verifiable rewards can be computed efficiently during the generation phase without requiring a separate reward model during optimization.

### From Theory to Paradigm

The theoretical results from the preceding sections, e.g., the closed-form optimal policy $$\pi^*$$ and the forward KL reformulation to weighted SFT, establish that online RLVR can be solved offline. However, direct implementation requires addressing two practical challenges: *(1) How to estimate the prompt-specific partition function* $Z(x)$, and *(2) How to compute and apply sample weights* $w(x,y)$ in a scalable two-phase algorithm.

**Partition function estimation.** The sample weight $w(x,y) = \exp(r(x,y)/\beta)/Z(x)$ requires computing $$Z(x) = \mathbb{E}_{y \sim \pi_{\mathrm{ref}}(\cdot\mid x)}[\exp(r(x,y)/\beta)]$$, which is intractable for large output spaces. DGO employs Monte Carlo estimation: for each prompt $x$, sample $N$ responses $$\{y_n\}_{n=1}^N \sim \pi_{\mathrm{ref}}(\cdot\mid x)$$ and compute

$$
\hat{Z}(x) = \frac{1}{N}\sum_{n=1}^N \exp(r(x,y_n)/\beta).
$$

This unbiased estimator converges to $Z(x)$ as $N \to \infty$ and enables prompt-level normalization, ensuring that training is not biased toward prompts where $\pi_{\mathrm{ref}}$ already generates high-reward responses.

<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key insight:</strong> $\hat{Z}(x)$ is the <em>performance baseline</em>, i.e., the average score across all sampled responses, that reveals what $\pi_{\mathrm{ref}}$ can truly achieve. Larger $N$ sharpens this estimate, leading to more stable, confident optimization.
</div>

**Sample weight computation and application.** Once $\hat{Z}(x)$ is estimated, the normalized importance weight for each sample is $w(x,y) = \exp(r(x,y)/\beta)/\hat{Z}(x)$. These weights transform the intractable forward KL objective into a tractable weighted SFT objective that can be optimized via standard mini-batch gradient descent. Crucially, because $w(x,y)$ depends only on the fixed reference policy $\pi_{\mathrm{ref}}$ and not on the evolving policy $\pi_{\theta}$, the weights remain valid throughout training, enabling multi-epoch optimization without regenerating samples.

<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key insight:</strong> DGO automatically <em>amplifies what works and suppresses what doesn't</em>: samples beating the baseline get $w(x,y) > 1$ (teaching the model to do more of this), while under-performers get $w(x,y) < 1$ (teaching the model to avoid this). This method extracts signal from both successful and unsuccessful attempts, enabling more data-efficient learning compared to methods that rely solely on positive examples.
</div>

These implementation choices coalesce into a two-phase paradigm that fully decouples sample generation from policy optimization (see Algo.1 below):

- **Phase 1: Generation.** Fix $\pi_{\mathrm{ref}} = \pi_{\theta_0}$, sample $N$ responses per prompt, evaluate verifiable rewards $r(x,y_n)$ (e.g., check correctness via execution, unit tests, or formal verification), estimate partition functions $\hat{Z}(x)$, and compute weights $w(x,y_n)$. Store the weighted dataset $\mathcal{D} = \{(x, y, w(x,y))\}$. This phase loads only the reference model (for generation). *Unlike general RL, no reward model is needed:* verifiable rewards can be computed via deterministic verification functions with negligible computational cost.
- **Phase 2: Optimization.** Initialize $\pi_{\theta} = \pi_{\theta_0}$ and optimize the weighted SFT loss $$-\mathbb{E}_{(x,y,w) \in \mathcal{D}}[w \log \pi_{\theta}(y\mid x)]$$ for $T$ iterations via mini-batch gradient descent. This phase loads only the policy model and its optimizer states, with no reference model required. The dataset $\mathcal{D}$ supports multi-epoch training, permitting $T \gg N$ gradient steps per generation cycle.

## Algorithm Overview

<div class="algo-card" style="background-color: rgba(0, 128, 0, 0.05); padding: 1.2em 1.4em; border-radius: 10px; border: 1px solid rgba(0, 128, 0, 0.25); margin: 1.5em 0; font-size: 0.95rem; line-height: 1.5; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">
  <div style="font-weight: 600; letter-spacing: 0.01em; margin-bottom: 0.6em;">
    Algorithm 1: The Decoupled Generation &amp; Optimization (DGO) Paradigm
  </div>

  <div style="margin-bottom: 0.6em;">
    <span style="font-weight: 600;">Input:</span>
    Prompts \(\mathcal{X}\), responses per prompt \(N\), initial policy \(\pi_{\theta_{0}}\), reward function \(r(\cdot, \cdot)\),
    temperature \(\beta\), learning rate \(\eta\), training iterations \(T\).
  </div>

  <div style="margin-top: 0.8em; font-weight: 600;">
    Phase 1: Generation
    <span style="font-weight: 400; font-style: italic;">(Scalable generation)</span>
  </div>

  <div style="margin-top: 0.4em; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">

    <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
      <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">1.</div>
      <div>Initialize reference model $\pi_{\mathrm{ref}} \leftarrow \pi_{\theta_{0}}$.</div>
    </div>

    <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
      <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">2.</div>
      <div>For each prompt $x \in \mathcal{X}$:</div>
    </div>

    <div style="margin-left: 2.5em;">
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(a)</div>
        <div>Sample \(N\) responses: $\mathcal{Y}_{x} = \{y_n \sim \pi_{\mathrm{ref}}(\cdot \mid x)\}_{n=1}^N$.</div>
      </div>
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(b)</div>
        <div>Evaluate rewards: $\{r(x, y_n)\}_{n=1}^N$.</div>
      </div>
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(c)</div>
        <div>Estimate partition function: $\hat{Z}(x) = \frac{1}{N}\sum_{n=1}^N \exp(r(x,y_n)/\beta)$.</div>
      </div>
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(d)</div>
        <div>Compute sample weights: $w(x,y_n) = \frac{\exp(r(x,y_n)/\beta)}{\hat{Z}(x)}$.</div>
      </div>
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(e)</div>
        <div>Store dataset $\mathcal{D} = \{(x, y, w(x,y)) : x \in \mathcal{X}, y \in \mathcal{Y}_x\}$.</div>
      </div>
    </div>

  </div>

  <div style="margin-top: 0.8em; font-weight: 600;">
    Phase 2: Optimization
    <span style="font-weight: 400; font-style: italic;">(Efficient training)</span>
  </div>

  <div style="margin-top: 0.4em; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">

    <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
      <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">1.</div>
      <div>Initialize policy model \(\pi_{\theta} \leftarrow \pi_{\theta_{0}}\).</div>
    </div>

    <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
      <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">2.</div>
      <div>For iterations \(t = 1, \dots, T\):</div>
    </div>

    <div style="margin-left: 2.5em;">
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(a)</div>
        <div>Sample minibatch \(\mathcal{B} = \{(x_b, y_b, w_b)\}_{b=1}^B\) from \(\mathcal{D}\).</div>
      </div>
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(b)</div>
        <div>Compute weighted loss: \(\ell = -\frac{1}{B}\sum_{b=1}^B w_b \log \pi_{\theta}(y_b \mid x_b)\).</div>
      </div>
      <div style="display: flex; align-items: flex-start; margin-bottom: 0.15em;">
        <div style="width: 2.5em; text-align: right; padding-right: 0.4em;">(c)</div>
        <div>Update parameters: \(\theta_{t} \leftarrow \theta_{t-1} - \eta \nabla_{\theta} \ell\).</div>
      </div>
    </div>

  </div>

  <div style="margin-top: 0.8em;">
    <span style="font-weight: 600;">Output:</span>
    Optimized policy \(\pi_{\theta}\).
  </div>

</div>

## Scaling Advantages

DGO's two-phase decoupling delivers critical scaling advantages:

- **Scalable generation:** Phase 1 can leverage highly optimized inference engines like vLLM <d-cite key="kwon2023efficient"></d-cite>, TensorRT-LLM <d-cite key="nvidia2023tensorrtllm"></d-cite>, or SGLang <d-cite key="zheng2024sglang"></d-cite> that support continuous batching, paged attention <d-cite key="kwon2023efficient"></d-cite>, and speculative decoding <d-cite key="leviathan2023fast"></d-cite>. These frameworks can generate thousands of responses in parallel across multiple GPUs with minimal memory overhead, achieving higher throughput compared to training-optimized frameworks.
- **Efficient optimization:** Phase 2 operates on pre-generated data, allowing the use of standard training infrastructure (DeepSpeed <d-cite key="jacobs2023deepspeed"></d-cite>, FSDP <d-cite key="zhao2023pytorch"></d-cite>, Megatron <d-cite key="shoeybi2019megatron"></d-cite>) without the memory overhead of maintaining a reference model. Gradient accumulation and data parallelism can be freely applied since no online sampling is required.
- **Horizontal scaling:** Generation and optimization can be independently scaled across different hardware. For example, generation can run on inference-optimized clusters with high GPU utilization, while optimization runs on training clusters with large batch sizes and gradient accumulation.
- **Asynchronous pipeline:** The two phases can be pipelined asynchronously: new batches of data can be generated while previous batches are being optimized, maximizing hardware utilization and minimizing idle time.
- **Seamless production integration:** DGO's two-phase structure naturally aligns with existing deployment workflows: Phase 1 maps to production inference, Phase 2 to offline training. This requires no additional infrastructure, enabling continuous improvement loops where live interactions automatically become training data for the next iteration.

## Analytical Comparison

Having established the theoretical foundation of DGO, we now reveal how it relates to and fundamentally differs from existing offline RL approaches. The key insight: all offline RL methods can be unified under a single weighted SFT framework, but they differ critically in *where* they sample from and *how* they construct weights. This comparison exposes three fundamental design choices that determine whether a method is theoretically grounded, scalable, and practically effective.

### The Unified Weighted SFT Framework

Every offline RL variant (including RLVR) performs weighted supervised fine-tuning of the form:

$$
\min_{\theta} \mathbb{E}_{x}\,\mathbb{E}_{y\sim q(\cdot\mid x)}\big[-w(x,y)\,\log \pi_{\theta}(y\mid x)\big],
$$

where two design choices fully specify the algorithm:

- **Sampling distribution** $q(\cdot\mid x)$: Where do training responses come from?
- **Sample weight** $w(x,y)$: How much should we trust each response?

While this template appears simple, the devil is in the details. As Table 2 reveals, seemingly minor differences in these choices lead to drastically different scalability, sample efficiency, and theoretical guarantees.

<table style="width:100%; border-collapse: collapse; margin: 1.2em 0 1.5em 0; font-size: 0.95rem; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">
  <caption style="caption-side: top; text-align: left; font-weight: 600; margin-bottom: 0.4em; text-align: center;">
    Table 2. Comparison of different offline RL algorithms for LLM finetuning.
  </caption>
  <thead>
    <tr style="background-color: rgba(0, 128, 0, 0.05); font-weight: 600; text-align: left;">
      <th style="padding: 10px; border-bottom: 1px solid #ccc;">Algorithm</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; text-align: center;">Sampling Distribution</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; text-align: center;">Data Source</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; text-align: center;">Weight Estimate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">VAR <d-cite key="du2025simplify"></d-cite></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$q = \pi^*\ \mathrm{(implicit)}$$</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Limited, Pre-Collected Positive Samples</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$w(x,y) = \frac{\pi_{\mathrm{ref}}(y\mid x) \exp(r(x,y)/\beta)}{\sum_y \pi_{\mathrm{ref}}(y\mid x) \exp(r(x,y)/\beta)}$$</td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">SPR <d-cite key="zhang2024llm"></d-cite></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$q = \pi^*\ \mathrm{(implicit)}$$</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Limited, Pre-Collected Positive Samples</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$w(x,y) = \exp((Q(x,y) − W(x)) / \beta)$$</td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">DFT <d-cite key="wu2025generalization"></d-cite></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$q = \pi^* \ \mathrm{(implicit)}$$</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Limited, Pre-Collected Positive Samples</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$w(x,y) = \mathrm{stop\_grad}(\pi_{\theta}(y \mid x))$$</td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">iw-SFT <d-cite key="qin2025supervised"></d-cite></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$q = \pi_{\mathrm{ref}}$$</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Unlimited, Generated & Curated Positive Samples</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$w(x,y) = q(y|x)/\pi_{\mathrm{ref}}(y \mid x)$$</td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Refit <d-cite key="mukherjeeoffline"></d-cite></td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$q = \pi_{\mathrm{ref}}$$</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Unlimited, Generated Samples</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$w(x,y) = r(x,y)$$</td>
    </tr>
    <tr>
      <td style="padding: 10px; border-bottom: 1px solid #eee; font-weight: 600;">DGO (Ours)</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$q = \pi_{\mathrm{ref}}$$</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">Unlimited, Generated Samples</td>
      <td style="padding: 10px; border-bottom: 1px solid #eee;">$$w(x,y) = \frac{\exp(r(x,y)/\beta)}{\frac{1}{N} \sum_{n=1}^N \exp(r(x,y_n)/\beta)}$$</td>
    </tr>
  </tbody>
</table>

### Fundamental Difference

Table 2 reveals two fundamental dimensions along which offline RL methods differ, each with profound implications.

**Dimension 1: Where do samples come from?** (Implicit $$\pi^*$$ vs. Explicit $\pi_{\mathrm{ref}}$)

Sampling distribution is the most striking division that separates VAR, SPR, and DFT from iw-SFT, Refit, and DGO. The former group assumes an implicit optimal policy $$q = \pi^*$$ and relies on pre-collected positive samples, which is their Achilles' heel. They inherit two crippling limitations: *(1) Training is fundamentally limited by the size of pre-collected datasets.* You cannot scale by simply generating more samples. *(2) Sampling from* $$\pi^*$$ *risks catastrophic forgetting.* Since the data-derived $$\pi^*$$ (defined by curated samples) can deviate substantially from $\pi_{\mathrm{ref}},$ potentially covering regions beyond what the KL-regularized RL objective would prescribe. As a result, training exclusively on such samples may cause catastrophic forgetting of capabilities encoded in $\pi_{\mathrm{ref}}$, leading to degraded performance on the broader distribution. In stark contrast, *iw-SFT, Refit, and DGO* sample from an explicit, controllable reference policy $q = \pi_{\mathrm{ref}}$. This unlocks *unlimited scalability*: generate as many samples as you need, whenever you need them. The cost of data is now just inference time, not expensive human annotation or cherry-picking. Moreover, *sampling around* $\pi_{\mathrm{ref}}$ *naturally preserves the reference policy's capabilities*, mitigating catastrophic forgetting while still improving task performance through importance weighting.

<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key insight:</strong> Sampling from $\pi_{\mathrm{ref}}$ transforms RL from a <em>data-limited</em> problem into a <em>compute-limited</em> problem, unlocking two critical advantages: <em>(1) Unlimited scalability</em>: it can generate as much data as needed at inference cost; <em>(2) Preservation of capabilities</em>: training data stays anchored around $\pi_{\mathrm{ref}}$, mitigating catastrophic forgetting while optimizing toward $\pi^*$ through importance weighting. For RLVR, the cost is even lower because verifiable rewards require no reward model—just lightweight verification functions (e.g., test execution, answer checking).
</div>

**Dimension 2: How do you weight samples?** (Theoretically grounded vs. Heuristic)

The weight formulas in Table 2 tell a revealing story about the gap between theoretical and empirical weights:

- **VAR, SPR & DFT:** These methods assume sampling from $$\pi^*$$ (implicitly through curated data). Under this assumption, the *theoretical optimal weight is 1* (uniform weighting), since samples already follow the target distribution. However, they use complex formulas involving $\pi_{\mathrm{ref}},$ value functions, or the current policy, which creates a mismatch between theory and implementation.
- **iw-SFT:** Uses importance weights $w(x,y) = q(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)$ under the assumption that the curated distribution $q$ approximates $$\pi^*$$. If this assumption holds, *weights should theoretically be close to 1*, leading to mismatched reweighing.
- **Refit:** Uses raw rewards as weights $w(x,y) = r(x,y)$, which *has a fundamental gap with the theoretical weight* $w(x,y) = \exp(r(x,y)/\beta)/Z(x)$ from a forward KL perspective. Refit's heuristic scheme may lead to unstable training, especially when rewards have large dynamic ranges across different prompts.
- **DGO:** The *only method where theoretical and empirical weights align perfectly*. By sampling from $\pi_{\mathrm{ref}}$ and using $w(x,y) = \exp(r/\beta)/Z(x)$, DGO implements the exact importance weight needed to transform $\pi_{\mathrm{ref}}$ samples into $$\pi^*$$ samples. This combines (1) explicit reference sampling, (2) raw unfiltered data, and (3) theoretical weighting with no theory-practice gap. DGO's weight computation is also more computationally efficient, requiring only Monte Carlo estimation via one pass over $N$ samples, unlike VAR, DFT, and iw-SFT (policy probability computation), or SPR (value function evaluation).

<div style="background-color: rgba(0, 128, 255, 0.05); padding: 1em 1.2em; border-radius: 8px; border: 1px solid rgba(0, 128, 255, 0.5);">
<strong>Key insight:</strong> <em>DGO closes the theory-practice gap by explicitly accounting for the sampling distribution</em> through theoretically-grounded importance weights $w(x,y) = \exp(r/\beta)/Z(x)$, ensuring that what we optimize in practice exactly matches what theory prescribes.
</div>

## Empirical Comparison

To validate our theoretical framework and demonstrate DGO's practical advantages, we conduct comprehensive experiments on mathematical reasoning tasks using the GSM8K <d-cite key="cobbe2021training"></d-cite> and MATH <d-cite key="hendrycks2021measuring"></d-cite> benchmarks. Our experimental setup uses Qwen3-8B <d-cite key="yang2025qwen3"></d-cite> as the base model with LoRA fine-tuning <d-cite key="hu2022lora"></d-cite>, comparing DGO against representative baselines across three dimensions that directly map to our theoretical claims: (1) Sample efficiency and performance (does theoretically-grounded weighting improve learning?), (2) Resource efficiency (does decoupling reduce computational and memory costs?), and (3) Capability preservation (does sampling from $\pi_{\mathrm{ref}}$ mitigate catastrophic forgetting?).

### Experimental Setup

**Datasets.** We evaluate on two mathematical reasoning benchmarks: GSM8K (grade school math, 7,473 training problems, max completion length 1024) and MATH (high school competition problems, 12,000 training problems, max completion length 2048). For each prompt, we generate $N=8$ rollout responses from the reference policy and evaluate them using exact-match verification against ground-truth answers, a canonical example of verifiable rewards.

**Baselines.** We compare against five representative methods spanning the design space:
- **SFT:** Standard supervised fine-tuning on ground-truth demonstrations (imitation learning).
- **VAR:** Variational alignment reweighting on curated positive samples (implicit $$\pi^*$$ sampling).
- **GRPO:** Group relative policy optimization with online rollouts (online RL baseline).
- **Refit:** Reward-weighted fine-tuning using raw rewards as weights (heuristic weighting).
- **DGO (Ours):** Decoupled generation & optimization with theoretically-grounded weights.

**Implementation.** All methods use LoRA (rank 8, alpha 64, dropout 0.05) with AdamW optimizer (lr=5e-6, weight decay=0.1, warmup ratio=0.1, cosine lr scheduler). SFT and VAR train on curated demonstrations; GRPO, Refit, and DGO generate $N=8$ rollouts per prompt. GRPO performs online optimization with reference model loaded; Refit and DGO perform offline optimization without reference model. We set temperature $\beta=0.1$ for DGO's importance weights. All methods train for the same number of gradient steps to ensure fair comparison.


### Main Results: Performance and Efficiency

We now present a comprehensive empirical comparison that directly tests our theoretical claims. Table 3 compares DGO against representative baselines across both performance and resource efficiency dimensions, revealing how our framework translates from theory to practice.

<table style="width:100%; border-collapse: collapse; margin: 1.2em 0 1.5em 0; font-size: 0.95rem; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">
  <caption style="caption-side: top; text-align: left; font-weight: 600; margin-bottom: 0.4em; text-align: center;">
    Table 3. Performance and resource efficiency on GSM8K and MATH benchmarks.
  </caption>
  <thead>
    <tr style="background-color: rgba(0, 128, 0, 0.05); font-weight: 600; text-align: left;">
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 20%;">Method</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 28%;">Data Source</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 18%;">Accuracy</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 17%;">Time Cost</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 17%;">Peak Memory</th>
    </tr>
  </thead>
  <tbody>
    <!-- GSM8K section header -->
    <tr>
      <td colspan="5" style="padding: 9px 10px; border-bottom: 1px solid #ddd; font-weight: 600; background-color: rgba(0, 0, 0, 0.02); letter-spacing: 0.02em; text-transform: uppercase; font-size: 0.9rem; text-align: center;">GSM8K</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Baseline</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">—</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">87.72%</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">—</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">—</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">SFT</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Demonstrations</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">88.40% (+0.68)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">2.85h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">57.59G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">VAR</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Demonstrations</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">88.64% (+0.92)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">6.46h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">58.39G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">GRPO</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Rollouts (Online)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">90.01% (+2.29)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">45.35h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">82.95G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Refit</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Rollouts (Offline)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">89.23% (+1.51)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">6.96h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">57.24G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">DGO (Ours)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Rollouts (Offline)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">90.67% (+2.95)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">6.79h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">57.23G</td>
    </tr>

    <!-- MATH section header -->
    <tr>
      <td colspan="5" style="padding: 9px 10px; border-bottom: 1px solid #ddd; font-weight: 600; background-color: rgba(0, 0, 0, 0.02); letter-spacing: 0.02em; text-transform: uppercase; font-size: 0.9rem; text-align: center;">MATH</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Baseline</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">—</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">55.80%</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">—</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">—</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">SFT</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Demonstrations</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">57.26% (+1.46)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">5.93h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">70.68G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">VAR</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Demonstrations</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">60.17% (+4.37)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">12.82h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">71.24G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">GRPO</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Rollouts (Online)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">61.37% (+5.57)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">55.68h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">83.96G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Refit</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Rollouts (Offline)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">58.36% (+2.56)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">14.31h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">69.37G</td>
    </tr>
    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">DGO (Ours)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Rollouts (Offline)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">61.96% (+6.16)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">13.79h</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight: 600;">68.96G</td>
    </tr>
  </tbody>
</table>

The empirical results presented in Table 3 provide strong evidence for the theoretical principles underlying DGO, demonstrating that our framework achieves superior performance while maintaining computational efficiency across both benchmarks. We summarize the main observations as follows:

- **DGO achieves the best task performance while maintaining resource efficiency.** On GSM8K, DGO reaches 90.67% accuracy (+2.95% over baseline), outperforming GRPO (90.01%). On the more challenging MATH benchmark, DGO achieves 61.96% accuracy (+6.16%), surpassing both GRPO (61.37%) and VAR (60.17%) with significantly lower resource requirements. This validates our core claim: *theoretically-grounded importance weighting enables better learning from the same data*.

- **Decoupling generation from optimization unlocks dramatic efficiency gains.** Comparing GRPO (online, coupled) vs. DGO (offline, decoupled) with identical rollout budgets ($N=8$), DGO reduces training time by *85% on GSM8K* and *75% on MATH* while cutting peak memory by *31% on GSM8K* and *18% on MATH*. This directly validates our architectural insight: *the memory overhead and slowdown of online RLVR are not fundamental; they stem from unnecessary coupling*.

- **Theoretically-grounded weighting outperforms heuristics.** DGO consistently outperforms Refit (which uses raw rewards $r(x,y)$ as weights) by *+1.44% on GSM8K* and *+3.60% on MATH*, despite both methods using identical rollout data and offline optimization. This gap directly reflects the value of DGO's principled importance weights $w(x,y) = \exp(r/\beta)/\hat{Z}(x)$, which correctly normalize across prompts and apply temperature scaling. *Theory-practice alignment matters*.

- **Demonstration quality vs. data scalability trade-off.** VAR achieves strong results on MATH (60.17%) by leveraging high-quality curated demonstrations, but is fundamentally bottlenecked by dataset size: performance is capped by what humans have annotated. In contrast, DGO's ability to generate unlimited rollouts from $\pi_{\mathrm{ref}}$ enables it to surpass VAR (+1.79% on MATH) while maintaining scalability. On GSM8K where demonstrations are more abundant, the gap is smaller but DGO still leads, demonstrating *compute-limited scaling beats data-limited scaling*.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-scaling-rlvr/gsm8k.png" class="img-fluid rounded" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-scaling-rlvr/math.png" class="img-fluid rounded" %}
    </div>
</div>
<div class="caption">
    <strong>Figure 1. Convergence comparison on GSM8K and MATH benchmarks.</strong>
</div>

<strong>Convergence dynamics.</strong> Figure 1 tracks convergence across 1000 training steps. DGO exhibits the fastest and most stable convergence. Demonstration-based methods (SFT, VAR) show slower, erratic convergence, reflecting the mismatch between their implicit $$\pi^*$$ (defined by demonstrations) and the KL-regularized objective when demonstrations deviate from $$\pi_{\text{ref}}$$. GRPO converges more slowly than DGO despite identical rollout data, likely due to variance from online importance sampling. Refit shows intermediate behavior, confirming that raw reward weighting lacks DGO's theoretically-grounded normalization. These dynamics demonstrate that theoretically-grounded importance weights enable faster, more stable convergence than either demonstration-based methods or heuristic alternatives.

### Catastrophic Forgetting Analysis

To test our claim that sampling from $\pi_{\mathrm{ref}}$ preserves general capabilities, we evaluate all GSM8K-trained models on four out-of-domain benchmarks: PIQA (physical commonsense) <d-cite key="bisk2020piqa"></d-cite>, HellaSwag (commonsense reasoning) <d-cite key="zellers2019hellaswag"></d-cite>, Winogrande (pronoun resolution) <d-cite key="sakaguchi2021winogrande"></d-cite>, and RACE-high (reading comprehension) <d-cite key="lai2017race"></d-cite>.

<table style="width:100%; border-collapse: collapse; margin: 1.2em 0 1.5em 0; font-size: 0.95rem; font-family: 'Palatino', 'Palatino Linotype', 'Book Antiqua', serif;">
  <caption style="caption-side: top; text-align: left; font-weight: 600; margin-bottom: 0.4em;; text-align: center;">
    Table 4. Out-of-domain performance after GSM8K training (measuring catastrophic forgetting).
  </caption>
  <thead>
    <tr style="background-color: rgba(0, 128, 0, 0.05); font-weight: 600; text-align: center;">
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 20%;">Method</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 20%; text-align: center;">PIQA</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 20%; text-align: center;">HellaSwag</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 20%; text-align: center;">Winogrande</th>
      <th style="padding: 10px; border-bottom: 1px solid #ccc; width: 20%; text-align: center;">RACE-high</th>
    </tr>
  </thead>
  <tbody>

    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Baseline</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">71.22</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">81.80</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">65.19</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">79.25</td>
    </tr>

    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">SFT</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">70.35 (−0.87)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">80.56 (−1.24)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">63.14 (−2.05)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">84.33 (+5.08)</td>
    </tr>

    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">VAR</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">70.26 (−0.96)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">80.73 (−1.07)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">63.24 (−1.95)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">82.13 (+2.88)</td>
    </tr>

    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">GRPO</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">70.69 (−0.53)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">81.03 (−0.77)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">64.26 (−0.93)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">80.26 (+1.01)</td>
    </tr>

    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee;">Refit</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">70.53 (−0.69)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center; font-weight:600;">81.26 (−0.54)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">64.01 (−1.18)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">80.11 (+0.86)</td>
    </tr>

    <tr>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; font-weight:600;">DGO (Ours)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center; font-weight:600;">71.01 (−0.21)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center;">81.13 (−0.67)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center; font-weight:600;">64.94 (−0.25)</td>
      <td style="padding: 8px 10px; border-bottom: 1px solid #eee; text-align: center; font-weight:600;">79.93 (+0.68)</td>
    </tr>

  </tbody>
</table>

Table 4 reports out-of-domain benchmark performance after fine-tuning on GSM8K. The results validate that DGO's explicit sampling from $\pi_{\mathrm{ref}}$ provides inherent regularization against catastrophic forgetting. We highlight the following key observations:

- **DGO exhibits minimal forgetting across general benchmarks.** DGO shows the smallest performance drops on PIQA (−0.21%) and Winogrande (−0.25%), with near-baseline retention on RACE-high (+0.68%). In contrast, demonstration-based methods (SFT, VAR) suffer larger degradation, particularly on Winogrande (−2.05% and −1.95%), validating our theoretical insight: *sampling from* $\pi_{\mathrm{ref}}$ *keeps training data anchored around the reference distribution, naturally preserving its capabilities*.

- **Demonstration-based methods (SFT, VAR) show unexpected gains on RACE-high.** The large positive shifts (+5.08% for SFT, +2.88% for VAR) on reading comprehension likely reflect dataset artifacts: GSM8K demonstrations contain rich mathematical reasoning narratives that transfer positively to RACE-high's comprehension format. However, this comes at the cost of larger drops on other tasks, suggesting *overfitting to demonstration distribution rather than robust capability preservation*.

- **Online RL (GRPO) achieves balanced retention.** GRPO's moderate forgetting across all tasks (−0.53% to −0.93% on most benchmarks) suggests that KL regularization provides some protection against catastrophic forgetting, but at massive computational cost (45.35h vs. 6.79h for DGO). *DGO matches or exceeds GRPO's forgetting resistance with 4-6× speedup*.

- **All rollout-based methods (GRPO, Refit, DGO) avoid overfitting.** Unlike demonstration-based approaches, methods that generate diverse rollouts from $\pi_{\mathrm{ref}}$ show minimal anomalous gains, maintaining near-baseline performance on out-of-domain tasks. This confirms that *explicit* $\pi_{\mathrm{ref}}$ *sampling not only enables scalability but also provides implicit regularization* against distribution shift.

### Summary

Our empirical results comprehensively validate the theoretical claims established in previous sections:

- **Theoretically-grounded weighting improves learning:** DGO's importance weights $w(x,y) = \exp(r/\beta)/\hat{Z}(x)$ consistently outperform heuristic alternatives (Refit) and match or exceed online RL (GRPO) in task performance.

- **Decoupling eliminates architectural bottlenecks:** By separating generation from optimization, DGO achieves *4-6× speedup* and *18-31% memory reduction* compared to online RL, proving the coupling is unnecessary.

- **Sampling from** $\pi_{\mathrm{ref}}$ **preserves capabilities:** DGO exhibits minimal catastrophic forgetting (≤0.67% drop on most benchmarks), validating that explicit reference sampling provides implicit regularization.

- **Compute-limited beats data-limited:** DGO surpasses demonstration-based methods (VAR) despite using generated rollouts, demonstrating that unlimited scalability through $\pi_{\mathrm{ref}}$ sampling outweighs curated dataset quality.

Together, these results establish DGO as the first practical implementation where *theory, scalability, and performance converge*, proving that online RLVR can indeed be scaled when done right through principled decoupling and theoretically-grounded design.

---

# Conclusion

The computational barriers limiting online RLVR are *architectural, not fundamental*. By reformulating online RL as offline weighted SFT through forward KL, DGO resolves this: decoupling generation from optimization with theoretically-grounded importance weights $w(x,y) = \exp(r/\beta)/\hat{Z}(x)$ achieves *4-6× speedup, 18-31% memory reduction, and superior performance* while preserving capabilities. Unlike existing methods that sample from implicit optimal policies or use heuristic weights, *DGO perfectly aligns theory with implementation*: what we optimize matches what theory prescribes. For RLVR, verifiable rewards eliminate reward hacking and remove reward model overhead. *Scaling online RLVR is achievable when done right*: through principled decoupling, theoretical grounding, and explicit reference sampling.
