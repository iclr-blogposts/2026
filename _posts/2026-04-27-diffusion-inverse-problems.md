---
layout: distill
title: Navigating the Manifold — A Geometric Perspective on Diffusion-Based Inverse Problems
description: This blogpost develops a geometric and probabilistic lens on diffusion priors for inverse problems. We show that a wide range of methods mostly instantiate two operator-splitting paradigms, i.e., posterior-guided sampling and clean-space local-MAP optimization. Through manifold diagrams, Tweedie-based animations, and step-by-step derivations, we explain how these paradigms decouple a pretrained diffusion prior from measurement physics, clarify when they approximate full posterior sampling versus MAP estimation, and distill practical design rules for building robust diffusion-based inverse solvers. 
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
#  - name: Albert Einstein
#    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
#    affiliations:
#      name: IAS, Princeton
#  - name: Boris Podolsky
#    url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#    affiliations:
#      name: IAS, Princeton
#  - name: Nathan Rosen
#    url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#    affiliations:
#      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-diffusion-inverse-problems.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 1. Background
  - name: 2. Diffusion priors for inverse problems
  - name: 3. Posterior-Guided Sampling Paradigm
  - name: 4. Local-MAP optimization Paradigm
  - name: 5. Conclusion


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
  .box-note {
        font-size: 18px;
        padding: 15px 15px 10px 10px;
        margin: 2px 2px 2px 5px;
        border-left: 7px solid #1976d2;
        border-radius: 1px;
    }
    d-article .box-note {
        background-color: #f5f9ff;
        border-left-color: #1976d2;
    }
---

## 1. Background
Inverse problems aim to recover an unknown signal $$\mathbf{x} \in \mathbb{R}^n$$ from indirect and noisy measurements $$\mathbf{y} \in \mathbb{R}^m$$. A large class of problems in imaging, signal processing, and computational physics can be written—at least approximately—as a linear–Gaussian observation model

$$
\mathbf{y} = \mathbf{A}\mathbf{x} + \mathbf{n}, \qquad \mathbf{n} \sim \mathcal{N}(0, \sigma^2 \mathbf{I})\tag{1.1}\label{eq:ip},
$$

where $$\mathbf{A}$$ encodes the physics or geometry of the measurement process (e.g., convolution with a blur kernel, subsampled Fourier transform, Radon transform), and $$\mathbf{n}$$ models sensor noise and modeling errors. The fundamental difficulty is that many such problems are **ill-posed**:

- **Non-uniqueness:** if $$\mathbf{A}$$ is rank-deficient or severely underdetermined, many different $$\mathbf{x}$$ produce the same $$\mathbf{y}$$.
  
- **Instability:** when $$\mathbf{A}$$ is ill-conditioned, small perturbations in $$\mathbf{y}$$ can cause large changes in the reconstructed $$\mathbf{x}$$.

If we ignore prior knowledge and rely solely on the measurements to solve via maximum likelihood estimation

$$
\hat{\mathbf{x}}_{\text{MLE}} = \arg\min_{\mathbf{x}} \frac{1}{2} \|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2,
$$

it yields the solution either does not exist, is not unique, or is extremely sensitive to noise. This motivates **regularization**: we bias the solution toward signals that are "plausible" under our prior knowledge. In probabilistic form, this leads to **Maximum A Posteriori (MAP)** estimation

$$
\hat{\mathbf{x}}_{\text{MAP}} = \arg\min_{\mathbf{x}} \left[ \underbrace{\frac{1}{2\sigma^2} \|\mathbf{y} - \mathbf{A}\mathbf{x}\|_2^2}_{\text{Data Consistency}} \quad + \underbrace{\lambda R(\mathbf{x})}_{\text{Prior / Regularization}} \right],
$$

where $$R(\mathbf{x})$$ is a regularizer encoding the prior, and $$\lambda > 0$$ controls the trade-off between data fidelity and prior strength.




From this perspective, the history of solving inverse problems is essentially the history of designing better functions for $R(\mathbf{x})$ — from handcrafted mathematical assumptions to data-driven generative manifolds.



### 1.1. Handcrafted Analytical Priors

Before  the era of deep learning, priors were designed based on mathematical and statistical intuition about natural signals.

- **Smoothness priors (Tikhonov / Gaussian):**  
  Assuming that $$\mathbf{x}$$ varies smoothly, one uses $$R(\mathbf{x}) = \|\mathbf{L}\mathbf{x}\|_2^2$$ for some differential or finite-difference operator $$\mathbf{L}$$. This corresponds to a Gaussian prior and leads to classical Tikhonov regularization <d-cite key="tikhonov1977solutions"></d-cite>. Such priors are stable and convex, but they oversmooth edges and cannot hallucinate missing high-frequency details.

- **Total Variation (TV) and piecewise-smooth priors:**  
  To better preserve edges, TV replaces the quadratic penalty with an $$\ell_1$$-type penalty on gradients, e.g., $$R(\mathbf{x}) = \|\nabla \mathbf{x}\|_1.$$ TV assumes that images are mostly smooth with sparse sharp transitions, which works well for piecewise-constant structures but can introduce “staircase” artifacts and still struggles with fine textures.<d-cite key="rudin1992nonlinear"></d-cite>

- **Sparsity priors (wavelets, dictionary learning):**  
  Another line of work assumes that $$\mathbf{x}$$ is sparse in a suitable transform domain, such as wavelets or learned dictionaries, leading to penalties like $$R(\mathbf{x}) = \|\mathbf{W}\mathbf{x}\|_1,$$ where $$\mathbf{W}$$ is a fixed or learned transform. These priors capture certain classes of textures and edges, but still act as **handcrafted surrogates** for the true (and highly complex) distribution of natural signals.

  
All of these classical priors are **analytical**: they are given by explicit formulas chosen by the designer. They are computationally convenient and theoretically well understood, but ultimately limited in expressiveness. They can discourage obviously “bad” solutions, yet they do not truly know what realistic images look like.


### 1.2. Explicit Generative Manifolds

With the rise of deep generative models, we no longer had to **guess** analytical forms for $$R(\mathbf{x})$$. Instead, we could **learn** the data distribution directly from examples, and then use that model as a prior.

A common idea is the **manifold constraint**: assume that plausible signals lie near the range of a generator $$G(\mathbf{z})$$, where $$\mathbf{z}$$ is a low-dimensional latent code. Rather than optimizing directly over $$\mathbf{x}$$, we optimize over $$\mathbf{z}$$:

$$
\hat{\mathbf{z}}
= \arg\min_{\mathbf{z}}
\left[
  \|\mathbf{y} - \mathbf{A}(G(\mathbf{z}))\|_2^2
  + \lambda \|\mathbf{z}\|_2^2
\right],
\qquad
\hat{\mathbf{x}} = G(\hat{\mathbf{z}}).
$$

Different generative models instantiate this idea in different ways:

*   **GANs <d-cite key="goodfellow2014generative"></d-cite>:** 
    Adversarial training yields highly realistic samples and sharp details. However, mode collapse and training instability make the learned manifold incomplete: if the ground-truth $$\mathbf{x}$$ is not well covered by the generator, optimization in $$\mathbf{z}$$ may fail or converge to a visually plausible but incorrect solution.

*   **VAEs <d-cite key="kingma2013auto"></d-cite>:** 
    VAEs provide a probabilistic decoder $$p_\theta(\mathbf{x}\mid \mathbf{z})$$ with a simple prior on $$\mathbf{z}$$ (often Gaussian). Their reconstructions are typically smoother and more stable than GANs but tend to be **blurry**, reflecting the ELBO objective’s tendency to average over modes.

*   **Normalizing Flows <d-cite key="dinh2016density,kingma2018glow"></d-cite> and related models:** 
    Flows provide exact likelihoods and invertible mappings, so in principle they offer a very clean way to define $$p(\mathbf{x})$$. In practice, however, they impose architectural constraints (e.g., invertibility and tractable Jacobians), and solving inverse problems with them often requires expensive Jacobian–vector products or MCMC steps.

In all these cases, the prior is no longer an explicit penalty $$R(\mathbf{x})$$ but an **implicit manifold** traced out by a neural generator. This brings much richer structure but also introduces new optimization and modeling challenges: enforcing measurement consistency while staying on (or near) the learned manifold is nontrivial.

### 1.3. The Shift to Diffusion Prior

Both previous families of priors have clear trade-offs:

- Classical analytical priors are **stable, convex, and easy to optimize**, but they are too simple to faithfully model complex natural images.
- Explicit generative manifolds are **expressive** and can synthesize highly realistic samples, but they are often **hard to optimize against** and can be brittle outside the training distribution.

Diffusion models <d-cite key="ho2020denoising,song2020score"></d-cite> offer a different compromise. Instead of defining a single deterministic manifold or a closed-form regularizer, they define a **stochastic generative process** that gradually transforms simple noise into data through a sequence of Gaussian denoising steps. Conceptually, they give us:

- A powerful, flexible **prior over signals**, encoded by a learned score or denoiser.
- A natural way to **interleave prior-driven denoising with measurement-driven corrections** along the sampling trajectory.

This "dynamic" view of the prior—where we guide a stochastic or deterministic trajectory through noisy space—makes diffusion models particularly attractive for inverse problems. It allows us to inject measurement information **gradually**, to decouple a pretrained diffusion prior from the forward operator $$\mathbf{A}$$, and to design algorithms that behave more like **operator-splitting schemes** than hard manifold projection. The rest of this post develops this perspective in detail.




## 2. Diffusion priors for inverse problems

In the background section we framed inverse problems as Bayesian inference with a prior over clean signals $$\mathbf{x}_0$$ and a likelihood induced by the forward operator $$\mathbf{A}$$. With a diffusion model, this prior is no longer given by an explicit energy $$R(\mathbf{x}_0)$$ or a simple manifold parameterization $$G(\mathbf{z})$$. Instead, it is encoded in a **stochastic generative process** that gradually transforms Gaussian noise into data. This section explains how that process works and why it makes inverse problems both powerful and subtle.


### 2.1. Bayesian “gold standard” with a diffusion prior

Consider again the inverse problem where the measurement is generated from a forward model $$\mathbf{y} \sim p(\mathbf{y}\mid \mathbf{x}_0),$$ with $$\mathbf{x}_0 \sim p(\mathbf{x}_0)$$ given by a diffusion prior. In the simplest case we assume additive Gaussian noise (Eq. \eqref{eq:ip}), so that

$$
p(\mathbf{y}\mid \mathbf{x}_0) \propto \exp\Big(-\frac{1}{2\sigma^2}\|\mathbf{y}-\mathbf{A}\mathbf{x}_0\|_2^2\Big).
$$

By Bayes’ rule, the posterior over clean signals is

$$
p(\mathbf{x}_0\mid \mathbf{y})
\;\propto\; p(\mathbf{y}\mid\mathbf{x}_0)\,p(\mathbf{x}_0).
$$

If we could evaluate $$\log p(\mathbf{x}_0)$$ and its gradient exactly, the **Bayesian gold standard** would be the MAP estimator

$$
\hat{\mathbf{x}}_{\text{MAP}}
= \arg\max_{\mathbf{x}_0}
\left[
  \underbrace{\log p(\mathbf{y}\mid \mathbf{x}_0)}_{\text{Data fidelity}}
  +
  \underbrace{\log p(\mathbf{x}_0)}_{\text{Prior}}
\right].
\tag{2.1}\label{eq:obj}
$$

Conceptually this is straightforward: we simply trade off how well $$\mathbf{x}_0$$ explains the measurement $$\mathbf{y}$$ versus how plausible $$\mathbf{x}_0$$ is under the diffusion prior. In practice, diffusion models make this optimization non-trivial, because the prior is not defined directly in the clean space $$\mathbf{x}_0$$.



### 2.2. Diffusion models live in noisy space

A modern diffusion model is trained in a **noisy** space. One common continuous-time formulation uses a forward SDE

$$
d\mathbf{x}_t = f(\mathbf{x}_t,t)\,dt + g(t)\,d\mathbf{w}_t,
\qquad t\in[0,1],
$$

where $$\mathbf{x}_0$$ is a clean data sample, $$\mathbf{x}_1$$ is approximately Gaussian noise, and $$\mathbf{w}_t$$ is standard Brownian motion. The forward SDE defines a family of intermediate marginals $$p_t(\mathbf{x}_t)$$ that smoothly interpolate between the data distribution at $$t=0$$ and a simple reference (typically Gaussian) at $$t=1$$. Training learns a **score network**

$$
s_\theta(\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t} \log p_t(\mathbf{x}_t),
$$

i.e., a vector field that points in the direction of steeper log-density at each noise level. Geometrically, the score pushes samples uphill towards regions of higher probability. At generation time, we can run the **reverse-time SDE** <d-cite key="anderson1982reverse"></d-cite>

$$
d\mathbf{x}_t
=
\Big[f(\mathbf{x}_t,t) - g^2(t)\,s_\theta(\mathbf{x}_t,t)\Big]\,dt
+ g(t)\,d\bar{\mathbf{w}}_t,
$$

from $$t=1$$ (pure noise) to $$t=0$$ (clean data), where $$\bar{\mathbf{w}}_t$$ is a standard Brownian motion in reversed time. Alternatively, Song et al. <d-cite key="song2020score"></d-cite> showed that there exists a deterministic **probability-flow ODE (PF-ODE)**

$$
\frac{d\mathbf{x}_t}{dt}
=
f(\mathbf{x}_t,t)
-
\frac{1}{2}g^2(t)\,\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t),
$$

whose solution curves share the same marginals $$\{p_t\}_{t\in[0,1]}$$ as the stochastic SDE. Intuitively, the SDE adds randomness while the ODE simply transports probability mass along the same flow.

Crucially, all of these objects – the forward SDE, the reverse SDE, and the PF-ODE – are defined in terms of the **noisy** variables $$\mathbf{x}_t$$ and their densities $$p_t(\mathbf{x}_t)$$. The prior $$p(\mathbf{x}_0)$$ only appears implicitly as the marginal at time zero of this process. This creates a mismatch for inverse problems:

- The likelihood $$p(\mathbf{y}\mid \mathbf{x}_0)$$ in Eq. \eqref{eq:obj} is naturally written in the **clean space** $$\mathbf{x}_0$$.
  
- The diffusion prior is only directly accessible through scores $$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$$ in the **noisy space** $$\mathbf{x}_t$$.

The rest of this blogpost can be read as a sequence of techniques for bridging this gap: how to couple measurement information with a prior that lives along a diffusion trajectory.


### 2.3. Tweedie’s estimator as a clean-space anchor

A key tool for connecting noisy and clean spaces is **Tweedie’s formula**. In the simplest (and widely used) setting, the forward diffusion at time $$t$$ can be written as

$$
\mathbf{x}_t
=
\alpha(t)\,\mathbf{x}_0
+
\sigma(t)\,\boldsymbol{\epsilon},
\qquad
\boldsymbol{\epsilon} \sim \mathcal{N}(0,\mathbf{I}),
$$

where $$\alpha(t)$$ and $$\sigma(t)$$ are scalar functions that control the signal and noise levels. Under this Gaussian corruption model, Tweedie’s formula tells us that the posterior mean of the clean signal given a noisy observation $$\mathbf{x}_t$$ is

$$
\hat{\mathbf{x}}_0(\mathbf{x}_t,t)
\;=\;
\mathbb{E}[\mathbf{x}_0\mid\mathbf{x}_t]
\;=\;
\frac{\mathbf{x}_t + \sigma^2(t)\,s_\theta(\mathbf{x}_t,t)}{\alpha(t)}.
$$

Here $$s_\theta(\mathbf{x}_t,t) \approx \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$$ is the score learned by the diffusion model. Thus, once the diffusion prior is trained, we can *decode* any noisy point $$\mathbf{x}_t$$ into a corresponding **clean-space estimate** $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$ using only the score network.

From a statistical perspective, $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$ is the **minimum mean-squared error (MMSE)** estimator of $$\mathbf{x}_0$$ given $$\mathbf{x}_t$$: among all possible functions of $$\mathbf{x}_t$$, it minimizes the expected squared error

$$
\mathbb{E}\big[\|\mathbf{x}_0 - \hat{\mathbf{x}}_0(\mathbf{x}_t,t)\|_2^2\big].
$$

It is therefore a principled, Bayes-optimal representative of the posterior distribution over $$\mathbf{x}_0$$ for that particular noise level. From a geometric perspective, we can interpret $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$ as a kind of **clean-space anchor**:

- Let $$M_0$$ denote the (unknown) data manifold where the diffusion prior assigns high density at $$t=0$$.
- For a noisy point $$\mathbf{x}_t$$, the posterior over $$\mathbf{x}_0$$ given $$\mathbf{x}_t$$ is typically concentrated near a small region on or near $$M_0$$.
- Tweedie’s estimator $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$ sits near the *center of mass* of that posterior cloud. It lives in a high-density neighborhood of $$M_0$$, even if it is not exactly on the manifold, and becomes increasingly accurate as $$\sigma(t)\to 0$$.

We will repeatedly exploit this anchor in later sections:

- In **posterior-guided sampling** (Section 3), we use Tweedie’s estimator to approximate *posterior* scores in noisy space, enabling us to modify the reverse SDE/ODE so that it samples from an approximation to $$p(\mathbf{x}_0\mid\mathbf{y})$$.
  
- In **clean-space Local-MAP optimization** (Section 4), we treat $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$ as a good initialization in clean space and locally refine it to better satisfy the measurement constraints.

<figure class="figure">
  <img 
    src="{{'/assets/img/2026-04-27-diffusion-inverse-problems/tweedie_anchor_sine.gif' | relative_url }}"
    alt="Posterior-guided sampling: prior vs naive guidance vs Jacobian guidance"
    style="max-width: 100%; height: auto; image-rendering: auto;">
</figure>
<div class="caption">
    Animation 1: Tweedie denoising as a clean-space anchor.
</div>

**Animated illustration – posterior mean vs. data manifold.**  
Consider a simple 2D sine-shaped data manifold $$M_0$$ and a single clean point $$\mathbf{x}_0$$ on it. When we apply the forward diffusion to $$\mathbf{x}_0$$, we obtain noisy samples $$\mathbf{x}_t$$ at different noise levels:

- At **high noise**, the posterior density $$p(\mathbf{x}_0\mid\mathbf{x}_t)$$ spreads along a large portion of the curve, and Tweedie’s estimator lies somewhere near the middle of this elongated cloud.
- As the noise **decreases**, the posterior collapses around the true point on $$M_0$$, and $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$ moves closer and closer to the manifold.

- The green semi-transparent disk illustrates an approximate posterior uncertainty region around the Tweedie anchor: as noise decreases, the posterior mass concentrates and the disk shrinks.
  
This visualization helps explain why Tweedie’s estimator is a useful building block for inverse problems with diffusion priors: even though it is not a strict orthogonal projection onto $$M_0$$, it provides a reliable and differentiable anchor in the clean space that we can refine using the measurement model.

In the next sections, we show how this anchor, together with the diffusion trajectory in noisy space, gives rise to two practical paradigms for solving inverse problems: **posterior-guided sampling** in noisy space and **clean-space Local-MAP optimization**.














## 3. Posterior-Guided Sampling Paradigm



In this section, we focus on **posterior-guided sampling**, a family of methods that solve inverse problems by directly sampling from the Bayesian posterior under a diffusion prior, exemplified by DPS <d-cite key="chung2022diffusion"></d-cite> and MCG methods <d-cite key="chung2022improving"></d-cite>. The key idea is extremely simple:

<aside class="l-body box-note" markdown="1">
Take the reverse-time dynamics of the diffusion model, and replace the <strong>prior score</strong> $$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$$ with the <strong>posterior score</strong> $$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid \mathbf{y})$$.
</aside>



Concretely, recall the probability-flow ODE (PF-ODE) associated with the forward SDE

$$
\frac{d\mathbf{x}_t}{dt}
=
f(\mathbf{x}_t,t)
-\frac{1}{2}g^2(t)\,s_\theta(\mathbf{x}_t,t),
\qquad
s_\theta(\mathbf{x}_t,t)\approx \nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t),
$$

whose solution at $$t=0$$ recovers samples from the data distribution $$p_{\text{data}}$$. Posterior-guided sampling modifies this ODE by swapping in the **posterior score**:

$$
\frac{d\mathbf{x}_t}{dt}
=
f(\mathbf{x}_t,t)
-
\frac{1}{2}g^2(t)
\underbrace{\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid \mathbf{y})}_{\text{posterior score}}.
\tag{3.1}\label{eq:pgs}
$$

By Bayes’ rule, the posterior score decomposes as

$$
\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid \mathbf{y}) =
\underbrace{\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)}_{\text{prior score}}
+
\underbrace{\nabla_{\mathbf{x}_t}\log p(\mathbf{y}\mid \mathbf{x}_t)}_{\text{likelihood / data-consistency score}}.
\tag{3.2}\label{eq:posterior_score}
$$

Intuitively, the **prior score** pulls trajectories back to high-density regions of the diffusion prior, while the **likelihood score** pushes them toward measurements that satisfy the forward model. 

Figure 1 visualizes these three vector fields at an intermediate time $$t$$: prior score (blue), likelihood score (orange), and their sum—the posterior score (black). The right panel sketches how the resulting trajectories land in the intersection between the diffusion prior manifold $$\mathcal{M}_{\mathrm{DM}}$$ and the observation-consistent manifold $$\mathcal{M}_{\mathrm{obs}}$$, forming a posterior manifold $$\mathcal{M}_{\mathrm{pos}}$$.


{% include figure.liquid
   path="assets/img/2026-04-27-diffusion-inverse-problems/posterior_guided.jpg"
   class="img-fluid"
   caption="Figure 1. Posterior-guided sampling. At intermediate time $t$, the prior score (blue) pulls samples back to the diffusion prior manifold, the likelihood score (orange) pushes them toward measurement-consistent states, and their sum (black) is the posterior score. Integrating the reverse-time PF-ODE with this posterior score yields samples on the posterior manifold $\mathcal{M}_{\mathrm{pos}} = \mathcal{M}_{\mathrm{DM}}\cap \mathcal{M}_{\mathrm{obs}}$ at $t=0$."
%}

The rest of this section explains:

1. **Why** integrating Eq. \eqref{eq:pgs} indeed samples from $$p(\mathbf{x}_0\mid \mathbf{y})$$;
2. **How** to approximate the intractable likelihood score $$\nabla_{\mathbf{x}_t}\log p(\mathbf{y}\mid \mathbf{x}_t)$$ using the Tweedie clean-space anchor introduced in the previous section;
3. **What** the Jacobian in this approximation does geometrically, and why it acts as a projection onto the manifold’s tangent space;
4. A **2D toy illustration** that makes these vector fields visible.



### 3.1. Why Inverse Problems Can Be Solved by Sampling

We first revisit why **sampling** from a suitable reverse-time process solves not only the unconditional generative problem, but also the **inverse problem**.

We begin with the **unconditional** case. Let $$\{ \mathbf{x}_t \}_{t\in[0,T]}$$ be a trajectory followed by the forward diffusion SDE,

$$
d\mathbf{x}_t = f(\mathbf{x}_t,t)\,dt + g(t)\,d\mathbf{w}_t,\qquad \mathbf{x}_0 \sim p_{\text{data}},
$$

with marginal densities $$p_t(\mathbf{x}_t)$$. These densities satisfy the **Fokker–Planck equation**

$$
\partial_t p_t(\mathbf{x}_t)
=
-\nabla_{\mathbf{x}_t}\!\cdot\!\big(f(\mathbf{x}_t,t)\,p_t(\mathbf{x}_t)\big)
+\frac{1}{2}\nabla_{\mathbf{x}_t}^2\!\big(g^2(t)\,p_t(\mathbf{x}_t)\big).
$$

Song et al. <d-cite key="song2020score,anderson1982reverse"></d-cite> showed that there exists both a **reverse-time SDE** and an equivalent **PF-ODE** whose marginals evolve backward in time but follow the **same** family of densities $$\{p_t\}$$. In particular, the PF-ODE

$$
\frac{d\mathbf{x}_t}{dt}
=
f(\mathbf{x}_t,t) - \frac{1}{2}g^2(t)\,\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)
$$

satisfies the same Fokker–Planck equation as the forward SDE, but with time running from $$t=1$$ back to $$t=0$$. If we start from $$\mathbf{x}_1\sim p_1\approx\mathcal{N}(0,I)$$ and integrate this PF-ODE with the **true** score, then $$\mathbf{x}_0\sim p_{\text{data}}$$.



Now, **condition** on a measurement $$\mathbf{y}$$ produced by a forward model $$\mathbf{y}\sim p(\mathbf{y}\mid \mathbf{x}_0)$$. 

Conditioning lifts to the **entire diffusion trajectory**: we obtain a posterior process $$\{\mathbf{x}_t\mid \mathbf{y}\}$$ with marginals $$p(\mathbf{x}_t\mid \mathbf{y})$$. Crucially, because conditioning does not change the forward SDE dynamics, the posterior densities satisfy the **same Fokker–Planck operator**, but with different initial and terminal marginals.

$$
\partial_t p(\mathbf{x}_t\mid \mathbf{y})
=
-\nabla_{\mathbf{x}_t}\!\cdot\!\big(f(\mathbf{x}_t,t)\,p(\mathbf{x}_t\mid \mathbf{y})\big)
+\frac{1}{2}\nabla_{\mathbf{x}_t}^2\!\big(g^2(t)\,p(\mathbf{x}_t\mid \mathbf{y})\big),
$$


By the same reasoning as in the unconditional case, there exists a **posterior PF-ODE**

$$
\frac{d\mathbf{x}_t}{dt}
=
f(\mathbf{x}_t,t)
-\tfrac{1}{2}g^2(t)\,\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid \mathbf{y}),
$$

such that, if we start at $$t=1$$ from $$\mathbf{x}_1\sim p(\mathbf{x}_1\mid \mathbf{y})$$ (which is typically close to the unconditional $$p_1$$) and integrate backward, then the marginals follow $$\{p(\mathbf{x}_t\mid \mathbf{y})\}_{t\in[0,1]}$$, and in particular

$$
\mathbf{x}_0 \sim p(\mathbf{x}_0\mid \mathbf{y}).
$$

This gives the fundamental justification behind posterior-guided sampling:

<aside class="l-body box-note" markdown="1">
If we can approximate the posterior score $$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid \mathbf{y})$$ and integrate the corresponding PF-ODE, then solving the inverse problem reduces to sampling from this posterior-guided flow.
</aside>

The only missing piece is an approximation of the **likelihood score**
$$\nabla_{\mathbf{x}_t}\log p(\mathbf{y}\mid \mathbf{x}_t)$$ in Eq. \eqref{eq:posterior_score}, which we turn to next.









### 3.2. Approximating the Likelihood Score via the Clean Space



The decomposition in Eq. \eqref{eq:posterior_score} is conceptually simple:

- The **prior score** $$s_\theta(\mathbf{x}_t,t)$$ is directly provided by the diffusion prior.
- The challenging part is the **likelihood score**: $$\nabla_{\mathbf{x}_t}\log p(\mathbf{y}\mid \mathbf{x}_t)$$.

Under the standard linear–Gaussian measurement model, the likelihood is naturally defined in the **clean space**:

$$
\log p(\mathbf{y}\mid \mathbf{x}_0)
=
-\frac{1}{2\sigma^2}\|\mathbf{y}-\mathbf{A}\mathbf{x}_0\|_2^2 + \text{const}
\quad\Rightarrow\quad
\nabla_{\mathbf{x}_0}\log p(\mathbf{y}\mid \mathbf{x}_0)
=
\frac{1}{\sigma^2}\mathbf{A}^\top(\mathbf{y}-\mathbf{A}\mathbf{x}_0).
$$

However, the diffusion model operates in the **noisy space** $$\mathbf{x}_t$$, where the noisy-space likelihood is obtained by marginalizing over $$\mathbf{x}_0$$:

$$
p(\mathbf{y}\mid \mathbf{x}_t)
=
\int p(\mathbf{y}\mid \mathbf{x}_0)\,p(\mathbf{x}_0\mid \mathbf{x}_t)\,d\mathbf{x}_0
=
\mathbb{E}_{\mathbf{x}_0\sim p(\mathbf{x}_0\mid \mathbf{x}_t)}\big[p(\mathbf{y}\mid \mathbf{x}_0)\big].
$$

This integral is intractable in high dimensions, and differentiating $$\log p(\mathbf{y}\mid \mathbf{x}_t)$$ w.r.t. $$\mathbf{x}_t$$ would require full access to the conditional distribution $$p(\mathbf{x}_0\mid \mathbf{x}_t)$$, which we only know implicitly through the score network.

Posterior-guided methods therefore make a **concentration assumption**: for a given $$(\mathbf{x}_t,t)$$, the conditional distribution $$p(\mathbf{x}_0\mid \mathbf{x}_t)$$ is sharply peaked around the Tweedie denoiser

$$
\hat{\mathbf{x}}_0(\mathbf{x}_t,t)
=
\mathbb{E}[\mathbf{x}_0\mid \mathbf{x}_t]
=
\frac{\mathbf{x}_t + \sigma^2(t)\,s_\theta(\mathbf{x}_t,t)}{\alpha(t)}.
$$

Approximating $$p(\mathbf{x}_0\mid \mathbf{x}_t)$$ by a point mass at $$\hat{\mathbf{x}}_0$$ (a delta / Laplace approximation) yields

$$
p(\mathbf{y}\mid \mathbf{x}_t)
\approx
p(\mathbf{y}\mid \hat{\mathbf{x}}_0(\mathbf{x}_t,t))
\quad\Rightarrow\quad
\log p(\mathbf{y}\mid \mathbf{x}_t)
\approx
\log p(\mathbf{y}\mid \hat{\mathbf{x}}_0(\mathbf{x}_t,t)).
$$

Differentiating this approximation with respect to $$\mathbf{x}_t$$ and applying the chain rule gives

$$
\nabla_{\mathbf{x}_t}\log p(\mathbf{y}\mid \mathbf{x}_t)
=
\Big(\frac{\partial \hat{\mathbf{x}}_0}{\partial \mathbf{x}_t}\Big)^\top
\nabla_{\mathbf{x}_0}\log p(\mathbf{y}\mid \mathbf{x}_0)
\Big|_{\mathbf{x}_0=\hat{\mathbf{x}}_0(\mathbf{x}_t,t)}.
\tag{3.3}\label{eq:lik}
$$

The likelihood score thus factorizes into:

1. a **clean-space data-consistency gradient** $$\nabla_{\mathbf{x}_0}\log p(\mathbf{y}\mid \mathbf{x}_0),$$ computable in closed form for many forward models; and

2. a **Jacobian–vector product (JVP)** involving the Tweedie denoiser $$(\tfrac{\partial \hat{\mathbf{x}}_0}{\partial \mathbf{x}_t})^\top,$$  which maps this clean-space gradient back into the noisy space.

Eq. \eqref{eq:lik} is the main technical bridge that connects the likelihood score in noisy space to a clean-space gradient plus a Jacobian term. In the next subsection we analyze what this Jacobian does geometrically.






### 3.3. The Jacobian as an approximate tangent-space projector

The key takeaway from Eq. \eqref{eq:lik} is that the Jacobian of the Tweedie denoiser does not act in an arbitrary way: under the standard "data lie near a smooth manifold" assumption and in the small-noise regime, it behaves **approximately** like a projection onto the tangent space of the data manifold. This is exactly what we want for inverse problems: the likelihood gradient should pull us **along** the manifold rather than pushing us off it.

To make this precise, assume that clean data concentrate near a smooth $d_0$-dimensional manifold $M_0 \subset \mathbb{R}^d$. For notational simplicity, work in local coordinates around some point $x^\star \in M_0$ and decompose any nearby point as

$$
x_t = x^\star + u + n,
$$

where $u \in T_{x^\star} M_0$ is the tangent component and $n \perp T_{x^\star} M_0$ is the normal component.

In the small-noise regime with Gaussian corruption $x_t = x_0 + \sigma \varepsilon$, $\varepsilon \sim \mathcal{N}(0,I)$, a standard manifold argument (see e.g. denoising-score literature) shows that the log-density of $x_t$ can be approximated as

$$
\log p_t(x_t)
\;\approx\;
C(u) \;-\; \frac{1}{2\sigma^2}\,\|n\|^2,
$$

where $C(u)$ varies slowly along the manifold and the quadratic term penalizes distance in the normal direction. Consequently, the score is approximately

$$
\nabla_{x_t} \log p_t(x_t)
\;\approx\;
-\frac{1}{\sigma^2}\,n,
$$

i.e., it points back toward the manifold by cancelling the normal offset $n$. Plugging this into Tweedie’s formula,

$$
\hat x_0(x_t,t) = x_t + \sigma^2 \nabla_{x_t} \log p_t(x_t),
$$

we obtain, to first order,

$$
\hat x_0(x_t,t)
\;\approx\;
x_t + \sigma^2\Big(-\frac{1}{\sigma^2} n\Big)
=
x_t - n
=
x^\star + u.
$$

In other words, when the noise is small and the model is accurate, the Tweedie
denoiser $\hat x_0(x_t,t)$ behaves like the **nearest-point projection** onto the
data manifold:

$$
\hat x_0(x_t,t) \;\approx\; \Pi(x_t)
\;:=\;
\arg\min_{z \in M_0} \|z - x_t\|_2^2.
$$

For this nearest-point projection map $\Pi$, classical differential geometry tells us that its Jacobian at a point $x_t$ whose projection is $x^\star = \Pi(x_t)$ is the orthogonal projector onto the tangent space $T_{x^\star} M_0$:

$$
D\Pi(x_t)
=
P_{T_{x^\star} M_0}.
$$

Intuitively, a small perturbation $\delta x_t$ of the input decomposes into

$$
\delta x_t
=
\delta x_{\mathrm{tan}} + \delta x_{\mathrm{nor}},
\quad
\delta x_{\mathrm{tan}} \in T_{x^\star} M_0,\;
\delta x_{\mathrm{nor}} \perp T_{x^\star} M_0.
$$

Under the projection, the tangent component moves the projected point along the manifold, while the normal component is largely “forgotten”:

$$
D\Pi(x_t)\,\delta x_t
\;\approx\;
\delta x_{\mathrm{tan}}.
$$

Since Tweedie’s denoiser $\hat x_0(x_t,t)$ is, in the small-noise limit, a smooth perturbation of this nearest-point projection, its Jacobian $J_{\hat x_0}(x_t) = \partial \hat x_0 / \partial x_t$ inherits the same behavior:

$$
J_{\hat x_0}(x_t)\,\delta x_t
\;\approx\;
\delta x_{\mathrm{tan}},
$$

i.e., it **keeps** the tangent component and **suppresses** the normal component of any perturbation.

Plugging this geometric picture back into Eq. \eqref{eq:lik}, we can reinterpret the likelihood score as

$$
\nabla_{x_t}\log p(y \mid x_t)
\;\approx\;
\underbrace{J_{\hat x_0}^\top(x_t)}_{\text{tangent-space projector}}\,
\underbrace{\nabla_{x_0}\log p(y \mid x_0)\big|_{x_0=\hat x_0(x_t,t)}}_{\text{clean-space DC gradient}}.
$$

Geometrically:

1. We first compute a **clean-space data-consistency gradient**
   
   $$
   g_{x_0}
   =
   \nabla_{x_0}\log p(y \mid x_0)
   =
   \frac{1}{\sigma^2} A^\top (y - A \hat x_0),
   $$

   which tells us how to move $\hat x_0$ to better match the measurement.

2. We then apply $J_{\hat x_0}^\top(x_t)$, which approximately projects this gradient onto the tangent space $T_{\hat x_0} M_0$, removing its normal component.

The resulting likelihood score in noisy space moves $x_t$ in a direction that (i) improves data consistency because it originates from $g_{x_0}$, and (ii) remains compatible with the diffusion prior because its normal component has been suppressed. In this approximate sense, the Jacobian of Tweedie’s estimator acts as a **tangent-space projector** for likelihood guidance.











### 3.4. 2D Toy Illustration


To make the geometric effect of the Jacobian term more concrete, we visualize a toy 2D inverse problem where the clean data lie on a sine-shaped manifold $$M_0 \subset \mathbb{R}^2$$. We fix a single target point $$x_0^\star \in M_0$$ (red star) that is consistent with the measurement $$\mathbf{y}$$, and start the reverse diffusion from a noisy point far away from the manifold. At each reverse-time step we consider three different update rules, corresponding to three panels in the animation.

<figure class="figure">
  <img 
    src="{{'/assets/img/2026-04-27-diffusion-inverse-problems/posterior_guided_sampling.gif' | relative_url }}"
    alt="Posterior-guided sampling: prior vs naive guidance vs Jacobian guidance"
    style="max-width: 100%; height: auto; image-rendering: auto;">
</figure>
<div class="caption">
     Animation 2: Clean-space local-MAP as a denoise–optimize–re-noise scheme.
</div>



* **Prior-only dynamics (left).**
  In the first panel we ignore the measurement and only follow the diffusion **prior score**. Geometrically, this score always pulls the current state $$x_t$$ back towards the nearest point on the learned manifold $$M_0$$. The trajectory (purple) quickly relaxes to some high-density region of the prior, but this point has no reason to satisfy the measurement. This corresponds to unconditional sampling: we solve the generative problem, not the inverse problem.

* **Prior + naive guidance in noisy space (middle).**
  In the second panel we add a **naive likelihood guidance** term directly in the ambient space, approximating the posterior score as
  
  $$
  \nabla_{x_t} \log p(x_t \mid y)
  \;\approx\;
  s_\theta(x_t,t)
  \;+\;
  \underbrace{\nabla_{x_t} \log p(y \mid x_t)}_{\text{naive DC term}}.
  $$

  The blue arrow shows the prior score $$s*\theta(x_t,t)$$, which pulls $$x_t$$ back to the manifold, while the orange arrow is the guidance vector pointing towards the target. Because this guidance is defined in the ambient space, it typically has a large normal component and **pushes the trajectory off the manifold**. As a result, the two forces frequently oppose each other (“guidance fights prior”), leading to zig-zag trajectories that cut across low-density regions and visibly leave $$M_0$$.

* **Prior + Jacobian-guided likelihood (right).**
  In the third panel we use the Tweedie denoiser $$\hat x_0(x_t,t)$$ as a clean-space anchor, compute the **clean-space measurement gradient** $$\nabla_{x_0}\log p(y \mid x_0)$$ at $$x_0 = \hat x_0$$, and then map it back to the noisy space via the Jacobian:
  
  $$
  \nabla_{x_t}\log p(y \mid x_t)
  \;\approx\;
  J_{\hat x_0}^\top(x_t)\,
  \nabla_{x_0}\log p(y \mid x_0)\Big|_{x_0=\hat x_0(x_t,t)}.
  $$

  As argued in the previous subsection, the Jacobian $$J_{\hat x_0}(x_t)$$ behaves like a **projection onto the tangent space** $$T_{\hat x_0} M_0$$. In the animation, this means the orange guidance arrow is no longer allowed to point arbitrarily into the ambient space: it is projected to lie **along the manifold**, while the blue prior score keeps $$x_t$$ attached to $$M_0$$. The resulting posterior-guided trajectory slides along the sine curve toward the target, achieving data consistency *without* drifting away from the learned generative manifold.

This toy example illustrates the core role of the Jacobian term in posterior-guided sampling: it converts a clean-space likelihood gradient into a noisy-space correction that is automatically constrained to the tangent space of the data manifold. In practice, this allows us to approximate the posterior score as

$$
\nabla_{x_t}\log p(x_t \mid y)
\;\approx\;
s_\theta(x_t,t)
\;+\;
J_{\hat x_0}^\top(x_t)\, \nabla_{x_0}\log p(y \mid x_0),
$$

so that the prior and likelihood contribute **compatible** vector fields—one keeps us on the manifold, the other moves us along it toward solutions that both match the measurements and remain realistic under the diffusion prior.







## 4. Local-MAP optimization Paradigm


The posterior-guided sampling paradigm views inverse problems as a **sampling task**: we try to simulate a posterior diffusion process in noisy space and read off samples from $p(\mathbf{x}_0 \mid \mathbf{y})$ at $t = 0$. In this section we turn to a complementary viewpoint, exemplified by DDS <d-cite key="chung2023decomposed"></d-cite>,  DDRM <d-cite  key="kawar2022denoising"></d-cite>, DDNM <d-cite key="wang2022zero"></d-cite>, DiffusionMBIR <d-cite key="chung2023solving"></d-cite>, and more recent LMAPS methods <d-cite key="zhang2025local"></d-cite>, which instead treat inverse problems as a sequence of **local optimization problems in the clean space** $\mathbf{x}_0$.


At a high level, these methods answer a different question:

<aside class="l-body box-note" markdown="1">
Rather than asking "how should we move $$\mathbf{x}_t$$ along the posterior vector field?", we repeatedly ask : <strong>"given our current noisy state, which clean image best balances data fidelity and the diffusion prior, locally?"</strong>
</aside>



With this in mind, each reverse step then consists of three conceptually separate operations:

1. **Denoise to a clean-space anchor:** map the current noisy point $\mathbf{x}_t$ to a Tweedie denoiser $\hat{\mathbf{x}}_0(\mathbf{x}_t, t)$, which serves as a local prior "anchor" in the clean space.
   
2. **Local MAP refinement:** around this anchor, solve a **small MAP problem** in $\mathbf{x}_0$ that trades off data consistency and proximity to $\hat{\mathbf{x}}_0$.

3. **Re-noising:** re-inject the optimized solution into the diffusion trajectory by adding back the appropriate amount of noise.

Over the full reverse trajectory, this yields a **denoise–optimize–re-noise** scheme that gradually transports the initial noise towards a measurement-consistent and prior-plausible solution. As shown in Figure 2, starting from pure noise $$z = x_1$$ at $$t=1$$, the reverse process produces intermediate noisy states $$x_t, x_{t-1}, \dots$$ in the noisy space. At each step, 

- Step 1: Tweedie denoising maps $$x_t$$ to a clean-space anchor on the diffusion prior manifold $$\mathcal{M}_{\mathrm{DM}}$$; 
  
- Step 2: a local MAP subproblem in  clean data space $$(x_0)$$ refines this anchor toward the posterior manifold $$\mathcal{M}_{\mathrm{pos}} = \mathcal{M}_{\mathrm{DM}}\cap \mathcal{M}_{\mathrm{obs}}$$, balancing data consistency with the observation manifold $$\mathcal{M}_{\mathrm{obs}}$$; 
- Step 3: the optimized clean solution is re-noised to obtain the next noisy state $$x_{t-1}$$. 

Iterating these three steps transports  samples towards posterior modes that are both measurement-consistent and realistic under the diffusion prior.


{% include figure.liquid
   path="assets/img/2026-04-27-diffusion-inverse-problems/local_map.jpg"
   class="img-fluid"
   caption="Figure 2. Clean-space local-MAP as a denoise–optimize–re-noise scheme."
%}


In what follows we first derive the local MAP objective, then show how to implement it algorithmically, and finally discuss when this local scheme approximates the **global** MAP solution.



### 4.1. From global MAP to local quadratic surrogates

Recall the global MAP objective from Eq. \eqref{eq:obj}:

$$
\hat{\mathbf{x}}_{\text{MAP}}
= \arg\max_{\mathbf{x}_0}
\left[ \log p(\mathbf{y}\mid \mathbf{x}_0) + \log p(\mathbf{x}_0)\right].
\tag{4.1}\label{eq:global_map}
$$

Under the linear–Gaussian measurement model $$\mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n}$$, the negative log-likelihood is a familiar quadratic:

$$
-\log p(\mathbf{y}\mid \mathbf{x}_0)
=
\frac{1}{2\sigma^2}\|\mathbf{y}-\mathbf{A}\mathbf{x}_0\|_2^2 + \text{const}.
$$

The difficulty lies in the prior term $$-\log p(\mathbf{x}_0)$$. For a diffusion prior, we do not have direct access to this function or its gradient in $$\mathbf{x}_0$$-space; all we can evaluate is the **noisy-space score** $$\nabla_{\mathbf{x}_t}\log p_t(\mathbf{x}_t)$$ along the diffusion trajectory.

This is where the Tweedie estimator $$\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$$, introduced in the previous section, becomes useful. For a fixed time $$t$$ and noisy sample $$\mathbf{x}_t$$, we can view

$$
\hat{\mathbf{x}}_0(\mathbf{x}_t, t)
= \mathbb{E}[\mathbf{x}_0 \mid \mathbf{x}_t]
$$

as a **local summary of the prior**: it lives in a high-density neighborhood of the data manifold, and in the small-noise regime it behaves like a soft projection onto $M_0$. It is therefore natural to use $\hat{\mathbf{x}}_0$ as the center of a **local quadratic surrogate** for $-\log p(\mathbf{x}_0)$,

$$
-\log p(\mathbf{x}_0)
\;\approx\;
\frac{1}{2}\|\mathbf{x}_0 - \hat{\mathbf{x}}_0(\mathbf{x}_t,t)\|_2^2 + \text{const},
$$

which says “in a small neighborhood of $\hat{\mathbf{x}}_0$, the prior behaves as if it were a Gaussian with mean $\hat{\mathbf{x}}_0$ and unit covariance.”

Substituting this approximation into Eq. \eqref{eq:global_map} and keeping the exact data term yields the **local MAP objective** at time $t$:

$$
\mathbf{x}_0^\star(\mathbf{x}_t)
\;\approx\;
\arg\min_{\mathbf{x}_0}
\underbrace{\frac{\gamma}{2}\|\mathbf{y}-\mathbf{A}\mathbf{x}_0\|_2^2}_{\text{data consistency}}
\;+\;
\underbrace{\frac{1}{2}\|\mathbf{x}_0-\hat{\mathbf{x}}_0(\mathbf{x}_t,t)\|_2^2}_{\text{local prior / trust-region term}},
\tag{4.2}\label{eq:local_map}
$$

where $\gamma$ is a tunable parameter that absorbs $1/\sigma^2$ and possible rescalings of the prior term.

The second term plays a dual role:

- **Local prior:** it encodes the diffusion prior **locally** around $\hat{\mathbf{x}}_0$;
- **Trust region:** it keeps the optimizer from wandering too far away from the current clean-space anchor, ensuring that the Gaussian approximation of the prior remains valid.

In other words, Eq. \eqref{eq:local_map} is a **local, trust-region version** of the global MAP problem \eqref{eq:global_map}, with the full diffusion prior replaced by a quadratic surrogate centered at $\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$.


### 4.2. Denoise–optimize–re-noise scheme

Given the local objective \eqref{eq:local_map}, local-MAP-based algorithms build a reverse-time solver by repeating the following pattern at each time step $t_k$:

1. **Step 1:** Denoise to obtain a clean-space anchor.
   
   Starting from the current noisy state $\mathbf{x}_{t_k}$, we first compute the Tweedie denoiser

   $$
   \hat{\mathbf{x}}_0^{(k)}
   =
   \hat{\mathbf{x}}_0(\mathbf{x}_{t_k}, t_k)
   =
   \frac{\mathbf{x}_{t_k} + \sigma^2(t_k)\,s_\theta(\mathbf{x}_{t_k},t_k)}{\alpha(t_k)},
   $$

   which serves as a **clean-space anchor** encoding the diffusion prior at time $t_k$. This step is identical to what we used in the posterior-guided sampling paradigm: Tweedie provides a statistically optimal estimate of the underlying clean image given the noisy observation $\mathbf{x}_{t_k}$.

2. **Step 2:** Local MAP optimization in the clean space. 
   
   Given $\hat{\mathbf{x}}_0^{(k)}$, we now solve the local surrogate problem

   $$
   \mathbf{x}_0^{(k)\,\star}
   =
   \arg\min_{\mathbf{x}_0}
   \frac{\gamma}{2}\|\mathbf{y}-\mathbf{A}\mathbf{x}_0\|_2^2
   +
   \frac{1}{2}\|\mathbf{x}_0-\hat{\mathbf{x}}_0^{(k)}\|_2^2.
   \tag{4.3}\label{eq:local_map_k}
   $$

   For linear measurement operators $\mathbf{A}$, Eq. \eqref{eq:local_map_k} is a convex quadratic with a closed-form solution

   $$
   \mathbf{x}_0^{(k)\,\star}
   =
   \left(\gamma\,\mathbf{A}^\top\mathbf{A}+\mathbf{I}\right)^{-1}
   \left(\gamma\,\mathbf{A}^\top\mathbf{y} + \hat{\mathbf{x}}_0^{(k)}\right),
   $$

   but explicitly inverting the matrix is usually impractical. Instead, DDS <d-cite key="chung2023decomposed"></d-cite> and diffusion MBIR-style methods <d-cite key="chung2023solving"></d-cite> solve \eqref{eq:local_map_k} with **iterative linear solvers**, most notably **conjugate gradients (CG)**. This has two important consequences:

   - The data-consistency structure (e.g., convolution, Fourier sampling, Radon transforms) can be exploited via fast operators $\mathbf{A}$ and $\mathbf{A}^\top$, without ever forming $\mathbf{A}^\top\mathbf{A}$ explicitly.
   - The local MAP update can be run for a small, fixed number of CG iterations, trading off accuracy for speed. In practice, a handful of CG steps per diffusion time step already yields strong reconstructions.

   Conceptually, this step is where **classical inverse-problem machinery** (quadratic data terms, Krylov solvers, proximal point iterations) interfaces with the **diffusion prior**. All prior information enters through the choice of $\hat{\mathbf{x}}_0^{(k)}$; the rest of the computation is purely deterministic optimization in the clean space.


3. **Step 3:** Re-noise the optimized solution back to $t_{k-1}$.
   
   Having obtained the locally optimal clean image $\mathbf{x}_0^{(k)\,\star}$ that balances measurement fit and closeness to the anchor, the final step is to **inject it back into the diffusion trajectory** by adding noise corresponding to the next time step.

   For a DDIM- or VE-/VP-style schedule, this can be written in the generic form

   $$
   \mathbf{x}_{t_{k-1}}
   =
   \alpha(t_{k-1})\,\mathbf{x}_0^{(k)\,\star}
   +
   \sigma(t_{k-1})\,\boldsymbol{\epsilon}_{\theta}(\mathbf{x}_{t_{k}}, t_k),
   $$

   possibly with additional deterministic or stochastic corrections depending on the chosen sampler. The key point is that **the generative noise injection is now decoupled from the data-consistency optimization**:

   - The diffusion prior governs how we move between time steps via $(\alpha(t),\sigma(t))$ and the score network;
   - The measurement model governs how we update $\mathbf{x}_0$ locally through Eq. \eqref{eq:local_map_k}.




### 4.3. How local MAP approximates global MAP

So far we have only claimed that Eq. \eqref{eq:local_map} is a **local surrogate** to the global MAP objective \eqref{eq:global_map}. A natural question is: **under what conditions does following these local updates approximate the true MAP solution?**

Recent analyses under the umbrella of LMAPS <d-cite key="zhang2025local"></d-cite> provide the following picture:

1. **Small time steps & accurate Tweedie.**
   If we use a fine time discretization and the Tweedie estimator is accurate (so that $\hat{\mathbf{x}}_0(\mathbf{x}_t,t)$ stays in a tight neighborhood of the true posterior mean), then the quadratic surrogate in \eqref{eq:local_map} is a good approximation of $-\log p(\mathbf{x}_0)$ in that neighborhood.

2. **Limited local movement per step.**
   If the trust-region penalty $|\mathbf{x}_0-\hat{\mathbf{x}}_0|^2$ is sufficiently strong (or equivalently, if we do not over-iterate the local solver), then each update $\mathbf{x}_0^{(k),\star}$ remains close to $\hat{\mathbf{x}}_0^{(k)}$, and the Gaussian approximation of the prior does not break down.

3. **Composition of local steps.**
   As we move backward in time, the composition of these local MAP updates traces out a trajectory in the clean space that, in the continuous-time limit, can be shown to follow a **gradient flow of the negative log-posterior**. Intuitively, each local solve takes one small step in the direction of the global MAP solution, but expressed in the natural coordinates of the diffusion prior.

Under these assumptions, the final $\mathbf{x}_0$ returned by the denoise–optimize–re-noise scheme is a **consistent approximation of the true MAP estimator** in \eqref{eq:global_map}.


### 4.4. Toy Illustration: Local-MAP as Denoise–Optimize–Re-noise

To make the local-MAP paradigm concrete, we build a simple 2D toy example and visualize one full reverse trajectory
as an animation, see Animation 3.

We place clean data on a 1D “sine-wave” manifold $$M_0 = \{(u, \sin u): u \in \mathbb{R}\}$$ (Please note that $M_0$ refers to $M_{\text{DM}}$), which plays the role of a toy diffusion prior: clean samples concentrate near this curve. The forward operator is a 1D linear measurement $$\mathbf{A} = [\cos\phi,\; \sin\phi],$$ so the set of points consistent with a given measurement $$y$$ forms a **slanted observation line**

$$
M_{\mathrm{obs}} = \{\mathbf{x} \in \mathbb{R}^2 : \mathbf{A}\mathbf{x} = y\}.
$$

We pick a ground-truth point $$\mathbf{x}_0^\star \in M_0$$ and set $$y = \mathbf{A}\mathbf{x}_0^\star$$, so the red
star in the animation marks the intersection $$M_0 \cap M_{\mathrm{obs}}$$ — the ideal MAP solution.



The purple curve traces the noisy trajectory $$\{\mathbf{x}_t\}_{t=1\to 0}$$. We initialize it at a point far away
from $$\mathbf{x}_0^\star$$ in the upper-right corner. As the reverse process proceeds, the trajectory gradually bends towards the red star, illustrating how repeated local-MAP updates can still recover the correct solution even without solving a global optimization problem.



Each reverse-time step in the animation is decomposed into three colored moves, directly mirroring the
**denoise–optimize–re-noise** structure in Sec. 4.2: **Purple ($$x_t$$)** represents current noisy state,  **Blue ($$\hat{\mathbf{x}}_0$$)** represents Tweedie denoising anchor, **Orange ($$\mathbf{x}_0^\mathrm{local}$$)** represents local MAP refinement in clean space, and  finally, we re-noise $$\mathbf{x}_0^\mathrm{local}$$ to obtain the next noisy state **Purple ($$x_{t-1}$$)**.



<figure class="figure">
  <img 
    src="{{'/assets/img/2026-04-27-diffusion-inverse-problems/local_map_2d_slanted_obs.gif' | relative_url }}"
    alt="Posterior-guided sampling: prior vs naive guidance vs Jacobian guidance"
    style="max-width: 100%; height: auto; image-rendering: auto;">
</figure>
<div class="caption">
     Animation 3: Clean-space local-MAP as a denoise–optimize–re-noise scheme.
</div>


To complement the static 2D toy example, we provide an interactive 3D visualization of the **clean-space Local-MAP paradigm** (see the following Interactive Figure). The blue bowl-shaped surface represents the diffusion prior manifold $$M_{\mathrm{DM}}$$, modeled here as an anisotropic Gaussian energy landscape. The light orange plane is the observation manifold $$M_{\mathrm{obs}}$$, corresponding to a one-dimensional linear measurement operator $$y = A x$$. Their intersection encodes the ideal posterior manifold where both the generative prior and the data-consistency constraint are satisfied.

The black curve shows a full **reverse-diffusion trajectory** ({x_t}_{t=T}^{0}) in the clean space. You can type an integer (t) into the input box above the figure to inspect a single reverse step. For each selected time step (t), the visualization highlights the three sub-steps of the Local-MAP update:

* A **green segment** connects the current noisy point $$x_t$$ to its Tweedie-denoised anchor $$\hat{x}_0^{t}$$.

* A **red segment** connects $$\hat{x}_0^{t}$$ to the Local-MAP solution $$\hat{x}_0^{t,*}$$.

* A **blue segment** connects $$\hat{x}_0^{t,*}$$ to the next noisy iterate $$x_{t-1}$$.
  
By rotating the 3D view and sweeping $$t$$ from large values down to small ones, one can see how Local-MAP repeatedly nudges the reverse trajectory toward the intersection of $$\mathcal{M}_{\mathrm{DM}}$$ and $$\mathcal{M}_{\mathrm{obs}}$$ while staying faithful to the diffusion prior.




<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-diffusion-inverse-problems/local_map_interactive.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="600px"
    width="100%"
  ></iframe>
</div>
<div class="caption">
Interactive Figure: Local-MAP as Denoise–Optimize–Re-noise. Drag the slider to sweep the time index $t$ and inspect how the three sub-steps evolve across the reverse process.
</div>

## 5. Conclusion

In this post, we examined diffusion priors for inverse problems through two complementary paradigms: **Posterior-Guided Sampling** and **Local-MAP Optimization**. While the landscape of diffusion inverse solvers is vast <d-dict key="daras2024survey"></d-dict> — including methods based on Variational Inference (VI) <d-cite key="mardani2023variational,alkan2023variational"></d-cite>, latent space optimization <d-cite key="daras2022score,wang2024dmplug"></d-cite>, and asymptotically exact MCMC/SMC approaches <d-cite key="wu2024principled,dou2024diffusion"></d-cite> – but we focused on these two because they have become the most practical and widely used ways to turn a powerful unconditional diffusion model into a plug-and-play inverse solver.

At a high level, both paradigms realize the same philosophy:

<aside class="l-body box-note" markdown="1">
<strong>Decouple the generative prior from the measurement physics.</strong>
</aside>

Once a diffusion prior has been trained (once and for all) on natural signals, we can pair it with an arbitrary
forward operator $$\mathbf{A}$$ and noise model to solve tasks as diverse as inpainting, deblurring, super-resolution,
and CT reconstruction – without retraining the generative model. What changes from task to task is not the
diffusion backbone, but **how we combine a prior operator with a data-consistency operator**.

### 5.1. Posterior-guided sampling: gradient-based flows in noisy space

The posterior-guided paradigm keeps the reverse diffusion entirely in the noisy space $$\mathbf{x}_t$$. By replacing
the prior score field with the posterior score (Eq. \eqref{eq:posterior_score}), the reverse PF-ODE becomes a
gradient-like flow along $$\nabla_{\mathbf{x}_t}\log p(\mathbf{x}_t\mid \mathbf{y})$$. The posterior score can be decomposed into two parts, namely the prior score and the likelihood score, this makes posterior-guided sampling look like an **operator splitting** scheme in noisy space: at each step we follow the diffusion prior flow and then apply a data-consistency correction that has been carefully projected onto the manifold’s tangent space.

From this perspective, methods such as DPS/DMPS and MCG can be viewed as different instantiations of the same
template: they are **gradient-based posterior flows** that operate directly on $$\mathbf{x}_t$$.

- **Strengths.** Conceptually close to Bayesian posterior sampling; naturally produces **multiple diverse
  reconstructions** and can, in principle, approximate the full posterior $$p(\mathbf{x}_0\mid \mathbf{y})$$.
  Works well with non-linear or differentiable black-box forward models, as long as we can backpropagate to
  obtain a likelihood gradient.

- **Limitations.** Accurate likelihood-score approximations typically require **Jacobian–vector products (JVP) through
  the Tweedie denoiser**, on top of standard forward passes. This increases memory and compute, especially at
  high resolution. The dynamics can also be sensitive to guidance strength and discretization; if the likelihood
  term is not geometrically aligned (e.g. without the Jacobian projection), it may fight against the prior
  instead of flowing along the data manifold.

### 5.2. Local-MAP optimization: denoise–optimize–re-noise in clean space

The local-MAP paradigm instead views inverse problems as a sequence of **small optimization problems in the
clean space** $$\mathbf{x}_0$$, this yields a clean-space **denoise–optimize–re-noise** operator splitting. The prior
trajectory (Steps 1 and 3) depends only on the unconditional diffusion model, while the inner optimization
(Step 2) depends only on the measurement model and our choice of local objective and solver.

Under this view, seemingly different methods – including DDS, DDRM, DiffusionMBIR, and various PnP/RED-style
algorithms – can be unified as instances of the same **local-MAP optimization template** with different choices
of surrogate, solver, and re-noising schedule.

- **Strengths.** Recasts each reverse step as a **standard optimization problem in $$\mathbf{x}_0$$**, allowing us
  to leverage decades of work on quadratic solvers, Krylov methods, and proximal algorithms. Avoids noisy-space
  JVPs entirely: all gradients live in clean space or through $$\mathbf{A}$$. Naturally targets **MAP-like
  reconstructions**, which is often exactly what practitioners want in imaging applications.

- **Limitations.** The method is only **locally** MAP: it relies on Tweedie anchors and quadratic surrogates
  remaining faithful to the true posterior landscape in a neighborhood. Solving a local subproblem at every
  step (e.g. via CG) can be costly for large-scale 3D problems or complex $$\mathbf{A}$$. By construction it is
  mode-seeking, and does not explore the full posterior without additional machinery.

### 5.3. Decoupling as the main design principle

Despite these differences, both paradigms share the same structural core:

<aside class="l-body box-note" markdown="1">
<strong>A diffusion prior operator that proposes plausible samples, and a data-consistency operator that enforces the physics.</strong>
</aside>



Posterior-guided methods implement this decoupling directly in noisy space via a posterior score field, while
local-MAP methods implement it in clean space via Tweedie anchors and trust-region MAP updates. This decoupling
is what makes diffusion priors so attractive for real-world inverse problems: once the prior is trained, we can
adapt to new measurement operators and noise models largely by changing **how** we plug in the data-consistency
operator, rather than retraining the model.

### 5.4. Limitations and Outlook

Although both paradigms are elegant from theoretically and practically perspective, they rely on approximations (Tweedie denoising, score-to-likelihood projections, local quadratic surrogates). When these approximations break (strongly ill-posed settings, heavy nonlinearity, severe out-of-distribution measurements), performance can degrade in ways that are hard to diagnose. In addition, the computation cost is also a problem that cannot be ignored, posterior-guided methods can be dominated by JVP/gradient cost; local-MAP can be dominated by per-step inner solves—either way, iterative sampling is often the bottleneck.

Looking forward, several frontiers remain active in this area:


- **Hybrid schemes.** Combine global exploration (posterior-guided at high noise) with sharp refinement (local-MAP near $$t\!\to\!0$$).

- **Beyond linear–Gaussian models.** Most current theory assumes $$ \mathbf{y} = \mathbf{A}\mathbf{x}_0 + \mathbf{n} $$ with Gaussian noise. Extending these paradigms to non-linear, non-Gaussian, or partially unknown forward models
  is still challenging.  

- **Acceleration and distillation.** Iterative sampling remains expensive. Distillation, consistency models, and
  related techniques offer promising ways to compress these multi-step procedures into a few learned updates
  while preserving their geometric structure.  

Our hope is that the geometric and probabilistic lens developed in this blog – posterior-guided flows in noisy
space, and local-MAP optimization in clean space – can serve as a compact mental model for navigating the rapidly
growing literature on diffusion-based inverse problems, and as a design toolkit for building the next generation
of robust and reusable inverse solvers.
