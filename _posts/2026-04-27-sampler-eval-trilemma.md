---
layout: distill
title: Trilemma for data-free sampler evaluation
description: Neural samplers like diffusion samplers aim to learn to sample a target unnormalized energy potential, which is known but considered challenging to sample from. Here, we prove an impossible trinity (trilemma) for data-free sampler evaluation; we can only have two among i) mode-covering metric, ii) stable with finite variance, iii) no dominance cycles. We situate the implications of this trilemma in the broader conceptual landscape of data-driven and data-free sampler evaluation.
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
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-sampler-eval-trilemma.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Informal overview of trilemma
  - name: Stein discrepancy
    subsections:
      - name: Mode-coverage pressure analysis via gradient norm
      - name: Experiment
  - name: Citations
  - name: Footnotes
#   - name: Code Blocks
#   - name: Diagrams
#   - name: Tweets
#   - name: Layouts

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

# Trilemma for data-free sampler evaluation

<!-- ## Abstract
Neural samplers like diffusion samplers aim to learn to sample a target unnormalized energy potential, which is known but considered challenging to sample from. Evaluation of learned sampler quality is a critical aspect for research progress, with mode-covering metrics of particular importance. Two general approaches are common: i) data-free, using only model samples and the target potential, and ii) data-driven, which can use experimental observables, summary statistics, and/or reference MCMC samples used as a gold standard. Data-driven evaluation is common, yet can suffer from misalignment with the target potential, trust issues in reference MCMC samples, and a catch-22 issue when evaluating on new, hard-to-sample densities lacking reference samples. This raises the question how well data-free evaluation could work. Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among: i) mode-covering metric, ii) stable with finite variance, iii) allows sampler ranking without cyclic dominance (disallows A>B>C>A). We situate the implications of this trilemma in the broader conceptual landscape of data-driven and data-free sampler evaluation. -->

---

Sampling from unnormalized energy potentials is a key problem in probabilistic inference, statistical physics, and molecular dynamics. In recent years, deep generative modeling approaches for sampling has been investigated in diffusion samplers, adjoint Schrodinger bridge sampler [cite], as well as generative flow networks and normalizing flows. 
These models aim to learn a generative model distribution $q_\theta(x)$ that approximates a known unnormalized density. For instance, in molecular dynamics, the energy potential $U(x)$ of molecular conformations is known analytically in closed-form, and we aim to sample the Boltzmann distribution $p^*(x) \propto \exp(-U(x))$. 

Markov-Chain Monte Carlo (MCMC) is a traditional approach for sampling with asymptotic guarantees, but in finite time, MCMC can mix poorly for rugged, high dimensional densities, yielding samples that are far from the target distribution. In particular, MCMC can fail to discover modes of the distribution. Mode discovery, possibly by generalizing from known modes to efficiently discover new modes, is thus a critical aspect to evaluate in neural samplers. 

Evaluations are fundamental to research progress. In the sampling problem, we evaluate how consistent a model is to the target distribution $p^*$, using model samples $x \sim q_\theta$) and/or model likelihoods $q_\theta(x)$.

There are two general approaches for sampler evaluation: i) data-free, where we only have the target unnormalized density. KL divergence or Stein discrepancy are examples of data-free sampler evaluation metrics. ii) Data-driven, where we have access to additional data about $p^*$ beyond its unnormalized density. For example, synthetic eval settings can be constructed with a known number of modes and locations, which can be used to count how many modes neural samplers recover. In molecular dynamics, molecules like alanine dipeptide (~20 atoms) and chignolin (~200 atoms) are deeply understood with known modes. For more complex molecules, experimental observables such as X can be used for evaluation [cite bioemu], but this can conflate sampler evaluation with misalignments between the target potential's model of reality and actual reality. For example, we might sample the target potential perfectly, but still fit observables poorly, because the target potential is imperfect with respect to reality. In other cases, for instance molecules that are less well understood, samplers may be evaluated using reference MCMC samples used as a "gold standard". However, it can be unclear how accurate these reference MCMC samples are, especially for target densities that are highly challenging to sample. In the most challenging sampling problems with no prior knowledge, data-driven eval faces a "catch-22" situation where we need trustworthy samples to evaluate whether our samples are trustworthy, making it impossible to self-bootstrap off the ground. These considerations motivate asking how well data-free sampler eval could work.

Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among: i) mode-covering metric, ii) stable with finite variance, iii) allows sampler ranking without cyclic dominance (disallows A>B>C>A).

Data-free sampler evaluation is challenging because we have samples only from the model, and not from the target distribution (otherwise the sampling problem is solved). With model samples and likelihoods, we can stably estimate the reverse KL $\mathbb{E}_{q}[\log(q/p)]$, but this is *mode-seeking* -- the reverse KL strongly rewards $q$ matching $p$ among model samples, and does not strongly penalize missing modes in $p$. This means the reverse KL is not a very useful sampler evaluation metric because it ignores the problem of mode discovery.

The forward KL $\mathbb{E}_{p}[\log(p/q)]$ is *mode-covering*: it strongly rewards the model for covering modes of $p$, making it ideal for evaluating mode discovery. Unfortunately, it is unstable to estimate. With access only to model samples, we require importance reweighting to estimate it as $\mathbb{E}_{q}[(p/q) \log(p/q)]$. This theoretically does not have bounded, finite variance, and in practice is prohibitively high variance in high-dimensional settings of interest to be a useful evaluation metric.

The contrast between the reverse KL and forward KL introduce the tension between items i) mode-covering metric, and ii) stable with finite variance, in our trilemma. By thinking beyond $f$-divergences to pairwise comparators, such as

$$
\int m(x) \sigma\left( \log \frac{ p(x)}{ m(x)} \right) \log \frac{p(x)}{ q(x)} ~dx, \quad m(x) = 0.5 q_1(x) + 0.5q_2(x)
$$

we can achieve a metric stably estimated with only model samples, that is also mode-covering among the modes within the mixture $m$. This pairwise comparator could thus score if a sampler $q_1$ is more mode-covering than $q_2$ head-to-head, while ignoring target modes that are unseen by both samplers. Unfortunately, such pairwise comparators introduce the third element of the trilemma: they may not be proper for $p$, and/or could introduce dominance cycles, and/or do not have pool independence.

---

# Informal overview of trilemma

<!-- TODO - add definition of O(p/q) mode covering pressure -->

<!-- TODO - add class of functions considered -->

Unfortunately, satisfying the following two properties is provably impossible:

1. No importance sampling, when estimating the metric as an expectation under $q$, meaning no importance weight factors like $p(x)/q(x)$.
2. $O(p/q)$ mode-covering pressure, like forward KL

Proof:

The proof follows by understanding where the $O(p/q)$ mode-covering pressure comes from mathematically in the forward KL definition. In short, it comes from the term $p \log (q)$ in the integrand, because:

$$
\frac{\partial}{\partial q} p\log(q) = p \frac{1}{q} = O(p/q)
$$

Thus, it is necessary to have $p \log (q)$ in the integrand to achieve $O(p/q)$ mode-covering pressure.

In contrast, consider what happens when we estimate a metric as an expectation under $q$: the integrand is multiplied by a factor $q(x)$. If we have a term $q \log(q)$ in the integrand, its derivative by the product rule becomes:

$$
\frac{\partial}{\partial q} q \log(q) = q\frac{1}{q} + 1 \log(1/q) = O(\log(1/q)
$$

Having an expectation w.r.t. $q$ significantly dampens the mode-covering pressure by the product rule. We can see that the desired $1/q$ term, which is the gradient of $\log(q)$ is canceled by the front term $q$. The only way to keep the forward KL-like pressure is to importance weight by $p/q$.

This completes the proof.

# Stein discrepancy

Previously, we pinned down a definition for mode-covering vs. mode-seeking behavior for distributional divergences. When fitting $q$ to a target $p$, we studied the derivative norm of the divergence wrt $q(x)$ where $p(x) > 0$ and $q(x) \to 0$. 

> *Mode-covering pressure* means that when $p(x) > 0$ and $q(x) \to 0$, the gradient of the divergence w.r.t. $q(x)$ diverges strongly (at least $O(p/q)$), pushing $q$ to cover that region.

The forward KL has $O(p/q)$ mode-covering pressure, which is exponentially stronger than the reverse KL with $O(\log(p/q))$ pressure.

Next, we'll apply our gradient analysis to the Stein discrepancy. We will show that the Stein discrepancy has $O(1)$ pressure (wrt $q$), so mathematically it is even less mode-covering than the reverse KL! Experimentally, we show that Stein discrepancy has a similar loss landscape as reverse KL, and not forward KL.

The Stein discrepancy is a measure of distributional similarity which can be used as a sampler evaluation metric, as it does not require iid samples from the target $p$, or its normalized density. Instead, we can compute it using:

- $\nabla_x \log p(x)$, the data score
- iid samples from $q$

The core object in the Stein discrepancy is the Stein operator, which acts on a vector test function $g(x)$. Using the data score, the Stein operator is:

$$
\mathcal{T}_p g(x) = \nabla_x \log p(x) \cdot g(x) + \text{div}~g(x)
$$

This is constructed to satisfy the key property that:

$$
\mathbb{E}_{p}[\mathcal{T}_p g(x)] = 0
$$

for all test functions $g$.

The Stein discrepancy is then defined as:

$$
S(q||p) \triangleq \sup_{g \in G} | \mathbb{E}_{x \sim q} [\mathcal{T}_p g(x)] |
$$

where $\mathcal{G}$ must be constrained to keep the supremum finite. One common choice is $\mathcal{G}$ as the unit ball in a reproducing kernel Hilbert space, which corresponds to kernel Stein discrepancy. Another choice is neural Stein discrepancy, where $\mathcal{G}$ is chosen to be the set of functions learnable by a (regularized) neural network.

## Mode-coverage pressure analysis via gradient norm

Let's denote $g_q^*$ as the $g$ that achieves the supremum. Then,

$$
S(q||p) = \mathbb{E}_{q} [ \nabla_x \log p(x) \cdot g^*_q(x) + \text{div}~g^*_q(x) ]
$$

Differentiating wrt q, the multiplicative factor of $q$ from the expectation disappears, and we keep the integrand:

$$
\frac{\partial}{\partial q(x)} S(q||p) = \nabla_x \log p(x) \cdot g^*_q(x) + \text{div}~g^*_q(x)
$$

*Note: This is justified by Danskin's theorem / the envelope theorem, which says that for some $f(x) = \max_{z} \phi(x,z)$, if the maximizer $z^*$ is unique, then $\frac{\partial}{\partial x} f(x) = \frac{\partial}{\partial x} \phi(x, z^*)$.*

Suppose $\mathcal{G}$ imposes bounds $\|g\|_{\infty} \leq B$ and $\| \nabla g \|_{\infty} \leq D$, which is a loose assumption satisfied by kernel and neural Stein discrepancies. Then, for any $g$ and all $x$,

$$
\begin{align}
| \nabla_x \log p(x) \cdot g(x) + \text{div}~g(x) | &\leq \| \nabla \cdot g(x) \| + \| g(x) \| \| \nabla_x \log p(x) \| \\
&\leq dD + B \| \nabla_x \log p(x) \|,
\end{align}
$$

where $d$ is the dimension, and we use the triangle inequality and $| \text{div}~ g| \leq d \|\nabla g\|_{\infty}$.

In particular,

$$
\frac{\partial}{\partial q(x)} S(q||p)
\leq dD + B \| \nabla_x \log p(x) \|.
$$

This is independent of $q(x)$. The pressure term is $O(\|\nabla_x \log p(x) \|)$.

Thus, the supremum Stein discrepancy, is not mode-covering: it exerts bounded, location-dependent pressure that does not increase when $q(x) \to 0$ in regions where $p(x) > 0$.

Interestingly, the Stein discrepancy at $O(1)$ pressure (wrt q) is even less mode-covering than the reverse KL at $O(\log(p/q))$ pressure!

## Experiment

We repeat the common experiment, where our target $p$ is a mixture of two Gaussians with means -4, +4, and std 1. We consider a model distribution $q$ that is a single Gaussian with free parameters mean and std, and plot the divergence/discrepancy between $p, q$ as the mean and std of $q$ vary.

We compute the reverse and forward KL for comparison, and computed Stein discrepancy using kernel Stein discrepancy with an IMQ kernel with $\beta=0.5$, $c=1$, which are common hyperparameter choices.

![divergence_landscapes_with_stein-min](img/2026-04-27-sampler-eval-trilemma/divergence_landscapes_with_stein-min.png)


Yellow indicates lower (better) divergence values. The red X's denote the mean and std of the two Gaussians in the target mixture, while the red + denotes the true mean and std of the target mixture, when fitting a single Gaussian to it. 

The top row depicts the reverse and forward KL, and the bottom row depicts the Stein discrepancy and log Stein discrepancy.
Visually, the Stein discrepancy's landscape is similar to the reverse KL: The brightest yellow is at the mode-seeking solutions; the mode-covering solution (red +) is less bright yellow. This relation is more clearly seen on the log scale plot.