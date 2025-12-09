---
layout: distill
title: An Impossibility Trilemma for Data-Free Sampler Evaluation
description: Neural samplers aim to learn to sample a target unnormalized energy potential. Sampler quality can be evaluated in a data-free manner, using only the model and the target potential, or in a data-driven manner, with additional data about the target distribution such as known modes, summary statistics, and reference MCMC samples. While data-driven eval is valuable, data-free eval has compelling conceptual advantages, raising the question of how well data-free eval could work. Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among i) mode-covering metric, ii) stable with finite variance, iii) transitivity guarantee (if A>B and B>C, then A>C). We situate the implications of this trilemma in the broader conceptual landscape of data-driven and data-free sampler evaluation.
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
  - name: Introduction
  - name: Informal overview of trilemma
  - name: Trilemma
    subsections:
    - name: Characterizing mode-covering with sensitivity analysis
    - name: Proof of dilemma for single-model data-free eval
    - name: Pairwise comparators
  - name: Stein discrepancy
  - name: Citations
  - name: Footnotes
---

<!-- # Trilemma for data-free sampler evaluation -->

<!-- ## Abstract
Neural samplers like diffusion samplers aim to learn to sample a target unnormalized energy potential, which is known but considered challenging to sample from. Evaluation of learned sampler quality is a critical aspect for research progress, with mode-covering metrics of particular importance. Two general approaches are common: i) data-free, using only model samples and the target potential, and ii) data-driven, which use additional data about the target distribution such as known modes, experimental observables, summary statistics, and/or reference MCMC samples as a gold standard. While data-driven eval is valuable, data-free eval has compelling conceptual advantages, raising the question of how well data-free eval could work. Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among i) mode-covering metric, ii) stable with finite variance, iii) no dominance cycles. We situate the implications of this trilemma in the broader conceptual landscape of data-driven and data-free sampler evaluation.
--- -->

## Introduction

Sampling from unnormalized energy potentials is a key problem in probabilistic inference, statistical physics, and molecular dynamics. In recent years, deep generative modeling approaches for sampling has been investigated in diffusion samplers, adjoint Schrodinger bridge sampler [cite], as well as generative flow networks and normalizing flows. 
These models aim to learn a generative model distribution $q_\theta(x)$ that approximates a known unnormalized density. For instance, in molecular dynamics, the energy potential $U(x)$ of molecular conformations is known analytically in closed-form, and we aim to sample the Boltzmann distribution $p(x) \propto \exp(-U(x))$. Mode discovery, possibly by generalizing from known modes to efficiently discover new modes, is a critical aspect to evaluate in neural samplers, because traditional MCMC methods can mix poorly for rugged, high dimensional densities.

There are two main approaches for evaluating how well a sampler matches the target distribution: data-free, and data-driven. In data-free sampler eval, we only have the target unnormalized density, and consider metrics like KL divergence, kernelized maximum mean discrepancy, and Stein discrepancy. 

In data-driven sampler eval, we have access to additional data about $p$ beyond its unnormalized density. For example, synthetic eval settings can be constructed with a known number of modes and locations, which can be used to count how many modes neural samplers recover. In molecular dynamics, molecules like alanine dipeptide (~20 atoms) and chignolin (~200 atoms) are deeply understood with known modes. For more complex molecules, experimental observables such as X can be used for evaluation [cite bioemu], but this can conflate sampler evaluation with misalignments between the target potential's model of reality and actual reality. For example, we might sample the target potential perfectly, but still fit observables poorly, because the target potential is imperfect with respect to reality. In other cases, for instance molecules that are less well understood, samplers may be evaluated using reference MCMC samples used as a "gold standard". However, it can be unclear how accurate these reference MCMC samples are, especially for target densities that are highly challenging to sample. In the most challenging sampling problems with no prior knowledge, data-driven eval faces a "catch-22" situation where we need trustworthy samples to evaluate whether our samples are trustworthy, making it difficult to self-bootstrap off the ground. 

While data-driven eval is valuable, the upsides of data-free sampler eval are appealing. In an ideal world where data-free sampler eval worked perfectly, research on neural samplers could be performed in gym-like environments with little overhead for supporting a huge diversity of target potentials, like with virtual environments or video games in reinforcement learning research.
These considerations motivate asking how well data-free sampler eval could work.

Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among: i) mode-covering metric, ii) stable with finite variance, iii) allows sampler ranking without cyclic dominance (disallows A>B>C>A).

## Informal overview of trilemma

Data-free sampler evaluation is challenging because we have samples only from the model, and not from the target distribution (otherwise the sampling problem is solved). With model samples and likelihoods, we can stably estimate the reverse KL $$ \mathbb{E}_{q}[\log(q/p)] $$, but this is mode-seeking -- the reverse KL strongly rewards $q$ matching $$ p $$ among model samples, and does not strongly penalize missing modes in $$ p $$. If we hill-climb the reverse KL as an evaluation metric, we would generally reward samplers that fit a subset of modes very well, even if they are missing other important target modes, over samplers that discovered more target modes. This means the reverse KL is not a very useful sampler evaluation metric because it ignores the problem of mode discovery.

The forward KL $$ \mathbb{E}_{p}[\log(p/q)] $$ is mode-covering: it strongly rewards the model for covering modes of $$ p $$, making it ideal for evaluating mode discovery. Unfortunately, it is unstable to estimate. With access only to model samples, we require importance reweighting to estimate it as $$ \mathbb{E}_{q}[(p/q) \log(p/q)] $$. This theoretically does not have bounded, finite variance, and in practice is prohibitively high variance in high-dimensional settings of interest to be a useful evaluation metric.

{% include figure.liquid path="assets/img/2026-04-27-sampler-eval-trilemma/rkl-fkl.png" class="img-fluid" %}
<div class="caption">
    Reverse KL is mode-seeking, while forward KL is mode-covering. Axes depict mean and std parameters for a Gaussian. Values plot discrepancy to a two-Gaussian mixture with mean, std depicted with the red x's.
</div>

The contrast between the reverse KL and forward KL introduces the tension between items i) mode-covering metric, and ii) stable with finite variance, in our trilemma. By thinking beyond $f$-divergences to pairwise comparators, such as

$$
\int m(x) \sigma\left( \log \frac{ p(x)}{ m(x)} \right) \log \frac{p(x)}{ q(x)} ~dx, \quad m(x) = 0.5 q_1(x) + 0.5q_2(x)
$$

we can achieve a metric stably estimated with only model samples, that is also mode-covering among the modes within the mixture $m$. This pairwise comparator could thus score if a sampler $q_1$ is more mode-covering than $q_2$ head-to-head, while ignoring target modes that are unseen by both samplers. Unfortunately, such pairwise comparators introduce the third element of the trilemma: they may not be proper for $p$, and/or could introduce dominance cycles, and/or do not have pool independence.

## Trilemma

In this section, we provide a more precise characterization of the trilemma. First, we set up a definition of a "mode-covering" metric via sensitivity analysis. We then offer a short proof of the aforementioned dilemma between mode-covering and importance weights for single-model evaluation metrics. We then consider pairwise model comparison evaluation metrics, which can achieve both mode-covering and stability, but at the cost of the third item of the trilemma.

### Characterizing mode-covering with sensitivity analysis

What does it mean for a metric to be mode-covering or mode-seeking? One natural approach is to consider how much the metric changes when the model likelihood shrinks to zero at a target mode. A small change shows the metric does not penalize a model for dropping modes -- less mode-covering -- while a large change represents more mode-covering behavior.

To quantify how much a metric changes when the model likelihood shrinks to zero at a target mode, we can take its derivative with respect to $q(x)$, and study its form as $q(x) \to 0$ at some $x$ that is a target mode (i.e., with $p(x) > 0$).

For the Forward KL, at a given $x$:

$$
-\frac{\partial p(x) \log\frac{p(x)}{q(x)} }{\partial q(x)} = \frac{p(x)}{q(x)} = O\left( \frac{p}{q} \right)
$$

For the reverse KL, at a given $x$:

$$
\begin{align}
-\frac{\partial q(x) \log\frac{q(x)}{p(x)} }{\partial q(x)} &= 
-\frac{\partial}{\partial q(x)} q(x) \log q(x) + \frac{\partial}{\partial q(x)} q(x) \log p(x) \\
&= -\left( 1 \log q(x) + q(x) \frac{1}{q(x)} \right) + \log p(x) \\
&= \log \frac{p(x)}{q(x)} + 1 \\
&= O \left(\log \frac{p}{q} \right)
\end{align}
$$

We can see that the reverse KL is exponentially less sensitive to $q \to 0$ than the forward KL. This provides a more quantitative way to characterize that the forward KL is more mode-seeking. For this work, we will operate with this definition:

**Definition**: A metric is mode-covering if its partial derivative with respect to $q(x)$ is $O(p/q)$.

As an aside, we can apply the same analysis to Stein discrepancy, and find that the partial derivative is $O(\|\nabla_x \log p(x) \|)$, which is constant in terms of $q$. Thus, the Stein discrepancy is even less mode-seeking than the reverse KL, which we can visualize experimentally.

{% include figure.liquid path="assets/img/2026-04-27-sampler-eval-trilemma/stein_discrepancy_experiment.png" class="img-fluid" %}
<div class="caption">
    Stein discrepancy vs. reverse and forward KL. Axes depict mean and std parameters for a Gaussian. Values plot discrepancy to a two-Gaussian mixture with mean, std depicted with the red x's.
</div>

### Proof of dilemma for single-model data-free eval

First, let's focus on evaluation metrics that consider one model at a time. These can be written $\mathcal{D}(p, q)$, in contrast to pairwise comparators $\mathcal{D}(p, q_1, q_2)$ which we will consider later.

**Lemma**: Let the evaluation metric be a function $\mathcal{D}(p, q)$ defined as an integral over the domain, for any inte:

$$
\mathcal{D}(p, q) = \int \phi(x, p(x), q(x)) ~dx
$$

$\mathcal{D}(p, q)$ cannot satisfy both properties:

1. Mode-covering
2. No importance sampling, when estimating the metric as an expectation under $q$, meaning no importance weight factors like $p(x)/q(x)$.

**Proof**. The proof follows by understanding that the $O(p/q)$ term arises if and only if the integrand has a leading term $p \log (q)$, because:

$$
\frac{\partial}{\partial q} p\log(q) = p \frac{1}{q} = O(p/q)
$$

Importantly, the term must be $p \log q$. For example, $q p \log q$ does not work, because the $q$ cancels the desired $1/q$ term by the product rule:

$$
\frac{\partial}{\partial q} q p \log(q) = q\frac{1}{q}p + p \log(1/q) = O(p\log(1/q))
$$

When the integrand has a leading term proportional to $p \log q$, if we wish to estimate the integral as an expectation under $q$, we must incur importance weights $\frac{p}{q}$. This shows the two conditions are incompatible.

#### Importance weights incur unbounded variance

Our estimator is an expectation under model samples. Its stability depends on its variance, which is governed by the second moment $\mathbb{E}_{x \sim q}[(...)^2]$. If the estimator contains importance weights, its variance depends on:

$$
\mathbb{E}_{x \sim q} \left[ \left( \frac{p}{q} \right)^2 \right] = \int \frac{p^2}{q} ~dx
$$

This integral diverges when $p$ has heavier tails than $q$, i.e., when there are missing modes, so the importance weights have unbounded variance.

### Pairwise comparators

We saw that achieving both stability and mode-covering is impossible for evaluation metrics that compare one model to the target. However, to drive progress in machine learning research, we don't need a "global" measure of sampler quality; it can suffice instead to just have a relative, or local measure, of whether one sampler is better than the other. In this section, we'll show that pairwise comparators can achieve both stability and mode-covering, but at the cost of other desirable attributes.

A key challenge with sampler evalution is that with only model samples, it's challenging to know what target modes we've missed. However, when we compare two samplers, it becomes easy to tell if one sampler missed modes that the other sampler found, by comparing samplers with the mixture distribution.

$$
m(x) = 0.5 q_1(x) + 0.5q_2(x)
$$

For stability, let's focus on the set of points with non-vanishing probability under the mixture:

$$
\Omega_\epsilon = \{ x : q_1(x) > \epsilon \text{~~or~~} q_2(x) > \epsilon \}
$$

Consider this pairwise comparator $\mathcal{D}(p, q_1, q_2)$ that compares two models $q_1$ and $q_2$. 

$$
\int_{ \Omega_\epsilon } p(x) \log \frac{q_1(x)}{ q_2(x)} ~dx
$$

which can be estimated from samples $x_i \sim m(x)$ as:

$$
\sum_{i=1}^N \left[ \mathbb{1} \left( m(x_i) > \epsilon \right) \frac{p(x_i)}{m(x_i)} \log \frac{q_1(x_i)}{q_2(x_i)} \right]
$$

On the shared support $\Omega_\epsilon$, this estimates the true difference in the forward KL. It is thus proper for $p$ within $\Omega_\epsilon$, meaning that $p$ is an optima of our pairwise comparator; no model can score better than $p$ if it is different than $p$ in $\Omega_\epsilon$. Because it estimates the forward KL difference, it is mode-covering in the mixture support, as the sensitivity of the integrand to a mode of $p$ found by $q_1$ is:

$$
-\frac{\partial p(x) \log(q_1(x) / q_2(x)) }{\partial q_2(x)} = O\left( \frac{p}{q_2} \right)
$$

Note that this pairwise comparator is blind to modes missed by both models, though this is a reasonable property in an evaluation metric. Further, by limiting importance weights to $\Omega_\epsilon$ such that $m(x_i) > \epsilon/2$, the importance weights are capped and do not explode to infinity. The estimator thus has bounded variance, improving the stability of this metric.

Sounds great, right? This pairwise comparator is both stable and mode-covering. What is the cost of this? Our pairwise comparator is no longer decomposable into the difference of single model evaluation metrics: $\mathcal{D}(p, q_1, q_2) \neq h(p, q_1) - h(p, q_2)$. In this situation, we lose a guarantee on transitivity. There can exist sets of samplers where $A>B>C>A$, forming a dominance cycle. This happens because the "evaluation set" $\Omega_\epsilon$ is dynamic and depends on the comparison participants.

**Lemma**: There exists a set of samplers that form a dominance cycle when scored by this pairwise comparator.

**Proof**. We provide a proof by construction. 


## Limitations

## Discussion

