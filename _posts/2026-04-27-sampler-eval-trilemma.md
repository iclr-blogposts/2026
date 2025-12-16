---
layout: distill
title: An Impossibility Trilemma for Data-Free Sampler Evaluation
description: Neural samplers aim to learn to sample a target unnormalized energy potential. Sampler quality can be evaluated in a data-free manner, using only the model and the target potential, or in a data-driven manner, with additional data about the target distribution such as known modes, summary statistics, and reference MCMC samples. While data-driven eval is valuable, data-free eval has compelling conceptual advantages, raising the question of how well data-free eval could work. Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among i) mode-covering metric, ii) stable with finite variance, iii) universal ranking (dominance transitivity guarantee / model score does not depend on other models). This note surveys underexplored design space of data-free sampler eval metrics, and asks the community which eval properties we are willing to sacrifice in the face of the impossibility of satisfying all of them.
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
    - name: Loss of transitivity
    - name: General properties of pairwise comparators
  - name: Limitations
  - name: Discussion
  - name: Code
---

<!-- # Trilemma for data-free sampler evaluation -->

<!-- ## Abstract
Neural samplers like diffusion samplers aim to learn to sample a target unnormalized energy potential, which is known but considered challenging to sample from. Evaluation of learned sampler quality is a critical aspect for research progress, with mode-covering metrics of particular importance. Two general approaches are common: i) data-free, using only model samples and the target potential, and ii) data-driven, which use additional data about the target distribution such as known modes, experimental observables, summary statistics, and/or reference MCMC samples as a gold standard. While data-driven eval is valuable, data-free eval has compelling conceptual advantages, raising the question of how well data-free eval could work. Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among i) mode-covering metric, ii) stable with finite variance, iii) no dominance cycles. We situate the implications of this trilemma in the broader conceptual landscape of data-driven and data-free sampler evaluation.
--- -->

## Introduction

Sampling from unnormalized energy potentials is a key problem in probabilistic inference, statistical physics, and molecular dynamics. In recent years, deep generative modeling approaches for sampling has been investigated in diffusion samplers<d-cite key="liu2025adjoint"></d-cite> as well as generative flow networks<d-cite key="towardsunderstandinggflownets"></d-cite> and normalizing flows<d-cite key="schopmans2025temperatureannealed"></d-cite>. 
These models aim to learn a generative model distribution $q_\theta(x)$ that approximates a known unnormalized density. For instance, in molecular dynamics, the energy potential $U(x)$ of molecular conformations is known analytically in closed-form, and we aim to sample the Boltzmann distribution $p(x) \propto \exp(-U(x))$. Mode discovery, possibly by generalizing from known modes to efficiently discover new modes, is a critical aspect to evaluate in neural samplers, because traditional MCMC methods can mix poorly for rugged, high dimensional densities.

There are two main approaches for evaluating how well a sampler matches the target distribution: data-free, and data-driven. In data-free sampler eval, we only have the target unnormalized density, and consider metrics like KL divergence, kernelized maximum mean discrepancy, and Stein discrepancy. 

In data-driven sampler eval, we have access to additional data about $p$ beyond its unnormalized density. For example, synthetic eval settings can be constructed with a known number of modes and locations, which can be used to count how many modes neural samplers recover<d-cite key="10.5555/3692070.3692239"></d-cite>. In molecular dynamics, molecules like alanine dipeptide (~20 atoms) and chignolin (~200 atoms) are deeply understood with known modes. For more complex molecules, experimental observables such as protein folding stability can be used for evaluation<d-cite key="doi:10.1126/science.adv9817"></d-cite>, but this can conflate sampler evaluation with misalignments between the target potential's model of reality and actual reality. For example, we might sample the target potential perfectly, but still fit observables poorly, because the target potential is imperfect with respect to reality. In other cases, for instance molecules that are less well understood, samplers may be evaluated using reference MCMC samples used as a "gold standard". However, it can be unclear how accurate these reference MCMC samples are, especially for target densities that are highly challenging to sample. In the most challenging sampling problems with no prior knowledge, data-driven eval faces a "catch-22" situation where we need trustworthy samples to evaluate whether our samples are trustworthy, making it difficult to self-bootstrap off the ground. 

While data-driven eval is valuable, the upsides of data-free sampler eval are appealing.
In an ideal world where data-free sampler eval worked perfectly, research on neural samplers could be performed in gym-like environments with little overhead for supporting a huge diversity of target potentials, like with virtual environments or video games in reinforcement learning research.
When training environments can be defined purely computationally, neural samplers might be pre-trained on an endless variety of target distributions, towards meta-learning samplers that may generalize to efficiently sample unseen distributions. 
These considerations motivate asking how well data-free sampler eval could work.

Here, we prove an impossibility trilemma for data-free sampler evaluation; we can only have two among: i) mode-covering metric, ii) stable with finite variance, iii) allows sampler ranking without cyclic dominance (disallows A>B>C>A).

## Informal overview of trilemma

Data-free sampler evaluation is challenging because we have samples only from the model, and not from the target distribution (otherwise the sampling problem is solved). With model samples and likelihoods, we can stably estimate the reverse KL $$ \mathbb{E}_{q}[\log(q/p)] $$, but this is mode-seeking -- the reverse KL strongly rewards $q$ matching $$ p $$ among model samples, and does not strongly penalize missing modes in $$ p $$. If we hill-climb the reverse KL as an evaluation metric, we would generally reward samplers that fit a subset of modes very well, even if they are missing other important target modes, over samplers that discovered more target modes. This means the reverse KL is not a very useful sampler evaluation metric because it ignores the problem of mode discovery.

The forward KL $$ \mathbb{E}_{p}[\log(p/q)] $$ is mode-covering: it strongly rewards the model for covering modes of $$ p $$, making it ideal for evaluating mode discovery. Unfortunately, it is unstable to estimate. With access only to model samples, we require importance reweighting to estimate it as $$ \mathbb{E}_{q}[(p/q) \log(p/q)] $$. This theoretically does not have bounded, finite variance, and in practice is prohibitively high variance in high-dimensional settings of interest to be a useful evaluation metric.

{% include figure.liquid path="assets/img/2026-04-27-sampler-eval-trilemma/rkl-fkl.png" class="img-fluid" %}
<div class="caption">
    Reverse KL is mode-seeking, while forward KL is mode-covering. Axes depict mean and std parameters for a Gaussian. Values plot discrepancy to a two-Gaussian mixture with mean, std depicted with the red x's.
</div>

The contrast between the reverse KL and forward KL introduces the tension between items i) mode-covering metric, and ii) stable with finite variance, in our trilemma. By thinking beyond single-model eval metrics to pairwise comparators, which evaluate whether one model is better than the other, we can design metrics that are stable and mode-covering (among the support of both models). This pairwise comparator could thus score if a sampler $q_1$ is more mode-covering than $q_2$ head-to-head, while ignoring target modes that are unseen by both samplers. Unfortunately, we will show that such pairwise comparators introduce the third element of the trilemma: they lose universal ranking, which means they can introduce dominance cycles, and/or do not have pool independence.

## Trilemma

In this section, we provide a more precise characterization of the trilemma. First, we set up a definition of a "mode-covering" metric via sensitivity analysis. We then offer a short proof of the aforementioned dilemma between mode-covering and importance weights for single-model evaluation metrics. We then consider pairwise model comparison evaluation metrics, which can achieve both mode-covering and stability, but at the cost of the third item of the trilemma.

Technical note: Our introduced setting assumes access only to the unnormalized density of $p$. However, in the remainder of this note, we focus on comparing samplers for a fixed target distribution, where we can safely ignore the unknown normalizing constant $Z$. Thus, for notation simplicity, we work directly with $p$ instead of introducing a different symbol for its unnormalized density.

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

**Lemma**: Let the evaluation metric be a function $\mathcal{D}(p, q)$ defined as an integral over the domain, for any integrand:

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
\Omega_\epsilon = \{ x : q_1(x) > \epsilon ~~ \text{or} ~~ q_2(x) > \epsilon \}
$$

Consider this pairwise comparator $\mathcal{D}(p, q_1, q_2)$ that compares two models $q_1$ and $q_2$. 

$$
\int_{ \Omega_\epsilon } p(x) \log \frac{q_1(x)}{ q_2(x)} ~dx
$$

which can be estimated from samples $x_i \sim m(x)$ as:

$$
\sum_{i=1}^N \left[ \mathbb{1} \left( m(x_i) > \epsilon \right) \frac{p(x_i)}{m(x_i)} \log \frac{q_1(x_i)}{q_2(x_i)} \right]
$$

On the shared support $\Omega_\epsilon$, this estimate is related to the difference in  forward KL. It is mode-covering in the mixture support, as the sensitivity of the integrand to a mode of $p$ found by $q_1$ is:

$$
-\frac{\partial p(x) \log(q_1(x) / q_2(x)) }{\partial q_2(x)} = O\left( \frac{p}{q_2} \right)
$$

Note that this pairwise comparator is blind to modes missed by both models, though this is a reasonable property in an evaluation metric. Further, by limiting importance weights to $\Omega_\epsilon$ such that $m(x_i) > \epsilon/2$, the importance weights are capped and do not explode to infinity. The estimator thus has bounded variance, improving the stability of this metric.

Finally, we note that this pairwise comparator is presented primarily for exposition, and should not be considered a bullet proof proposal, until its properties such as being proper (optimal) for a scaled version of $p$<d-footnote>For any fixed $\Omega$ (determined by the choice of two models), the optimal distribution scored by the pairwise comparator is a scaled version of $p$ on $\Omega$ with zero density outside of $\Omega$.</d-footnote>, and its computation on un-normalized truncated model densities<d-footnote>Denote $\tilde{p}, \tilde{q_1}, \tilde{q_2}$ as the normalized densities on $\Omega$. Then, the true forward KL difference on the restricted set $KL(\tilde{p}\|\tilde{q_1}) - KL(\tilde{p}\|\tilde{q_2}) = \frac{1}{Z_p} \mathcal{D}(p, q_1, q_2) + \log(\frac{Z_{q_1}}{Z_{q_2}})$, where $Z_p = \int_{\Omega} p(x)~dx$ and similarly for $Z_{q_1}, Z_{q_2}$. Overall, the pairwise comparator favors models that put more total probability mass into $\Omega$, which is a reasonable property.</d-footnote> are more deeply studied.

### Loss of transitivity

Sounds great, right? This pairwise comparator is both stable and mode-covering. What is the cost of this? Our pairwise comparator is no longer decomposable into the difference of single model evaluation metrics: $\mathcal{D}(p, q_1, q_2) \neq h(p, q_1) - h(p, q_2)$. In this situation, we lose a guarantee on transitivity. There can exist sets of samplers where $A>B>C>A$, forming a dominance cycle. This happens because the "evaluation set" $\Omega_\epsilon$ is dynamic and depends on the comparison participants.

**Lemma**: There exists a set of samplers that form a dominance cycle when scored by this pairwise comparator.

**Proof**. We provide a proof by construction. Consider a target $p$ with three equal Gaussian modes at -5, 0, and +5. We have three samplers A, B, C, each with the same shape. Sampler A places large mass on the mode -5, small mass on mode 0, and no mass on mode +5. Sampler B and C are like sampler A but rotated on the modes (see figure). Here, epsilon masks no mass and small mass.

Comparing A to B, the union support is modes -5 and 0, with equal target $p$ mass on both, but A beats B on -5 more than B beats A on 0. Thus overall, A wins. Applying the same argument, B>C, and C>A. This yields A>B>C>A, a dominance cycle. We confirmed this argument holds experimentally by simulation, and provide code in the appendix.

{% include figure.liquid path="assets/img/2026-04-27-sampler-eval-trilemma/cycle_plot.png" class="img-fluid" %}
<div class="caption">
    A constructed example of a dominance cycle, with three samplers A>B>C>A.
</div>

We remark that losing the guarantee of transitivity does not mean that the pairwise comparator will commonly encounter dominance cycles. For realistic evaluation settings, dominance cycles are likely rare, or if they do occur due to "merry-go-round" mode discovery, this cause can be identified and understood to build better samplers. Furthermore, non-transitive pairwise comparisons can still be usefully ranked in a leaderboard via ELO scores<d-cite key="chiang2024chatbotarenaopenplatform"></d-cite>.

### General properties of pairwise comparators

Three properties of a pairwise comparator:

- Transitivity guarantee (no dominance cycles)
- Completeness (all pairs are comparable)
- Pool independence (results do not depend on the identity of all opponents)

are all jointly satisfied if and only if the pairwise comparator can be decomposed into single-model evaluation metrics: $\mathcal{D}(p, q_1, q_2) = h(p, q_1) - h(p, q_2)$. The connections between preference relations and utility functions has been studied in microeconomics Debreu's theorems<d-cite key="Debreu2008-dd"></d-cite>, Luce's Choice Axiom, and Bradley-Terry models<d-cite key="Bradley1952-tn"></d-cite>.

In our example above, we studied a particular pairwise comparator where the evaluation set depended on the support of the mixture distribution, which violates the single-model decomposition, and thereby breaks the guarantee of transitivity, as well as pool independence (as the exact score depends on the opponent).

For a leaderboard ranking of a set of models, we can improve transitivity among the model set by evaluating using the joint support of all models. Then, among models on the leaderboard, transitivity is retained. However, we still lack pool independence, so whether A>B can flip when a new model C is added to the leaderboard.

## Limitations

Our analysis has only considered single-model eval metrics expressible as integrals over the domain. While this family is fairly broad and such metrics are directly compatible with evaluation with iid model samples, it is possible that other families of single-model eval metrics can exist with improved properties.

Our working definition of "mode-covering" considers point-wise partial derivatives $O(p/q)$, which matches that of the forward KL. It could be possible that more mathematically relaxed definitions of mode-covering could still be useful for scoring mode discovery, and could lead to sampler eval metrics with improved properties that may "bypass" the trilemma as proved here.

## Discussion

In this work, we investigated the design space of data-free sampler evaluation metrics.
We first showed that for single-model eval, there is an impossibility dilemma between stability and mode-covering.
We then showed that by moving from single-model eval to pairwise comparators, we can achieve both stability and mode-covering, but lose universal ranking, which can manifest in losing some or all of: transitivity guarantee, completeness, and pool independence.

In particular, we studied a specific pairwise comparator that estimates the forward KL difference on the support of the mixture distribution with non-vanishing probability from either model. We showed that this pairwise comparator loses the transitivity guarantee, by constructing an example target where three samplers beat each other in a rock-paper-scissors cycle. Nevertheless, we expect that such dominance cycles might be rare in practice. Furthermore, ELO rankings can be used to convert pairwise scores into leaderboard rankings, even if dominance cycles exist.

We further discussed the possibility of a leaderboard that compares models on the joint support of all models submitted to the leaderboard. In this option, we ensure that transitivity holds for models in the leaderboard, but we still violate pool independence, wherein model rankings can flip when new models are added to the leaderboard.

Evaluation is an important engine for driving research progress, and sampler evaluation is a particularly challenging area to set up evaluations. In this note, we proved the impossibility of simultaneously achieving many desirable properties for an eval metric, which are easily achieved, and perhaps taken for granted, in other subfields. For example in computer vision, FID is stable, relevant to image quality (the analogue to mode-covering for sampler eval), and produces leaderboards with universal ranking, guaranteeing transitivity, pool independence, and completeness.

We hope that this note highlights underexplored aspects of the design space of data-free sampler evaluation methods, and may spur the community to think and discuss more about eval design, as well as desiderata on the best subset of properties, and agreement on which properties to sacrifice, for improved eval metrics to drive research progress.

## Code

This code demonstrates three samplers with a dominance cycle using our pairwise comparator.

{% highlight python linenos %}
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

class GaussianMixture:
    def __init__(self, means, stds, weights):
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.weights = np.array(weights) / np.sum(weights)
        
    def pdf(self, x):
        prob = np.zeros_like(x)
        for m, s, w in zip(self.means, self.stds, self.weights):
            prob += w * norm.pdf(x, loc=m, scale=s)
        return prob
    
    def sample(self, n_samples):
        # Choose component indices
        indices = np.random.choice(len(self.weights), size=n_samples, p=self.weights)
        # Sample from selected components
        samples = np.random.normal(
            loc=self.means[indices],
            scale=self.stds[indices]
        )
        return samples

def pairwise_snis_score(q1, q2, target_p, n_samples=50000, epsilon_threshold=1e-3):
    """
    Computes the Pairwise SNIS score: Score(q1) - Score(q2)
    Positive value means q1 is better than q2.
    """
    # 1. Sample from the mixture m = 0.5*q1 + 0.5*q2
    # We simulate this by sampling n/2 from q1 and n/2 from q2
    n_half = n_samples // 2
    samples_q1 = q1.sample(n_half)
    samples_q2 = q2.sample(n_half)
    x = np.concatenate([samples_q1, samples_q2])
    
    # 2. Evaluate densities
    prob_p = target_p.pdf(x)
    prob_q1 = q1.pdf(x)
    prob_q2 = q2.pdf(x)
    
    # 3. Compute mixture density m(x)
    prob_m = 0.5 * prob_q1 + 0.5 * prob_q2
    
    # 4. Filter by Threshold (The "Blindness" condition)
    # Only keep samples where the mixture density is significant
    mask = prob_m > epsilon_threshold
    
    if np.sum(mask) == 0:
        return 0.0 # No visible overlap
        
    x_valid = x[mask]
    prob_p_valid = prob_p[mask]
    prob_m_valid = prob_m[mask]
    prob_q1_valid = prob_q1[mask]
    prob_q2_valid = prob_q2[mask]
    
    # 5. Compute Self-Normalized Importance Weights
    # w = p / m
    log_weights = np.log(prob_p_valid + 1e-10) - np.log(prob_m_valid + 1e-10)
    weights = np.exp(log_weights)
    
    # Normalize weights (SNIS step)
    norm_weights = weights / np.sum(weights)
    
    # 6. Compute Weighted Difference of Log Likelihoods
    # Delta = log(q1) - log(q2)
    # Add tiny constant to avoid log(0)
    log_diff = np.log(prob_q1_valid + 1e-10) - np.log(prob_q2_valid + 1e-10)
    
    score = np.sum(norm_weights * log_diff)
    
    return score

def run_simulation():
    np.random.seed(42)
    
    # --- Configuration ---
    # Target P: 3 Equal Modes at -5, 0, 5
    means = [-5, 0, 5]
    std = 0.6
    p = GaussianMixture(means, [std]*3, [1/3, 1/3, 1/3])
    
    # --- The "Leakage" Cycle Construction ---
    # We design the weights such that:
    # A dominates B on Mode 1.
    # B dominates A on Mode 2 (but less severely because A leaks there).
    # Mode 3 is "invisible" to the A vs B metric (below threshold).
    
    # Weights format: [Mode 1, Mode 2, Mode 3]
    # A: High on 1, Leaks on 2, Blind on 3
    w_a = [0.85, 0.15, 0.00] 
    
    # B: High on 2, Leaks on 3, Blind on 1 (Cyclic Shift)
    w_b = [0.00, 0.85, 0.15]
    
    # C: High on 3, Leaks on 1, Blind on 2 (Cyclic Shift)
    w_c = [0.15, 0.00, 0.85]
    
    q_a = GaussianMixture(means, [std]*3, w_a)
    q_b = GaussianMixture(means, [std]*3, w_b)
    q_c = GaussianMixture(means, [std]*3, w_c)
    
    # --- Threshold Selection ---
    # Max density of a single gaussian ~0.66.
    # Mixture density at Main mode ~ 0.5 * 0.85 * 0.66 ≈ 0.28
    # Mixture density at Leak mode ~ 0.5 * (0.85 + 0.15) * 0.66 ≈ 0.33
    # Mixture density at Blind mode ~ 0.5 * (0.00 + 0.15) * 0.66 ≈ 0.05
    # We need a threshold roughly between 0.05 and 0.28 to hide the Blind mode.
    # Let's pick 0.1.
    epsilon = 0.1

    print(f"--- Running Validation (Threshold = {epsilon}) ---")
    
    # --- Run Comparisons ---
    # 1. A vs B
    score_ab = pairwise_snis_score(q_a, q_b, p, epsilon_threshold=epsilon)
    print(f"Match 1 (A vs B): Score = {score_ab:.4f}  => {'A Wins' if score_ab > 0 else 'B Wins'}")
    
    # 2. B vs C
    score_bc = pairwise_snis_score(q_b, q_c, p, epsilon_threshold=epsilon)
    print(f"Match 2 (B vs C): Score = {score_bc:.4f}  => {'B Wins' if score_bc > 0 else 'C Wins'}")
    
    # 3. C vs A
    score_ca = pairwise_snis_score(q_c, q_a, p, epsilon_threshold=epsilon)
    print(f"Match 3 (C vs A): Score = {score_ca:.4f}  => {'C Wins' if score_ca > 0 else 'A Wins'}")
    
    print("-" * 30)
    if score_ab > 0 and score_bc > 0 and score_ca > 0:
        print("RESULT: Strict Dominance Cycle Confirmed (A > B > C > A)")
    else:
        print("RESULT: Cycle not found (Check parameters)")

    # --- Visualization ---
    x_plot = np.linspace(-8, 8, 1000)
    plt.figure(figsize=(12, 5))
    
    # Plot Target
    plt.plot(x_plot, p.pdf(x_plot), 'k--', linewidth=2, label='Target p', alpha=0.3)
    
    # Plot Samplers
    plt.plot(x_plot, q_a.pdf(x_plot), 'r-', label='Sampler A (High 1, Leak 2)')
    plt.plot(x_plot, q_b.pdf(x_plot), 'g-', label='Sampler B (High 2, Leak 3)')
    plt.plot(x_plot, q_c.pdf(x_plot), 'b-', label='Sampler C (High 3, Leak 1)')
    
    # Plot Threshold visualizer for A vs B
    m_ab = 0.5*q_a.pdf(x_plot) + 0.5*q_b.pdf(x_plot)
    plt.fill_between(x_plot, 0, 0.05, where=(m_ab < epsilon), color='gray', alpha=0.3, label='Blind Spot (A vs B)')
    
    plt.axhline(y=epsilon, color='orange', linestyle=':', label='Threshold Epsilon')
    
    plt.title("The Leakage Cycle Construction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('leakage_cycle_plot.png')
    print("\nPlot saved to 'leakage_cycle_plot.png'")

if __name__ == "__main__":
    run_simulation()
{% endhighlight %}
