---
layout: distill
title: Model Misspecification in Simulation-Based Inference - Recent Advances and Open Challenges
description:
  Model misspecification is a critical challenge in simulation-based inference (SBI),
  particularly in neural SBI methods that use simulated data to train flexible neural
  density estimators. These methods typically assume that simulators faithfully represent
  the true data-generating process, an assumption that is often violated in practice.
  Resulting discrepancies can make observed data effectively out-of-distribution relative
  to the simulations, leading to biased posterior distributions and misleading uncertainty
  quantification. This post reviews recent work on model misspecification in neural SBI,
  covering formal definitions, methods for detection and mitigation, and their underlying
  assumptions. It also discusses practical implications for SBI workflows and outlines
  open challenges for developing robust SBI methods that remain reliable in realistic,
  imperfectly specified applications.
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
bibliography: 2026-04-27-model-misspecification-in-sbi.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.

toc:
  - name: Introduction
  - name: A Concrete Example - SIR Model with Weekend Reporting Delay
  - name: Defining Model Misspecification
    subsections:
    - name: Model Misspecification in Simulation-Based Inference
  - name: Mitigating Model Misspecification in SBI
    subsections:
    - name: Learning Explicit Misspecification Models
    - name: Detecting Misspecification with Learned Summary Statistics
    - name: Learning Misspecification-Robust Summary Statistics
    - name: Addressing Misspecification with Optimal Transport
  - name: Practical Implications for SBI Workflows
  - name: Open Challenges
---

Simulation-based inference (SBI) provides a powerful framework for applying Bayesian
inference to complex scientific simulators where direct likelihood computation is
infeasible <d-cite key="cranmer_frontier_2020"></d-cite>. By requiring only simulated
data to approximate the posterior distribution over simulator parameters,
$p(\mathbf{\theta}\mid \mathbf{x_o})$, SBI has found applications across neuroscience,
physics, climate science, and epidemiology <d-cite
key="goncalves_training_2020,brehmer_simulationbased_2020,mckinley2014simulation"></d-cite>.
All of these applications rely on a critical assumption: that the simulator faithfully
represents the true data-generating process. When this assumption is violated, the
resulting model misspecification can undermine the reliability of inference.

This issue is particularly acute in _neural_ SBI, where neural networks are trained on
simulated data to approximate posterior distributions, likelihoods, or likelihood
ratios. Neural networks are known to produce arbitrarily incorrect predictions when
queried with out-of-distribution (OOD) inputs <d-cite
key="szegedy_intriguing_2014"></d-cite>. Under model misspecification, the observed data
$\mathbf{x}_o$ can be effectively OOD relative to the training simulations, leading to
biased posterior estimates and misleading uncertainty quantification. Accordingly,
empirical studies confirm that even seemingly minor mismatches between simulator and
reality can induce substantial parameter bias and severely miscalibrated credible
intervals <d-cite key="cannon_investigating_2022"></d-cite>.

These observations motivate methods that detect and mitigate model misspecification in
SBI, rather than assuming perfectly specified simulators. This blog post reviews recent
advances in this area for neural SBI. It introduces a concrete running example that
illustrates key concepts, formalizes the notion of model misspecification in SBI,
surveys four categories of methods for addressing misspecification, discusses their
assumptions and practical implications, and concludes with open challenges for the
field.

## A Concrete Example - SIR Model with Weekend Reporting Delay

Before formalizing these concepts, we introduce a concrete running example: the
Susceptible-Infected-Recovered (SIR) epidemic model with weekend reporting delays
<d-cite key="ward_robust_2022"></d-cite>. The SIR model describes disease spread through
infection and recovery rates $(\beta, \gamma)$, with reproduction number
$R_0 = \beta/\gamma$. In the clean simulator, infection reports occur uniformly across
all days. In contrast, real-world data often exhibit systematic patterns—here, a
fraction $\alpha$ of weekend infections goes unreported until Monday, creating
characteristic weekly oscillations (Figure 1).

{% include figure.html path="assets/img/2026-04-27-model-misspecification-in-sbi/sir_figure_row.png" class="img-fluid" %}
<div class="caption">
  <strong>Figure 1:</strong> SIR model with weekend reporting delay.
  <strong>A</strong>: Schematic SIR model and misspecification: a fraction α of weekend infections is reported on Monday.
  <strong>B</strong>: Example epidemic trajectory showing true S, I, R curves and observed infections with a weekend delay.
  <strong>C</strong>: SBI posterior samples for $(\beta, \gamma)$ for clean data (α = 0%, blue) and misspecified data (α=30%, orange), with true parameters marked.
  <strong>D</strong>: Posterior predictive checks for both cases, illustrating how misspecification shifts and distorts the inferred dynamics. Results generated using the `sbi` package and specificatins from Cannon et al..
</div>

To infer infection parameters from observed data, a common SBI approach would be using
neural posterior estimation (NPE, <d-cite key="papamakarios_fast_2016"></d-cite>),
training a conditional density estimator $q_{\phi}(\theta \mid x)$ on clean simulated
parameter–data pairs. When NPE is trained on clean simulations but encounters
observations with weekend delays, the network faces out-of-distribution data. Figure 1
demonstrates the consequences: even with mild misspecification (α=20%), the posterior
shifts away from true parameters, uncertainty increases, and posterior predictive
samples fail to capture the systematic Monday spikes. Ward et al. (2022) <d-cite
key="ward_robust_2022"></d-cite> quantify this effect, showing that such discrepancies
can bias parameter estimates by over 40% and cause credible interval coverage to drop
from the nominal 95% to below 60%.

**Note on the example:** This scenario is designed for pedagogical illustration—in
practice, if the weekend reporting delay were known, practitioners would explicitly
model it in the simulator. The methods discussed below target situations where
misspecification is unknown or difficult to parametrize, or where the goal is to develop
inference procedures that remain robust to unanticipated discrepancies.

Having seen the practical impact of misspecification in this example, we now turn to
formal definitions and a systematic review of approaches for addressing this challenge.

## Defining Model Misspecification

Model misspecification occurs when the assumptions of the model do not align with the
true data-generating process. In Bayesian inference, this problem arises when the true
data-generating process cannot be captured within the family of distributions defined by
the model. Walker (2013) provides a foundational definition <d-cite
key="walker_bayesian_2013"></d-cite>:

<blockquote>
  A statistical model $p(\mathbf{x}_s | \theta)$ that relates a parameter of interest
  $\theta \in \Theta$ to a conditional distribution over simulated observations
  $\mathbf{x}_s$ is said to be misspecified if the true data-generating process
  $p(\mathbf{x}_o)$ of the real observations $\mathbf{x}_o \sim p(\mathbf{x}_o)$ does not
  belong to the family of distributions $\{p(\mathbf{x}_s | \theta); \theta \in \Theta\}$.
</blockquote>

This structural definition provides a theoretical basis for understanding model
misspecification but does not yet specify how misspecification manifests in SBI workflows.

### Model Misspecification in Simulation-Based Inference

SBI is particularly sensitive to model misspecification because the model is defined
through a simulator, and inference relies entirely on simulator-generated data. Unlike
classical Bayesian inference, where the likelihood function is explicit, simulators in
SBI may introduce subtle discrepancies that propagate through the inference pipeline,
resulting in biased posterior estimates.

The consequences of model misspecification in SBI were first analyzed systematically by
Frazier et al. (2019) in the context of rejection sampling-based Approximate Bayesian
Computation (ABC) <d-cite key="frazier_model_2019"></d-cite>. They showed that, under
misspecification, ABC procedures concentrate on pseudo-true parameters that minimize
discrepancies between simulated and observed data, and that resulting credible sets can
exhibit distorted frequentist coverage. This work established that misspecification is
not merely a philosophical concern but has concrete implications for SBI workflows.

In _neural_ SBI methods, where neural networks approximate posterior distributions (or
likelihoods or likelihood ratios) based on simulations, the problem becomes particularly
acute. A popular approach is neural posterior estimation (NPE), where a neural network
learns a parametric approximation of the posterior distribution (e.g., a mixture of
Gaussians, a normalizing flow, or a diffusion model) using simulated data. Cannon et al.
(2022) <d-cite key="cannon_investigating_2022"></d-cite> conducted a first
comprehensive study of neural SBI algorithms under different forms of model
misspecification and found that performance can degrade severely: neural networks
trained on simulations can fail catastrophically when applied to observed data that lie
outside the training distribution—producing arbitrarily incorrect predictions with
overconfident uncertainty estimates—and existing mitigation strategies do not prevent
failure in all cases.

Before reviewing methods to mitigate misspecification in neural SBI, it is important to
distinguish between different sources of misspecification in the workflow:

1. **Misspecification of the Simulator:** The true data-generating process does not
   belong to the family of distributions induced by the simulator. This corresponds to
   the classical Bayesian notion of misspecification described by Walker (2013). For
   example, if a simulator lacks the capacity to model key features of the observed
   data, the resulting posterior may fail to capture the true parameter values
   accurately.
2. **Misspecification of the Prior:** Misspecification can also occur when the prior
   used in the inference process does not incorporate the "true parameter" underlying
   the data-generating process. Prior mismatch can distort posterior estimates, leading
   to inferences that reflect artifacts of the assumed prior rather than the true
   underlying process.

Prior misspecification is a general challenge in Bayesian inference and can be addressed
with standard Bayesian workflow tools like prior predictive checks <d-cite
key="gelman_bayesian_2020"></d-cite>. While it has received less attention in the
SBI-specific literature in general, there has been growing interest in this topic
recently <d-cite
key="wehenkel_addressing_2024,bockting_simulationbased_2024,bishop_learning_2025"></d-cite>.

<!-- Note that even with well-specified simulator and prior, the inference algorithm itself
may introduce errors, such as systematically biased posteriors or uncalibrated
uncertainty estimates due to neural network training issues. These implementation
quality concerns are typically addressed through calibration tests such as
simulation-based calibration <d-cite key="talts_validating_2020"></d-cite>, expected
coverage diagnostics <d-cite key="deistler_truncated_2022,miller_truncated_2021a"></d-cite>,
and classifier-based calibration
<d-cite key="zhao_diagnostics_2021,linhart_lc2st_2024"></d-cite>, which validate posterior
accuracy assuming the simulator is correct. -->

The methods reviewed below, and much of the recent literature on model misspecification
in neural SBI, primarily address the first case: detecting and mitigating
simulator-related misspecification. The remainder of this post provides an overview of
these approaches.

## Mitigating Model Misspecification in SBI

Recent works have introduced a range of methods to address model misspecification in
simulation-based inference (SBI). These approaches can be broadly categorized into four
strategies: learning explicit mismatch models, detecting misspecification through
learned summary statistics, learning misspecification-robust statistics, and aligning
simulated and observed data using optimal transport. Each method has unique strengths
and limitations, which we summarize below.

### Learning Explicit Misspecification Models

{% include figure.html path="assets/img/2026-04-27-model-misspecification-in-sbi/ward_et_al.png" class="img-fluid" %}
<div class="caption">
    Figure 2 (adapted from <d-cite key="ward_robust_2022"></d-cite>): Visualization of the robust neural posterior estimation (RNPE) framework.
</div>

Ward et al. (2022) <d-cite key="ward_robust_2022"></d-cite> propose **Robust Neural
Posterior Estimation (RNPE)**, an extension of neural posterior estimation (NPE) that
addresses misspecification by explicitly modeling discrepancies between simulated and
observed data. RNPE introduces an error model $p(\mathbf{y} \mid \mathbf{x})$, where
$\mathbf{y}$ denotes observed data and $\mathbf{x}$ simulated data, and combines it with
a marginal density model $q(\mathbf{x})$ trained on simulations. A standard NPE is
trained on $(\theta, \mathbf{x})$ pairs, while at inference time Monte Carlo sampling
from $p(\mathbf{x} \mid \mathbf{y}) \propto q(\mathbf{x}) p(\mathbf{y} \mid \mathbf{x})$
produces denoised latent variables $\mathbf{x}_m$ that are passed through NPE to
approximate $p(\theta \mid \mathbf{x}_m)$.

The results in <d-cite key="ward_robust_2022"></d-cite> demonstrate that RNPE can
substantially improve robustness to misspecification on several benchmark tasks and an
epidemic application. By explicitly modeling the error for each data dimension, the
approach also facilitates model criticism, highlighting features of the data that are
likely to be misspecified. However, performance hinges on choosing a suitable error
model (such as a spike-and-slab distribution), which may not generalize across
scenarios, and the additional inference steps make the method computationally demanding,
particularly for high-dimensional observations.

In the SIR weekend-delay example, RNPE uses a spike-and-slab error model to capture the
Monday aggregation effect. Ward et al. show that this effectively denoises the Monday
spikes back to plausible weekend values, recovering parameter estimates within about 5%
of ground truth, compared to roughly 40% bias with standard NPE.

### Detecting Misspecification with Learned Summary Statistics

{% include figure.html path="assets/img/2026-04-27-model-misspecification-in-sbi/schmitt_et_al.png" class="img-fluid" %}
<div class="caption">
    Figure 3 (adapted from <d-cite key="schmitt_detecting_2024"></d-cite>): Simulated data is used to train a neural network to map into a latent space designed to detect misspecification. At inference time, the observed data is embedded mapped into the latent space to detect misspecification.
</div>

Schmitt et al. (2024) <d-cite key="schmitt_detecting_2024"></d-cite> focus on
_detecting_ misspecification using learned summary statistics. The method employs a
summary network, $h_\psi(\mathbf{x})$, to encode both observed and simulated data into a
structured summary space, typically following a multivariate Gaussian distribution.
Discrepancies between distributions in this space are quantified using metrics like
Maximum Mean Discrepancy (MMD), with significant divergences indicating
misspecification.

The training procedure for this approach remains the same as in standard neural SBI
methods except for an additional MMD term in the NPE loss function:

$$
\mathcal{L}_{\phi, \psi} = \mathcal{L}_{\text{inference}}(\phi) + \lambda \cdot
\text{MMD}^2[p(h_{\psi}(\mathbf{x})), \mathcal{N}(\mathbf{0}, \mathbb{I})].
$$

Intuitively, the additional MMD loss term encourages the embedding network to obtain a
Gaussian structure in the latent summary space, while not directly affecting the quality
of the posterior estimation ensured by the standard NPE loss <d-cite
key="schmitt_detecting_2024"></d-cite>. At inference time, the learned embedding network
can then be used to detect misspecification for unseen, e.g., observed, data points.

This approach is adaptable to diverse data types and does not require explicit knowledge
of the true data-generating process. Additionally, it is amortized, i.e., it can be
applied to new observed data without re-training because the training does not depend on
$\mathbf{x}_o$. However, its performance depends on the design of the summary network and the
choice of divergence metric. While effective for detecting misspecification, it does not
directly correct for it, instead providing insights for iterative simulator refinement.

For the SIR weekend delay example, practitioners could apply embedding-based detection
to summary statistics (mean, median, maximum, day of maximum, day when half of
cumulative infections reached, and autocorrelation) or train an unconditional
normalizing flow over these statistics. Both approaches would quantitatively flag the
observed data as out-of-distribution, signaling that the simulator fails to capture
systematic patterns present in real epidemic data.

Beyond the embedding-based approach described above, another practical option for
detection is to learn the marginal distribution $p(\mathbf{x})$ directly using density
estimation—for instance, via normalizing flows. Trained on simulated data, the learned
density can then evaluate whether observed data $\mathbf{x}_o$ has anomalously low
log-probability, flagging it as out-of-distribution. This density-based approach is
conceptually straightforward but limited to relatively low-dimensional data, whereas the
embedding approach scales better to higher dimensions including time series. 

### Learning Misspecification-Robust Summary Statistics

Huang & Bharti et al. (2023) <d-cite key="huang_learning_2023"></d-cite> propose a
method for learning summary statistics that are both informative about parameters and
robust to misspecification. Their approach modifies the standard NPE loss function by
introducing a regularization term that balances robustness to misspecification with
informativeness:

$$
\mathcal{L} = \mathcal{L}_{\text{inference}} + \lambda \cdot \text{MMD}^2[h_\psi(\mathbf{x}_{s}), h_\psi(\mathbf{x}_{o})].
$$

Here, $h_\psi$ represents the summary network, $\mathbf{x}\_{s}$ and $\mathbf{x}\_{o}$ are
simulated and observed data, respectively, and $\lambda$ controls the trade-off between
inference accuracy and robustness. Unlike diagnostic methods, this approach directly
adjusts the summary network during training to mitigate the impact of misspecification
on posterior estimation.

Benchmarking results presented in Huang & Bharti et al. (2023) demonstrate improved
performance compared to the RNPE approach, with the additional advantage of
applicability to high-dimensional data. However, the method has several limitations. The
modified loss function introduces additional complexity, and its success depends on
selecting appropriate divergence metrics and regularization parameters, which often
require domain-specific tuning. Furthermore, because robustness is implicitly learned
during training and operates in the latent space, there is limited direct control over
how and where misspecification is mitigated.

### Addressing Misspecification with Optimal Transport

{% include figure.html path="assets/img/2026-04-27-model-misspecification-in-sbi/wehenkel_gamella_et_al.png" class="img-fluid" %}
<div class="caption">
    <strong>Figure 4:</strong> Schematic of RoPE <d-cite key="wehenkel_addressing_2024"></d-cite>. Standard NPE with embedding network is trained on potentially misspecified simulations; a calibration set is then used to fine-tune the embedding to the observed data and learn an optimal transport mapping, which is used to construct a misspecification-robust posterior as a weighted mixture of NPE posteriors.
</div>

Wehenkel & Gamella et al. (2024) <d-cite key="wehenkel_addressing_2024"></d-cite>
propose Robust Posterior Estimation (RoPE), which combines neural posterior estimation
(NPE) with optimal transport (OT) to address model misspecification. The approach
requires a calibration set of real-world observations with known ground-truth
parameters, for example from expensive experiments where parameters can be measured
directly while a cheaper simulator models only part of the process.

The core idea is to use OT to relate simulated and observed data in an embedding space
learned by NPE. A standard NPE is first trained on simulated $(\theta, \mathbf{x}\_s)$
pairs to obtain an embedding network and posterior estimator. The embedding is then
fine-tuned on the calibration set to better align simulated and observed data. At
inference time, OT computes a matching between embedded simulated data
$x_s^j$ and observed data $\mathbf{x}\_o$, yielding weights
$\alpha_{ij}$ that define a mixture of NPE posteriors,

$$
\tilde{p}(\theta \mid \mathbf{x}_o)
= \sum_{j=1}^{N_s} \alpha_{ij} \, q(\theta \mid \mathbf{x}_s^j),
$$

where $q(\theta \mid \mathbf{x}_s^j)$ denotes the NPE posterior for simulated input
$\mathbf{x}_s^j$. Increasing the number of simulated samples $N_s$ makes the mixture
more conservative and, in the limit, approach the prior. This underconfidence property
provides a mechanism to avoid overconfident posteriors under severe misspecification,
at the cost of broader, less informative posteriors.

In the SIR weekend-delay example, RoPE would use a calibration set of outbreaks with
known ground-truth transmission parameters and learn to align Monday spikes in real data
with weekend patterns in simulations, yielding corrected posteriors for new outbreaks
despite the simulator’s limitations.

While conceptually elegant, RoPE relies on calibration data, which may not be available
in all fields. Moreover, it is transductive: the OT problem is solved on a batch of test
observations, so inference for one observation depends on which others appear in the
batch and cannot be performed fully independently.

Addressing this limitation, Senouf et al. (2025) <d-cite
key="senouf_inductive_2025a"></d-cite> introduced an extension called FRISBI, which
shifts the OT step from test time to training time and makes the approach inductive and
amortized. During training, FRISBI applies mini-batch OT on the calibration set to learn
aligned embeddings and then trains a conditional normalizing flow to approximate the
resulting mixture posterior. At inference, a single forward pass yields a
misspecification-robust posterior, making FRISBI more scalable while preserving the
robustness properties of RoPE.

### Summary of Approaches

The methods discussed above tackle different facets of model misspecification in SBI,
ranging from explicit error modeling to the development of robust summary statistics and
the alignment of simulated and observed data distributions. While each approach
demonstrates unique strengths, their applicability depends on the specific
misspecification scenario, computational budget, and the availability of calibration
data.

It is useful to contrast what “robustness” means in each family of methods. RNPE aims
for robustness under an explicitly *parameterized error model*, providing reliable
inference when this model captures the dominant discrepancies between simulator and
reality. Detection-oriented methods based on embeddings or density estimation instead
provide robustness in the sense of *awareness*: they quantify when simulated and
observed data are incompatible but do not directly correct the posterior. Approaches
that learn misspecification-robust summary statistics trade some information for
stability, implicitly downweighting aspects of the data that are particularly sensitive
to simulator errors. OT-based methods such as RoPE and FRISBI promote robustness through
conservative posteriors aligned with calibration data, at the cost of broader posteriors
when misspecification is severe.

These approaches also rely on different assumptions, which shape their practical
applicability. RNPE requires an explicit error model and is most natural in low- to
medium-dimensional observation spaces where plausible mismatch mechanisms can be
formulated. Detection methods require only simulated and observed data but depend on
choices of summary network, density estimator, and divergence thresholds, and they are
most useful when the primary goal is diagnosis rather than automatic correction. Methods
based on robust summary statistics assume access to observed data during training and
hinge on regularization choices that encode the desired robustness–accuracy balance.
OT-based methods, in turn, rely critically on calibration data with known parameters and
on the assumption that optimal transport in an embedding space yields meaningful
correspondences between simulated and real observations; FRISBI additionally assumes
that the induced mixture posteriors can be well-approximated by an amortized
conditional density estimator.

Taken together, these differences illustrate that there is no single “best” approach to
model misspecification in SBI, but rather a spectrum of tools that make different
assumptions and offer different robustness properties. The next section discusses the
resulting practical implications for SBI workflows and outlines typical use cases and
trade-offs for the available methods.

## Practical Implications for SBI Workflows

Building on this comparison, the remarks in this section summarize typical use cases and
trade-offs rather than prescribing a fixed workflow. The choice of method depends on the
application, data characteristics, and available resources.

For *detection* of misspecification, a natural starting point is prior predictive checks,
which are often the most interpretable diagnostic. Quantitative approaches include
density estimation-based detection (learning $p(\mathbf{x})$ via normalizing flows,
typically limited to relatively low-dimensional data) and embedding-based detection
(using learned summary spaces with divergence metrics, which scale to higher-dimensional
settings). Both are implemented in popular toolboxes such as the `sbi` Python package
([documentation](https://sbi.readthedocs.io/en/latest/how_to_guide/18_model_misspecification.html))
and the `BayesFlow` package
([documentation](https://bayesflow.org/stable-legacy/_examples/Model_Misspecification.html)).

For *mitigation* of misspecification, the methods reviewed above target different
scenarios. RNPE is most effective when the structure of the misspecification can be
characterized and data dimensionality is moderate, offering interpretability through the
learned error model. Methods that learn robust summary statistics scale better to
high-dimensional data when the misspecification structure is unclear, though they
sacrifice amortization and introduce additional hyperparameters governing the
robustness–accuracy trade-off. RoPE and FRISBI leverage optimal transport for domain
alignment when calibration data (real observations with ground-truth parameters) is
available: RoPE is transductive and operates on batches, while FRISBI is fully inductive
and amortized, making it more scalable for per-sample inference once training is
complete.

The table below summarizes key characteristics of the different approaches to help
organize their typical use cases:

| Method | Primary Goal | Calibration Data? | Amortized? | Data Dimensionality | Best For |
|--------|--------------|-------------------|------------|---------------------|----------|
| **Detection (Flow)** | Identify misspecification | No | Yes | Low ($<$20D) | Diagnostics, low-dimensional data |
| **Detection (Embedding)** | Identify misspecification | No | Yes | Medium-High | Diagnostics, scalable detection |
| **RNPE** | Correct via error model | No | Yes | Low ($<$20D) | Known error structure, interpretability |
| **Robust Summary Stats** | Learn implicit robustness | Partial (observed $\mathbf{x}_o$) | No | High ($>$50D) | High-dimensional data, unclear misspecification |
| **RoPE** | Align distributions (transductive) | Yes (small set) | No | Any | Calibration data, batch inference |
| **FRISBI** | Align distributions (inductive) | Yes (small set) | Yes | Any | Calibration data, per-sample inference |

While this high-level overview helps to organize current approaches, it also highlights
several gaps in our understanding and tooling, which motivate the open challenges
discussed next.

## Open Challenges

The recent works outlined above have made significant progress in addressing model
misspecification in simulation-based inference (SBI), introducing methods for detecting
and mitigating its effects. However, the problem is far from resolved, and several key
challenges remain:

1. **Better Methods for Detecting and Addressing Model Misspecification:** Recent
   methods have improved our ability to diagnose and mitigate model misspecification,
   but important limitations remain. Many techniques target specific aspects of
   misspecification (e.g., discrepancies in summary statistics or shifts in data
   distributions via optimal transport) and often require additional assumptions,
   computational overhead, or prior knowledge about the misspecification. A central
   challenge is to develop more flexible and scalable methods that can

   - detect misspecification in a principled, data-driven manner, without relying
     heavily on hand-crafted summaries or manual tuning,
   - provide interpretable diagnostics that clarify the sources and consequences of
     misspecification, and
   - offer robust mitigation strategies that work across different misspecification
     regimes without requiring large additional datasets or costly corrections.

2. **A Common and Precise Definition of Model Misspecification in SBI:** As highlighted
   in this post, model misspecification in SBI can arise from different sources,
   including mismatches between the simulator and the true data-generating process and
   prior misspecification. A common and formally precise definition of these cases is
   essential for unifying the field. Such a framework would provide clarity for
   researchers and practitioners, enabling more systematic comparisons of methods and
   their applicability to specific types of misspecification.

3. **Common Benchmarking Tasks for Evaluating Methods:** Another obstacle to progress is
   the lack of an established set of benchmarking tasks tailored to the various forms of
   model misspecification. Current evaluations often focus on specific scenarios or
   datasets, limiting generalizability. There are promising developments—for instance,
   Wehenkel & Gamella et al. <d-cite key="wehenkel_addressing_2024"></d-cite> re-used
   tasks proposed by Ward et al. <d-cite key="ward_robust_2022"></d-cite> and introduced
   new tasks to probe different aspects of misspecification—but these efforts need to be
   integrated into a common benchmarking framework and made accessible through
   open-source software. Such a framework would enable rigorous, comparable evaluations
   under realistic misspecification conditions and encourage the development of methods
   that are robust across diverse settings.

4. **Comprehensive Practical Guidelines for Model Misspecification:** While initial
   guidance exists (such as the practical implications outlined above and in <d-cite
   key="deistler_simulationbased_2025a"></d-cite>), the field would benefit from a
   comprehensive workflow study on model misspecification in SBI, similar in scope to
   the Bayesian workflow of Gelman et al. <d-cite key="gelman_bayesian_2020"></d-cite>.
   The preliminary guidance already shows that assembling current methods into a coherent
   workflow involves many subjective choices, underscoring the need for more systematic
   studies. A dedicated workflow, systematically investigating different types of
   misspecification, their detection, and suitable mitigation strategies, would provide
   practitioners with actionable decision frameworks, including recommendations on how
   to diagnose misspecification, select methods based on problem characteristics, and
   interpret posteriors under potential misspecification.

Addressing these challenges will pave the way for more robust and practical SBI methods
capable of handling model misspecification effectively. A unified framework, rigorous
benchmarks, and practical guidelines will not only advance research on model
misspecification but also simplify its handling in applied settings. Together, these
efforts will strengthen SBI as a reliable tool for scientific inference in complex and
realistic scenarios.
