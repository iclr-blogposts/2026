---
layout: distill
title: Diffusion Guidance - Opportunities for Physical Sciences
description: Guidance has been a central driver of the success of diffusion models, enabling precise control over the sampling process toward desired target conditions. The most widely used techniques include Classifier Guidance and Classifier-Free Guidance. Recently, however, there has been growing interest in alternative guidance strategies. In this blog post, we review recent progress in training-free diffusion guidance methods and highlight their applications in scientific domains.

date: 2026-04-27
future: true
htmlwidgets: true
hidden: true
math: true

# Mermaid diagrams
mermaid:
  enabled: true
  zoomable: true

# Anonymize when submitting
# authors:
#   - name: Anonymous
authors:
  - name: Anonymous

bibliography: 2026-04-27-diffusion-guidance.bib

toc:
  - name: Introduction
  - name: Background
    subsections:
      - name: Diffusion Models
      - name: Conditional Diffusion Models
  - name: Classifier Guidance
  - name: Analytical Likelihoods
    subsection:
      - name: Diffusion Posterior Sampling (DPS)
  - name: Applications in physical sciences
    subsection:
      -name: DiffusionPDE
      -name: Inequality constraints for rare event sampling
  - name: Closing takeaways

_styles: >
  :root {
    --color-unconditional: #ff0000;
    --color-posterior: #004e64;
    --color-prior: #25a18e;
    --color-likelihood: #00a5cf;
  }
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

  /* Details animation */
  details {
    overflow: hidden;
    background-color: #f8f9fa !important;
    border-radius: 4px;
    margin: 0;
  }

  details summary {
    background-color: #f8f9fa !important;
    padding: 8px 12px;
    cursor: pointer;
  }

  details[open] {
    background-color: #f8f9fa !important;
  }

  details[open] summary {
    background-color: #f8f9fa !important;
  }

  details[open] summary ~ * {
    animation: slideDown 0.3s ease-out;
  }

  @keyframes slideDown {
    from {
      opacity: 0;
      transform: translateY(-10px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
---

{% include figure.liquid path="assets/img/2026-04-27-diffusion-guidance/diff_prior.png" class="img-fluid" %}

$$
% Define macros for common notation
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bc}{\mathbf{c}}
\newcommand{\bu}{\mathbf{u}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bI}{\mathbf{I}}
\newcommand{\w}{\mathbf{w}}
\newcommand{\bzt}{\mathbf{z}_t}
\newcommand{\xs}{\mathbf{x}_s}
\newcommand{\xzero}{\mathbf{x}_0}
\newcommand{\bf}{\mathbf{f}}
\newcommand{\gt}{\mathrm{g}_t}
\newcommand{\fg}{\mathrm{g}}
\newcommand{\ff}{\mathrm{f}}
\newcommand{\at}{\alpha_t}
\newcommand{\st}{\sigma_t}
\newcommand{\eps}{\boldsymbol{\epsilon}}
\newcommand{\thetav}{\boldsymbol{\theta}}
\newcommand{\muv}{\boldsymbol{\mu}}
\newcommand{\sigmav}{\boldsymbol{\sigma}}
\newcommand{\logp}{\log p}
\newcommand{\score}{\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)}
\newcommand{\scoreprior}{\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t)}
\newcommand{\scorepost}{\nabla_{\mathbf{x}_t} \log p(\mathbf{x}_t \mid \mathbf{y})}
\newcommand{\scorelike}{\nabla_{\mathbf{x}_t} \log p(\mathbf{y} \mid \mathbf{x}_t)}
\newcommand{\posterior}{p(\mathbf{x} \mid \mathbf{y})}
\newcommand{\prior}{p(\mathbf{x})}
\newcommand{\likelihood}{p(\mathbf{y} \mid \mathbf{x})}
\newcommand{\evidence}{p(\mathbf{y})}
\newcommand{\E}{\mathbb{E}}
\newcommand{\Var}{\mathbb{V}}
\newcommand{\Cov}{\mathrm{Cov}}
\newcommand{\dt}{\mathrm{d}t}
\newcommand{\dw}{d\mathbf{w}}
\newcommand{\SDE}{d\bzt}
\newcommand{\normal}{\mathcal{N}}
\newcommand{\uniform}{\mathcal{U}}
$$

## Introduction

Diffusion models have emerged as a state-of-the-art approach for sampling from complex probability distributions. Prominent examples are image-generation models like Stable Diffusion, where the model generates a high-quality image within seconds based on a given text prompt like _"A corgi with sunglasses on the beach"_. What makes these models stand out is their ability to faithfully adhere to a given prompt.

These capabilities arise from techniques collectively known as guidance, which direct the output of diffusion models toward specified conditions. To guide the models, we use a score function divided into a <strong style="color: #25a18e;">prior</strong> and <strong style="color: #00a5cf;">likelihood term</strong>:

$$
  \begin{equation*}
    \textcolor{#A125A1}{\nabla_{\bx} \log \, p(\bx \mid \by)} = \textcolor{#25a18e}{\nabla_{\bx}  \log p(\bx)} + \textcolor{#00a5cf}{\nabla_{\bx} \log p(\mathbf{y} \mid \bx)}
  \end{equation*}
$$

<div class="text-center">
{% include figure.liquid path="assets/img/2026-04-27-diffusion-guidance/score_decomposition.svg" class="img-fluid" max-width="25%" %}
</div>

This blog post aims to share insights into methods that go beyond traditional guidance techniques. We begin by explaining the fundamentals of the classifier guidance approach in the first section and then explore recent developments in guiding diffusion models. Our goal is not to favor any specific method but to present alternatives, especially useful when data is limited or training resources are constrained.

## Background

In this section, we will briefly summarize the key features of diffusion models. Readers familiar with the score-based formulation might want to skip ahead.

### Diffusion Models

A _forward_ diffusion process, which gradually destroys data over time, can be defined via a stochastic differential equation (SDE)<d-cite key="song2021generative"></d-cite>:

$$
  \begin{equation}
    d\bzt = \bf(\bzt, t) \; dt + \fg(t) \; \dw \textrm{,}
  \end{equation}
$$

where $\bf$ and $\fg$ are determined by the noise schedule, and $\dw$ is a Brownian motion. For an affine drift $\bf$, the forward process can be rewritten in closed form:

$$
  \begin{equation}
    \label{eq:diff_co}
    \bz_t = \alpha_t \bx + \sigma_t \boldsymbol{\epsilon} ,
  \end{equation}
$$

with $\bx \sim p_{\text{data}}(\bx)$ sampled from the data distribution and $\epsilon \sim \normal(\mathbf{0}, \mathbf{I})$ drawn from a standard gaussian distribution.
The forward diffusion processes starts with a clean data sample $\bz_0=\bx$ (where $\alpha_0=1$ and $\sigma_0=0$), and gradually adds noise.
In the case of a variance-exploding noise schedule, $\alpha_t$ remains $1$ and $\sigma_t$ grows with $t$.

<!-- <details>
  <summary>
    Noise Schedules
  </summary>

<b>Variance Preserving (VP)</b> SDE

$$
  \begin{aligned}
    \bf(\bzt,t) &= -\frac{1}{2}\Bigg (\frac{d}{dt}\log(1+e^{-\lambda_t}) \Bigg) \; \bzt \\
    \fg(t)^2 &= \frac{d}{dt} \log(1+e^{-\lambda_t}) \\
    \alpha^2_{\lambda} &= \textrm{sigmoid}(\lambda) \\
    \sigma^2_{\lambda} &= \textrm{sigmoid}(-\lambda) \\
    p(\mathbf{x}_1) &= \normal(0, \mathbf{I})
  \end{aligned}
$$

<b>Variance Exploding (VE)</b> SDE

$$
  \begin{aligned}
    \bf(\bzt,t) &= 0 \\
    \fg(t)^2 &= \frac{d}{dt} \log(1+e^{-\lambda_t}) \\
    \alpha^2_{\lambda} &= 1 \\
    \sigma^2_{\lambda} &= e^{-\lambda_t} \\
    p(\mathbf{x}_1) &= \normal(0, e^{-\lambda_{min}}\mathbf{I})
  \end{aligned}
$$

<d-cite key="kingma2023understanding"></d-cite>

</details> -->

<!-- Looking at the _reverse_ process<d-cite key="song2021generative,anderson1982reverse"></d-cite>, we find that it is also an SDE running backward in time: -->

The _reverse_ process<d-cite key="song2021generative,anderson1982reverse"></d-cite> is also an SDE running backward in time:

$$
  \begin{equation}
  \label{eq:reverse_process}
    \mathrm{d}\bzt = [\bf(\bzt, t) - \fg(t)^2 \; \textcolor{#25a18e}{\nabla_{\bzt} \log p(\bzt)}]\; dt + \fg(t) \; \dw ,
  \end{equation}
$$

where $\textcolor{#25a18e}{\nabla_{\bzt} \log p(\bzt)}$ is called the score function (aka Stein score), and is approximated by a neuronal network $s_{\theta}(\bzt, t) \approx \nabla_{\bzt} \log p(\bzt)$ using denoising score matching<d-cite key="hyvarinen2005estimation"></d-cite>.

The score function defines a **time-dependent vector field** that guides points toward the data distribution.

<!-- Below we visualize the vector field for a diffusion model trained on the Swissroll dataset. Depending on the time step, the vector field points in different directions.
During the early time steps, the vectors point toward the mean of the data distribution,
whereas near the end of the diffusion process the vector field points directly toward the data manifold.
Observe the red sampling trajectories: they begin with large movements toward the center (the global prior mean) and then settle into refined, smaller steps, locking onto the distinct curves of the Swiss Roll manifold. -->

### Conditional Diffusion Models

Previously, we demonstrated how to create a process for sampling from an unconditional distribution. Extending this to the conditional case, we aim to sample from a posterior $p(\bx \mid \by) \text{,}$ which we can decompose using Bayes' rule:

$$
  \begin{equation}
    p(\bx \mid \by) = \frac{p(\by \mid \bx) p(\bx)}{p(\by)} .
  \end{equation}
$$

Applying the logarithm and differentiating with respect to $\bx$, allows us to define a conditional form of the score function (relying on the fact that the denominator does not depend on $\bx$):

$$
  \begin{equation}
    \textcolor{#A125A1}{\nabla_{\bx} \log \, p(\bx \mid \by)} = \textcolor{#25a18e}{\nabla_{\bx}  \log p(\bx)} + \textcolor{#00a5cf}{\nabla_{\bx} \log p(\mathbf{y} \mid \bx)} .
  \end{equation}
$$

Where the conditional _reverse process_ from Eq. \eqref{eq:reverse_process} is given by:

$$
  \begin{equation}
    \mathrm{d}\bzt = [\bf(\bzt, t) - \fg(t)^2 \; (\textcolor{#25a18e}{\nabla_{\bzt}  \log p(\bz_t)} + \textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)})] \; dt + \fg(t) \; d\mathbf{w} .
  \end{equation}
$$

This formulation allows us to reuse our unconditional model ($\textcolor{#25a18e}{\nabla_{\bzt} \log p(\bz_t)}$) and simply add a guidance term $\textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)}$. The guidance term acts like a force pushing our samples to be consistent with the condition $\by$. The remaining practical challenge is deriving

$$
    \textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)} .
$$

There are two distinct strategies for doing so:

<div style="padding: 10px 10px 10px 10px; border-left: 6px solid #FFD700; margin-bottom: 20px;">
  <p>1. <strong>Classifier Guidance (Learning the Likelihood)</strong>: <br>Learn a time-dependent classifier that approximates the likelihood score through supervised training on labeled data.</p>
  <p style="margin: 0;">2. <strong>Analytical Likelihoods (Defining the Likelihood)</strong>: <br>Leverage analytically tractable likelihood functions, which are particularly valuable for inverse problems, where paired training data $(\bx, \by)$ is scarce or unavailable.</p>
  <!-- <p style="margin: 0;"> <strong>In both approaches, the diffusion model itself remains frozen</strong>, serving only as a fixed prior distribution $p(\bx)$. We are left to define or learn how to assess likelihoods, particularly their gradients, which makes this framework highly data-efficient for adapting to new tasks.</p> -->
</div>

<strong>In both approaches, the diffusion model itself remains frozen</strong>, serving only as a fixed prior distribution $p(\bx)$. We are left to define or learn how to assess likelihoods, particularly their gradients, which makes this framework highly data-efficient for adapting to new tasks.

## Classifier Guidance (CG)

Classifier Guidance<d-cite key="dhariwal2021diffusion"></d-cite> trains a time-dependent neuronal network to approximate the likelihood:

$$
  \begin{equation}
    p_{\phi}(\by \mid \bzt, t) \approx p(\by \mid \bzt, t) .
  \end{equation}
$$

We therefore train a classifier whose inputs are noisy samples that resemble the intermediate steps of the reverse diffusion process. In particular, given a dataset of paired samples $(\bx,\by)$, we can train a time-conditional classifier $p_{\phi}(\by \mid \bzt, t)$ by minimizing

$$
  \begin{equation*}
    \E_{t \sim \uniform(0, T), (\bx, \by) \sim p_{\text{data}}, \boldsymbol{\epsilon} \sim \normal(\mathbf{0}, \mathbf{I})}[-\log p_{\phi}(\by \mid \bzt, t)] ,
  \end{equation*}
$$

with the noisy sample $\bz_t = \alpha_t \bx + \sigma_t \boldsymbol{\epsilon}$.
Handling inputs at varying noise levels allows the classifier to operate across the entire diffusion trajectory.

The trained classifier can be used as an approximation of the <strong><span style="color: var(--color-likelihood);">log-likelihood gradient</span></strong>:

$$
  \begin{equation}
     \textcolor{#00a5cf}{\nabla_{\bzt} \log p(\mathbf{y} \mid \bzt)} \approx \textcolor{#00a5cf}{\nabla_{\bzt} \log p_{\phi}(\by \mid \bzt, t)} ,
  \end{equation}
$$

where the gradient with respect to the input is easy to compute using automatic differentiation frameworks such as PyTorch or JAX.

Finally we combine the prior score with the log likelihood gradient. In practice, classifier guidance works with a scaled version of the guidance term controlled by the parameter $\gamma$,

$$
  \begin{equation}
    \textcolor{#A125A1}{\nabla_{\bz_t} \log \, p_{\gamma}(\bz_t \mid \by)} = \textcolor{#25a18e}{\nabla_{\bz_t} \log p(\bz_t)} + \gamma \textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)} .
  \end{equation}
$$

This scaling lets us adjust the guidance strength: if $\gamma$ is too high, we obtain unrealistic samples that move away from the data manifold, whereas if $\gamma$ is too low, the guidance has little effect.

Before continuing, we want to mention that classifier-free guidance is currently more commonly used, as it often outperforms classifier guidance and does not require training a separate classifier.
However, since this blog post focuses on techniques that modify sampling behavior without retraining the original diffusion model, classifier guidance serves as a better illustrative example.
Additionally, the name classifier-free guidance is somewhat misleading, because a classifier is still involved, it is simply embedded implicitly within the diffusion model itself.
For completeness, we have included classifier-free guidance in the expandable section below.

<details>
  <summary>
     Classifier-Free Guidance (CFG)
  </summary>

<div style="color: black;">
Classifier-free guidance can be derived in a similar manner, the main difference is that the diffusion model itself acts as classifier.

Applying Bayes' rule to

$$
    \nabla_{\bz_t}\,p(\by \mid \bz_t)
$$

results in

$$
  \nabla_{\bz_t}\,p(\by \mid \bz_t)=\nabla_{\bz_t}\,p(\bz_t \mid \by) - \nabla_{\bz_t}\,p(\bz_t).
$$

To avoid training of an unconditional and a conditional model, we train one single model with an additional condition:

$$
  \nabla_{\bz_t}\,p(\by \mid \bz_t)=\nabla_{\mathbf{\bz_t}}\,p(\bz_t \mid \by) - \nabla_{\bz_t}\,p(\bz_t \mid \emptyset),
$$

where we condition the model on the null token $\emptyset$, to represent the unconditional model.

We can now replace the likelihood term in classifier guidance:

$$
  \nabla_{\bz_t} \log \, p_{\gamma}(\bz_t \mid \by) = (1 - \gamma) \nabla_{\bz_t} \log p(\bz_t, \mid \emptyset) + \gamma \nabla_{\mathbf{\bz_t}}\,p(\bz_t \mid \by)
$$

</div>
<!-- $$
  \begin{equation}
    \nabla_{\bx} \, \log p(\mathbf{y} \mid \bz_t) = \textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\bz_t) + \gamma \nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)}
  \end{equation}
$$ -->

</details>

## Analytical Likelihoods

In the last section, we discussed how to train an additional model serving as a likelihood.
For a class of problems where the relationship between the data and the observations follows a known probabilistic model, the likelihood function can be derived analytically.
A canonical example are inverse problems, where we have observations $\by$ and we want to recreate $\bx$. We consider observations of the form:

$$
\begin{equation}
    \mathbf{y} = \mathcal{A}(\bx) + \mathbf{n} \quad \textrm{where} \quad \mathbf{y}, \mathbf{n} \in \mathbb{R}^m, \mathbf{x} \in \mathbb{R}^n.
\end{equation}
$$

Here $$\mathbf{y}$$ is our observation, $$\mathbf{x}$$ is the underlying clean data, $$\mathcal{A}:\mathbb{R}^n \mapsto \mathbb{R}^m$$ is the **forward operator** (which may be linear or nonlinear), and $\mathbf{n}$ is observation noise.

For Gaussian noise $\mathbf{n} \sim \normal(\mathbf{0}, \sigma^2 \mathbf{I})$, the likelihood is defined by:

$$
  \begin{equation}
    \label{eq:y-given-x-gaussian-noise}
    p(\by \mid \bx) = \normal(\by; \mathcal{A}(\bx), \sigma^2 \mathbf{I}) \propto \exp \left(-\frac{1}{2 \sigma^2} ||\by - \mathcal{A}(\bx)||^2 \right) .
  \end{equation}
$$

Typical inverse problems are:

- **Inpainting**, where the operator $\mathcal{A}$ acts as a mask that zeros out out certain pixels, requiring the model to fill in the missing areas.

- **Super-resolution**, where the operator $\mathcal{A}$ performs a downsampling operation and the model’s task is to generate the image at high resolution.

- **Deblurring**, where the operator $\mathcal{A}$ acts as a gaussian filter operation blurring out pixels.

{% include figure.liquid path="assets/img/2026-04-27-diffusion-guidance/cover.png" class="img-fluid" title="Posterior Sampling with Diffusion Models" caption="Figure 1: Source: https://dps2022.github.io/diffusion-posterior-sampling-page" %}

### Diffusion Posterior Sampling (DPS)

Direct computation of the time-dependent likelihood gradient $\textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)}$ poses significant computational challenges. Let's calculate the likelihood by marginalizing over $\bx$:

$$
  \begin{equation}
    p(\by \mid \bz_t) = \int p(\by \mid \bx) \, p(\bx \mid \bz_t) d \bx = \E_{\bx \sim p(\bx \mid \bz_t)}[p(\by \mid \bx)] .
  \end{equation}
$$

The intractability arises from two sources: (1) the marginalization over the high-dimensional space of clean samples $\bx$, and (2) the dependence of $p(\by \mid \bz_t)$ on the reverse diffusion process, which requires integrating over all trajectories from time $t$ to $0$. DPS<d-cite key="chung2022diffusion"></d-cite> addresses this through an approximation that avoids explicit marginalization:

$$
  \begin{equation}
    p(\by \mid \bz_t) \approx p(\by \mid \hat{\bx}(\bz_t)),
  \end{equation}
$$

where $$\hat{\bx}(\bz_t) = \E[\bx \mid \bz_t]$$.
By Tweedie's formula<d-cite key="efron2011tweedie,tweedie1957"></d-cite>, the posterior mean is given by

$$
  \begin{equation}
    \E[\bx \mid \bz_t] = \frac{1}{\alpha_t} (\bzt + \sigma_t^2 \nabla_{\bzt}  \log p(\bz_t)) .
  \end{equation}
$$

Substituting $\nabla_{\bzt}  \log p(\bz_t)$ with the learned score function $s_\theta(\bzt;t)$ gives the estimator:

$$
  \begin{equation}
    \bx_\theta(\bzt;t) = \frac{1}{\alpha_t} (\bzt + \sigma_t^2 s_\theta(\bzt;t)) .
  \end{equation}
$$

Instead of learning the score function, we can train a network to predict the clean data directly, i.e., act as a denoiser<d-cite key="karras2022elucidating"></d-cite>.
There are three main parameterizations: score prediction, data prediction, and noise prediction.
These parameterizations are mathematically equivalent and allow seamless integration of all current diffusion model formulations into this framework. <d-cite key="kingma2023understanding"></d-cite> provides an extensive analysis.

We can now combine the Gaussian likelihood from Eq. \eqref{eq:y-given-x-gaussian-noise} with the network’s data prediction, $$\bx_\theta(\bzt;t)$$, to analytically obtain the log-likelihood gradient:

$$
  \begin{equation}
    \nabla_{\bzt}  \log p(\by  \mid \bz_t) \approx -\frac{1}{\sigma^2} \nabla_{\bzt} ||\by - \mathcal{A}(\bx_\theta(\bzt;t))||^2_2 .
  \end{equation}
$$

Again, we combine the prior and the likelihood term to obtain the score function:

$$
  \begin{equation}
    \textcolor{#A125A1}{\nabla_{\bzt}  \log p(\bx  \mid \bz_t)} \approx \textcolor{#25a18e}{s_\theta(\bzt;t)} - \textcolor{#00a5cf}{\zeta \nabla_{\bzt} ||\by - \mathcal{A}(\bx_\theta(\bzt;t))||^2_2} .
  \end{equation}
$$

### Example

Let's consider a simple example using the Swiss Roll data to demonstrate the effectiveness of diffusion priors in solving inverse problems.
We define a linear forward operator $\mathcal{A}$ that performs a projection of the 2D data onto the $$x_1$$ axis:

$$
  \begin{equation}
    \by = \mathcal{A}(\bx) = \mathbf{A}\bx = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix} \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
  \end{equation}
$$

We sample observations from the ground-truth manifold but restrict them to the left side of the Swiss Roll.
Next, for each observation $$\by$$, we sample from the posterior distribution $$p(\bx \mid \by)$$, which places the samples back on the spiral manifold.
The results are displayed in the figure below, where we have:

1. **Observations (Left)**: This plot shows the observations $p(\by \mid \bx)$. Since the operator projects everything onto the x-axis, the unique spiral structure of the Swiss Roll is entirely gone. This creates an ill-posed inverse problem: for any observed point on this line, there are multiple possible "correct" locations on the original spiral that could have produced the outcome.

2. **Standard Sampling (Right)**: This represents the unconditional generation from our trained diffusion model. While these points perfectly inhabit the Swiss Roll manifold, they are random samples from the prior $p(\bx)$. They show us what the model "knows" about the data distribution, but they have no connection to the specific measurements we observed.

3. **DPS Sampling (Center)**: Here we show the reconstruction using Diffusion Posterior Sampling with $\zeta=0.25$. By using the measurement gradient to guide the generation process, DPS pushes $$\bx$$ to match the observation $$\by$$. It solves the ambiguity by finding points that are both consistent with the measurement $\by$ and highly probable under the learned prior $p(\bx)$, recovering the spiral shape despite information loss through the operator.

<iframe src="{{ 'assets/html/2026-04-27-diffusion-guidance/measurements_dps_standard.html' | relative_url }}" frameborder='0' scrolling='no' height="400px" width="100%"></iframe>

<br>

We note that DPS is only one possible way to estimate the likelihood term $$\textcolor{#00a5cf}{\nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)}$$. For interested readers, we recommend the excellent survey by Daras et al. <d-cite key="daras2024survey"></d-cite>, which compares a range of different approaches.

## Applications in physical sciences

We now turn to scientific applications and examine how flexible the discussed approaches are in practice, particularly in settings where physical constraints must be respected. The following is just an high level overview.

### DiffusionPDE

DiffusionPDE<d-cite key="huang2024diffusionpde"></d-cite> use DPS to sample Partial Differential Equations (PDEs) solutions from sparse observation, by exploiting the structure of the underlying PDE, a guidance term can be derived to align the sampling to the underlying structure of the PDE.

We take the **Darcy flow** PDE equation as an example:

$$
\begin{aligned}
-\nabla \cdot \big(a(\bc)\nabla u(\bc)\big) &= q(\bc), \quad \bc \in \Omega, \\
u(\bc) &= 0, \quad \bc \in \partial\Omega.
\end{aligned}
$$

Defining the residual as an operator

$$
\begin{aligned}
f(\bc) = \nabla \cdot \big(a(\bc)\nabla u(\bc)\big) + q(\bc),
\end{aligned}
$$

so that valid solutions satisfy $f(\bc) = 0 $.
This residual is used to construct an additional physics-based guidance loss:

$$
\begin{aligned}
\mathcal{L}_{\textrm{pde}}
= ||\mathbf{0} - f(\hat{\bx}(\bz_t)) ||^2_2 .
\end{aligned}
$$

DiffusionPDE then augments the conditional score function as

$$
  \begin{equation}
    \textcolor{#A125A1}{\nabla_{\bz_t} \log p(\bz_t \mid \by, f)}
    \approx
    \textcolor{#25a18e}{s_\theta(\bzt;t)}
    \textcolor{#00a5cf}{+  \zeta \nabla_{\bz_t} \log p(\mathbf{y} \mid \bz_t)
    - \zeta_{\text{pde}} \nabla_{\bz_t} \mathcal{L}_{\text{pde}}}.
  \end{equation}
$$

The sampling process is very similar to the one we saw before, combining the part we saw in DPS sampling and the additional term for the PDE solution.

### Inequality constraints for rare event sampling

Rare events play a central role in many scientific settings, yet they are often difficult to sample. This is evident in weather prediction, where extreme events such as floods are of particular concern.

The work by Finzi et al. <d-cite key="finzi2023user"></d-cite> showed that operators can also be defined via inequality constraints.
For a one-dimensional inequality constraint $$\mathcal{A}(\bx) > y$$, we want to sample from $$p(\mathcal{A}(\bx) > y \mid \bz_t)$$.

This inequality constraint is defined by a Gaussian CDF function $$\Phi$$:

$$
  \begin{equation}
    p(\mathcal{A}(\bx) > y \mid \bz_t) \approx \Phi \left( \frac{\mathcal{A}(\bx_\theta(\bzt;t)) > y}{\sqrt{\nabla \mathcal{A}(\bx_\theta(\bzt;t))^T \hat{\Sigma}(\bz_t) \nabla \mathcal{A}(\bx_\theta(\bzt;t)) }} \right) ,
  \end{equation}
$$

with the covariance of $$\bx$$ given $$\bz_t$$

$$
  \hat{\Sigma}(\bz_t)=\frac{\sigma_t^2}{\alpha_t^2} (\bI + \sigma_t^2 \nabla_{\bzt}^2 \log p(\bzt) )
$$

and

$$
  \nabla \mathcal{A}(\bx_\theta(\bzt;t))= \left. \nabla \mathcal{A}(\bx) \right|_{\bx = \bx_\theta(\bzt;t)} .
$$

Using the Gaussian CDF function assigns high probability to events with $$\mathcal{A}(\bx) > y$$.

We can then sample from these event by using

$$
  \begin{equation}
    \textcolor{#A125A1}{\nabla_{\bz_t} \log \, p(\bz_t \mid \mathcal{A}(\bx) > y)} = \textcolor{#25a18e}{\nabla_{\bz_t} \log p(\bz_t)} + \textcolor{#00a5cf}{\zeta \nabla_{\bz_t} \log p(\mathcal{A}(\bx) > y \mid \bz_t)} .
  \end{equation}
$$

The work shows how to effectively sample extreme events in a Fitzhugh-Nagumo system, modeling neuron spiking events occurring only in 1/30 of the trajectories.

## Closing takeaways

To sum it up, guidance enables us to adapt the sampling process of diffusion models by modifying the direction of the underlying vector field. While Classifier Guidance and Classifier-Free Guidance are well known tools, guidance based on analytical likelihoods is still less widely known, especially in scientific applications.

- **Analytical Likelihoods:**
  Enable pre-trained diffusion models to be reused as flexible priors across many downstream tasks, without retraining.
- **Applications in Physical Sciences:**
  By defining forward operators for PDE residuals, or inequality constraints, guidance can steer the sampler toward solutions that are not only data-consistent but also physically more plausible.

The approaches we discussed, represent just a small part of the full landscape and we see that there is currently growing interest in new guidance strategies. A particularly exciting path is the idea of learning strong universal priors in the physics domain and using them across various downstream tasks.
