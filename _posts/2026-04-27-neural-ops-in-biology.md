---
layout: distill
title: "Beyond Black-Box Predictions: Neural Operators as a Bridge to Interpretable Governing Equations in Biology"
description: Your blog post's abstract.
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
    url: "https://en.wikipedia.org/wiki/Albert_Einstein"
    affiliations:
      name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-neural-ops-in-biology.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: 1. Introduction
  - name: "2. Background: Neural Operators & Sparse Identification"
    subsections:
      - name: 2.1 Neural Operators
      - name: 2.2 Sparse Identification (SINDy)
  - name: 3. Biological Motivation & Challenges
  - name: "4. Case Study: Simplified NF-κB Signaling Model"
  - name: 5. Methods
  - name: 6. Results & Interpretation
  - name: 7. Key Contributions & Insights
  - name: 8. Discussion & Limitations
  - name: 9. Conclusion & Outlook
  - name: References

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

## 1. Introduction

Biological systems such as gene regulatory networks, signaling cascades, and cellular feedback loops are often modeled as nonlinear dynamical systems: coupled ODEs or PDEs with feedback, cooperativity, and saturation <d-cite key="brunton2016discovering, dunn2022identification"></d-cite>. In principle, learning such equations directly from data would give compact, mechanistic models that extrapolate beyond the training set and suggest new experiments—a much stronger outcome than pure black-box prediction.

In practice, however, biological data sit in an awkward regime for most existing tools. Experiments usually observe only a few readouts (for example, one or two fluorescent reporters) while many internal species remain hidden, measurements are noisy and sparsely sampled in time, and perturbation budgets are limited <d-cite key="hoffmann2002ikb, dunn2022identification"></d-cite>. Classical parameter inference assumes a correct mechanistic form and can become ill-posed in this setting, while generic deep networks, neural ODEs, and PINNs tend to learn accurate but opaque predictors that entangle biology-specific structure with generic function approximation, making it hard to extract interpretable “governing equations.”

Two strands of recent work point toward an interesting different perspective:

- **Neural Operators (NOs)** learn maps between *function spaces*, such as initial conditions or input signals to full solution trajectories, with architectures that are approximately mesh / resolution independent <d-cite key="kovachki2023neural, li2020fourier"></d-cite>.
- **Sparse Identification of Nonlinear Dynamics (SINDy)** discovers parsimonious governing equations from time-series data by selecting a small number of terms from a library of candidate nonlinear functions <d-cite key="desilva2020pysindy, brunton2016sindy, champion2022ensemble"></d-cite>.

This post investigates how these two ideas can be combined to move “beyond black-box predictions” in biology. The case study is a five-dimensional ODE model inspired by NF-κB signaling, in which only three species are treated as observed; a Fourier Neural Operator (FNO) is trained to reconstruct full trajectories, including hidden variables, from roughly twenty measurements per trajectory, after which SINDy is applied to dense trajectories from either the ground-truth simulator or the learned operator <d-cite key="hoffmann2002ikb"></d-cite>. Empirically, the FNO accurately interpolates unobserved species with Hill-type nonlinearities and supports arbitrary temporal resampling, while SINDy finds only approximate governing equations and performs similarly on simulator- and FNO-generated data—suggesting that the bottleneck lies in the complexity of biological dynamics and library design, not in operator learning itself.

---

## 2. Background: Neural Operators & Sparse Identification

### 2.1 Neural Operators

A neural operator is designed to approximate a mapping between function spaces, for example
$$
\mathcal{G}: u(\cdot) \mapsto v(\cdot),
$$
where both the input $u$ and output $v$ are functions (such as fields over space and/or time), rather than finite-dimensional vectors <d-cite key="kovachki2023neural, li2021fourier"></d-cite>.

Architectures such as the Fourier Neural Operator (FNO) parameterize $\mathcal{G}$ by repeatedly lifting the input into a higher-dimensional channel space, applying global convolutions in a spectral basis (e.g. via FFT), and then projecting back, yielding a model that can be evaluated on different meshes than it was trained on. This mesh-independence is crucial in scientific computing and makes neural operators particularly attractive for biological applications where temporal or spatial sampling can vary across experiments.

Key properties relevant here are:

- **Resolution flexibility:** once trained, an FNO can be evaluated on arbitrarily dense time grids (and space grids in PDE settings) without retraining.
- **Family-level generalization:** if trained over a range of initial conditions or parameters, the neural operator represents a *family* of dynamical responses, not a single trajectory.
- **Compatibility with partial observability:** encoders and decoders can map between partial observations (e.g. a subset of species) and full state trajectories, enabling hidden-state reconstruction.

These features differentiate NOs from standard MLPs, RNNs, or neural ODEs, which typically operate on fixed-size vectors and are tied to specific discretizations or sampling schemes <d-cite key="brunton2016discovering"></d-cite>.


### 2.2 Sparse Identification (SINDy)

Sparse Identification of Nonlinear Dynamics (SINDy) starts from time-series data $x(t)$ and seeks a sparse representation of the vector field $f$ in the ODE
$$
\frac{dx}{dt} = f(x)
$$
using a library of candidate basis functions $\Theta(x)$, such as polynomials, trigonometric terms, or domain-specific nonlinearities. The method writes as,
$$
\frac{dx}{dt} \approx \Theta(x)\,\Xi,
$$
where $\Xi$ is a sparse matrix of coefficients, and solves a sequence of regularized least-squares problems to find a sparse $\Xi$ that still explains the observed dynamics.

In practice, SINDy works best when:

- Time derivatives can be estimated accurately (requiring dense and relatively low-noise data).
- The true dynamics can be represented by a relatively small number of basis functions present in $\Theta$.
- The measurement space is not too high-dimensional and does not mix many hidden processes.

Extensions such as SINDy-PI for implicit dynamics, Ensemble-SINDy for robustness to low-data and noise, and SINDy-CRN / Reactive SINDy for biochemical reaction networks broaden the method’s applicability and robustness.

Biological systems pose challenges for SINDy: hidden species, cooperative Hill nonlinearities with unknown exponents, and stiff binding/unbinding kinetics can all make sparse recovery difficult, especially from realistically noisy, sparse data.

---

## 3. Biological Motivation & Challenges

Many biological systems are naturally described as dynamical systems: gene regulatory networks, signaling pathways, and cell–cell communication can often be written as coupled ODEs or PDEs with nonlinear regulatory terms (for example, Hill functions, saturating kinetics, or cooperative binding). In principle, methods like FNOs and SINDy are appealing because they promise two complementary capabilities: learning how trajectories evolve from data (operator learning) and extracting why they evolve that way in the form of interpretable governing equations (sparse identification).

However, several aspects of biological data make direct application of existing tools difficult. First, experiments usually provide only partial observability: one or a few fluorescent reporters or bulk readouts, while many internal species (mRNA, complexes, post-translationally modified proteins) remain hidden but dynamically important. Second, measurements are sparse in time and noisy, because high-frequency imaging or repeated sampling is expensive or damaging, and individual cells exhibit strong variability. Third, nonlinearities such as cooperative Hill regulation, complex formation, and feedback loops introduce stiffness and multi-timescale behavior that can be hard for both classical parameter inference and generic deep networks to fit robustly <d-cite key="hoffmann2002ikb, champion2019data"></d-cite>.

Classical parameter inference approaches, which fit a mechanistic ODE or PDE model to data, require committing to a specific functional form and often struggle when many parameters are unidentifiable from limited, noisy measurements <d-cite key="dunn2022identification"></d-cite>. Generic deep learning models, including standard sequence models or vanilla neural ODEs, can interpolate trajectories but typically do not generalize across conditions in a controlled way and offer little interpretability about underlying mechanisms. SINDy provides interpretability by selecting sparse libraries of candidate terms, but it assumes access to reasonably dense, relatively low-noise trajectories in a space where the “right” basis functions are already present—conditions rarely met in biology <d-cite key="desilva2020pysindy, kaheman2020sindypi"></d-cite>.

This motivates a hybrid approach in which (i) a neural operator learns to reconstruct full, dense trajectories—including hidden species—from limited observations, and (ii) SINDy operates on these richer trajectories to test how far sparse equation discovery can go in a more realistic biological regime. The NF-κB signaling case study below is chosen as a concrete, yet tractable, arena to explore this idea.

---

## 4. Case Study: Simplified NF-κB Signaling Model

NF-κB signaling is a canonical nonlinear feedback system in immunology and systems biology, involving cytoplasmic–nuclear shuttling of NF-κB, synthesis and degradation of its inhibitor IκB, and negative feedback that produces pulsed or oscillatory nuclear localization dynamics <d-cite key="hoffmann2002ikb"></d-cite>. Many established NF-κB models include Hill-type transcriptional activation, formation of NF-κB:IκB complexes, and nonlinear degradation, making them representative of the kinetic motifs seen across signaling and gene regulation.

To explore the use of neural operators and SINDy in a controlled setting, this work uses a five-variable ODE model inspired by classic NF-κB signaling modules. The state variables are:

- $N$: cytoplasmic NF-κB,
- $N_n$: nuclear NF-κB,
- $I$: IκB protein,
- $M$: IκB mRNA,
- $C$: cytoplasmic NF-κB:IκB complex.

In the *synthetic experiment*, only three of these are treated as “observable”:

- $N$, $N_n$, and $I$ are taken as observable, mimicking fluorescent readouts of NF-κB localization and IκB abundance.
- $M$ and $C$ are treated as *hidden*.

The governing equations are of the form    
$$
\begin{aligned}
\dot N &= -k_{\mathrm{imp}}\,N + k_{\mathrm{exp}}\,N_n - k_{\mathrm{bind}}\,I\,N + k_{\mathrm{diss}}\,C,\\
\dot N_n &= k_{\mathrm{imp}}\,N - k_{\mathrm{exp}}\,N_n,\\
\dot I &= \alpha_I\,M - \delta_I\,I + k_{\mathrm{diss}}\,C,\\
\dot M &= \alpha_m\,\frac{N_n^n}{K_m^n + N_n^n} - \delta_m\,M,\\
\dot C &= k_{\mathrm{bind}}\,I\,N - k_{\mathrm{diss}}\,C - k_{\mathrm{degC}}\,C.
\end{aligned}
$$

Here:

- $k_{\mathrm{imp}}$, $k_{\mathrm{exp}}$ govern NF-κB nuclear import/export.
- $k_{\mathrm{bind}}$, $k_{\mathrm{diss}}$, $k_{\mathrm{degC}}$ govern complex formation and degradation.
- $\alpha_m$, $\delta_m$, $\alpha_I$, $\delta_I$ govern transcription/translation and decay.
- $n$ and $K_m$ define a Hill-type transcriptional activation of $M$ by nuclear NF-κB.

Appropriate parameter choices yield rich transient and oscillatory-like response dynamics under variations in initial conditions and parameter jitter, consistent with simplified NF-κB models in the literature. This model is deliberately chosen because it concentrates several key challenges for data-driven discovery in biology within a small ODE system: (i) only a subset of variables is measured, (ii) one equation contains a nonlinear Hill function with exponent $n>1$, and (iii) feedback and binding introduce multi-timescale dynamics. At the same time, it is simple enough to simulate efficiently and to serve as a controlled benchmark for comparing dense trajectories from the ground-truth ODE to trajectories reconstructed by a learned FNO, before passing both into SINDy. This makes it a natural testbed for assessing how well neural operators can interpolate hidden biological state and how robust sparse equation discovery remains under realistic biological constraints.

---

## 5. Methods

### 5.1 Synthetic data generation

To emulate realistic biological experiments, an ensemble of trajectories is generated by numerically integrating the NF-κB ODE system under multiple initial conditions and small parameter variation.

- **Initial conditions:** about 40 random initial states for $(N, N_n, I, M, C)$, sampled around a baseline equilibrium.
- **Parameter jitter:** multiplicative perturbations of key parameters (e.g. $\pm 10\%-15\%$) to mimic cell-to-cell variability.
- **Sampling:** each trajectory is sampled at a modest temporal resolution (e.g. 20 measurement times over the response window), reflecting realistic imaging frequencies.
- **Noise:** additive noise is applied to the observable channels $(N, N_n, I)$ to approximate measurement noise.

The resulting dataset consists of partial, noisy time series for $(N, N_n, I)$ and corresponding full-state trajectories from the simulator (used only as “ground truth” for evaluation, not exposed to the learner in the FNO training objective).

> **Figure placeholder:** Insert a schematic or example of NF-κB trajectories for the observable and hidden species (e.g. nuclear translocation pulse, delayed mRNA and IκB response).

### 5.2 FNO for full-state reconstruction

A one-dimensional Fourier Neural Operator is trained to map sparse trajectories of the observable species to full trajectories of all five state variables. Concretely:

- **Input to the FNO:** a time series of shape (time, channels) for the three observed variables $(N, N_n, I)$ on a coarse temporal grid.
- **Output:** a time series for all five variables $(N, N_n, I, M, C)$ on the same grid.
- **Architecture:** a compact FNO implementation in PyTorch with a few spectral convolution layers, following the general pattern of Li et al. (2020) and Kovachki et al. (2023) <d-cite key="kovachki2023neural, li2021fourier"></d-cite>.
- **Loss:** mean squared error between predicted and true trajectories for all five variables, with an emphasis on the observed variables during early training epochs.

Because the FNO learns an operator from *functions of time* (partial observations) to *functions of time* (full-state trajectories), it can naturally be evaluated at any temporal resolution once trained, simply by providing the input on its original grid and requesting output on a finer time grid.

Training uses roughly 20 measurement points per trajectory, demonstrating that the operator can reconstruct hidden and observed species from quite sparse observations.

> **Figure placeholder:** Insert a small architecture diagram of the FNO mapping observed channels to full state channels.

### 5.3 Generating dense trajectories: ODE vs FNO

To support SINDy, two sources of “dense” trajectories are constructed:

1. **Dense ODE ground truth:** the NF-κB ODE system is integrated with a fine step size, and the full state is saved at a dense time grid (e.g. hundreds to thousands of timepoints). This represents an idealized, noise-free, fully observed dataset that is not attainable experimentally but serves as a benchmark for SINDy.
2. **Dense FNO reconstructions:** the trained FNO takes the original sparse, noisy observation trajectories for $(N, N_n, I)$ and outputs full-state trajectories on a *dense* time grid, expanded from the original 20 measurement points to a much finer mesh.

The ability to *expand the mesh* in this way — leveraging operator learning rather than pointwise interpolation — is a key practical advantage of FNOs over traditional feedforward networks trained to map fixed-length input vectors to fixed-length output vectors, which are meant to mimic what a real biological experiment might produce.

To query the FNO on a finer time grid, these sparse observations are first linearly interpolated in time, creating a higher-resolution input signal for the operator. The key point is that this interpolation is just a way of providing the operator with values on the new grid; it is not the mechanism that creates meaningful dynamics between measurement times.

The important dynamical structure actually comes from the trained FNO itself. Once trained, the FNO represents an operator that maps any reasonable time series of observables (including their interpolated versions) to a full-state trajectory consistent with what it learned during training.[@kovachki2023neural; @li2021fourier] When fed linearly interpolated inputs on a dense grid, the FNO effectively smooths and regularizes the observable channels and, at the same time, predicts the hidden species at those new timepoints according to the learned dynamics. In other words, the operator “fills in” the trajectory in a way that is constrained by the NF-κB-like ODE behavior it has internalized, rather than simply connecting the dots between measurements.

Without a trained neural operator (or an equivalent dynamical model), linear interpolation alone would only give straight-line segments between data points for the observables and no information at all about the unobserved variables. By contrast, in this pipeline linear interpolation is merely a numerical convenience to sample the learned operator at arbitrary times: the FNO uses those interpolated observables as a conditioning signal and then generates a dynamically coherent, multi-species trajectory on the dense mesh. This is precisely the kind of densified, multi-variable, low-noise dataset that SINDy and related sparse model discovery methods are designed to work with, but that cannot be obtained from raw experimental traces alone.


### 5.4 SINDy setup

SINDy is then applied to both dense datasets to attempt recovery of the NF-κB equations:

- **State variables:** the full 5D state $(N, N_n, I, M, C)$ is used for SINDy in both cases, with numerical derivatives estimated by smoothed finite differences or regularized differentiation operators.
- **Library:** a biology-aware library is constructed, including:
  - polynomials up to degree 2 in each variable,
  - selected interaction terms (e.g. $I N$),
  - a fixed Hill-type nonlinearity in $N_n$ to capture the $M$ production term.
- **Regression:** sparse regression with sequential thresholding is used to identify nonzero coefficients in each row of $\Xi$, following the PySINDy defaults.

Two SINDy experiments are run:

1. **Ideal SINDy:** on dense ODE ground truth.
2. **FNO-SINDy:** on dense trajectories generated by the FNO from sparse observations.

Comparing these two settings isolates the contribution of operator reconstruction from the inherent difficulty of the SINDy problem.

---

## 6. Results & Interpretation

### 6.1 FNO reconstruction quality

The trained FNO achieves low mean squared error on the observed channels and reconstructs the hidden variables $M$ and $C$ with qualitatively correct dynamics across test trajectories. In particular, it captures:

- Delayed transcriptional pulses in $M$ following nuclear NF-κB translocation.
- Gradual accumulation and decay of IκB protein $I$.
- Transient formation and degradation of the NF-κB:IκB complex $C$.

These behaviors are recovered despite training on only ~20 measurement points per trajectory and with additive noise on the observables, highlighting the sample efficiency and denoising properties of neural operators in this setting.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-neural-ops-in-biology/fno-fit-1.png" class="img-fluid"  %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-neural-ops-in-biology/fno-fit-2.png" class="img-fluid" %}
    </div>
    <div class="caption">
      Two sample predictions from Fourier Neural Operator model fit (in red) on sparse experimental (synthetic) data with noise and jitter added (in purple). The (grey) line shows the ideal trajectory based on numerically simulating the 5-ODE system.
  </div>
</div>

{% include figure.liquid path="assets/img/2026-04-27-neural-ops-in-biology/fno-perf-table.png" class="img-fluid" %}
<div class="caption">
      Quantitative performance measure of the FNO model.
  </div>

### 6.2 SINDy on dense ODE vs dense FNO data

On the dense ODE ground truth:

- SINDy recovers many of the qualitative structures in the NF-κB model:
  - linear import/export terms for $N$ and $N_n$,
  - production and degradation terms for $I$ and $M$,
  - a Hill-type nonlinear term in the $M$ equation.
- Quantitatively, recovered coefficients can deviate from true parameters by 20–30% or more, depending on thresholding and library choices, and some spurious low-magnitude terms may appear.

On the dense FNO reconstructions:

- SINDy’s performance is **broadly similar**:
  - The same qualitative term structure is often recovered.
  - Coefficients differ somewhat and may reflect reconstruction biases from the FNO.
  - Occasional extra terms appear, again sensitive to thresholds and noise.

Crucially, SINDy does **not** degrade dramatically when using FNO-generated trajectories versus ideal ODE trajectories: the symbolic recovery is imperfect in both cases, and the gap between them is modest. This suggests that, in this example, the limiting factor is the intrinsic complexity of the NF-κB dynamics and the expressiveness of the chosen library, rather than the neural operator.

> **Figure 2 placeholder:** Side-by-side comparison of discovered equations (symbolic forms and coefficients) for key variables using (a) dense ODE ground truth and (b) dense FNO reconstructions.

> **Figure 3 placeholder:** Simulations of the SINDy-inferred ODE systems vs the true ODE on held-out initial conditions.

Overall, these experiments support the view that neural operators can serve as high-quality surrogates and data densifiers for biological systems, even if full symbolic recovery remains challenging.

---

## 7. Key Contributions & Insights

Summarizing the main insights from this case study:

- **Hidden-state interpolation from sparse observations:** The FNO successfully reconstructs unobserved species with complex Hill-type dynamics from limited measurements of three observable species, illustrating how operator learning can bridge gaps in biological observability.
- **Mesh expansion and dense resampling:** Once trained, the FNO can generate dense trajectories at arbitrary temporal resolution, providing the kind of data SINDy needs but is rarely available experimentally, without retraining a separate model for each resolution.
- **Comparable SINDy performance on ODE vs FNO data:** The similar quality of SINDy’s output on dense ODE and dense FNO trajectories indicates that neural operators can be used as front-ends for symbolic discovery without fundamentally limiting performance in this setting.
- **Biology-aware modeling:** The experiment explicitly targets features common in biological systems — Hill kinetics, complex formation, feedback — and shows that FNOs handle these dynamics robustly, complementing existing uses of neural operators for more traditional physical PDE systems.
- **Extensibility to PDE models:** While this case study focuses on ODEs, the same methodology applies directly to PDE-based biological models (e.g. NF-κB or morphogen gradients in tissues, reaction–diffusion systems), where neural operators have already been used as surrogates for complex PDE solvers <d-cite key="kovachki2023neural"></d-cite>.

---

## 8. Discussion & Limitations

From a method perspective, this work suggests that neural operators are particularly well-suited to the realities of biological data:

- They naturally operate on functions (time series, spatial fields) rather than fixed-length vectors.
- They can integrate information across time and conditions to infer hidden variables.
- They enable re-sampling at arbitrary time grids, helpful for downstream tasks like SINDy.

Compared to neural ODEs or PINNs, which typically require explicit parameterization of the underlying ODE/PDE and strong supervision, neural operators can be trained more directly on input–output pairs of functions, making them attractive in settings where the mechanistic equations are only partially known or too complex to write down.

However, several important limitations remain:

- **Symbolic recovery is hard:** Even with ideal dense ODE data, SINDy only partially recovers the NF-κB equations. Cooperative Hill nonlinearities, stiff binding dynamics, and partial observability push the method beyond its most comfortable regime.
- **Library design:** Including Hill functions with unknown exponents, saturating terms, and domain-informed motifs in the SINDy library is nontrivial. Learning or adapting the library, potentially guided by the neural operator’s local response, remains an open problem.
- **Data richness:** The NF-κB case study uses a modest set of initial conditions and parameter jitters. Larger, multi-perturbation datasets (e.g. varying stimuli, genetic backgrounds) would likely be needed to more fully identify nonlinear regulatory structure.
- **Model mismatch:** The synthetic ODE model, while biologically inspired, is still a simplification. In real systems, unmodeled processes, delays, and stochasticity further complicate both operator learning and symbolic discovery.

These limitations suggest that the current results should be viewed as a *step in the right direction* rather than a complete solution: neural operators make it feasible to reconstruct hidden states and densify data, but full equation discovery in realistic biological systems remains a challenging research frontier.

---

## 9. Conclusion & Outlook

This blog post presented a proof-of-concept integration of neural operators and sparse identification for a biologically motivated NF-κB signaling model. The main conclusion is that neural operators can act as powerful, data-efficient surrogates that reconstruct hidden species and provide dense, noise-reduced trajectories from limited measurements, thereby enabling symbolic methods like SINDy to be applied where they otherwise could not.

While the SINDy results are not yet “textbook perfect,” their similar performance on dense ODE and FNO-generated data indicates that neural operators do not fundamentally hinder symbolic discovery in this setting. Instead, the difficulty lies in the complexity of biological kinetics and the need for richer, more biologically structured libraries and datasets.

Looking forward, this operator-plus-symbolic paradigm is promising for:

- PDE-based biological models (morphogen gradients, tissue-scale NF-κB waves, reaction–diffusion patterning),
- single-cell dynamical systems with partial readouts (e.g. live-cell imaging with a few fluorescent reporters),
- spatial omics dynamics (e.g. coarse-grained spatial transcriptomics over time),
- and synthetic biology control systems where interpretable equations remain key for design and analysis.

Neural operators such as the FNO provide a flexible bridge between sparse biological measurement and dense, mechanistically meaningful dynamical models. Combining them with advances in sparse model discovery may eventually yield a practical workflow for “learning the equations of life” from real experimental data.
