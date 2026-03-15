---
layout: distill
title: Discretisation invariance
description:
  Discretisation invariance, a recent innovation in scientific machine learning, is a requirement that ensures an architecture can consistently process inputs of different resolutions. In this post, we formally define this property, provide examples, generate datasets, train architectures, and discuss whether discretisation invariance delivers its intended benefits in practice.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

authors:
- name: Vladimir Fanaskov
  affiliations:
    name: AXXX
- name:  Ivan Oseledets
  affiliations:
    name: AXXX

bibliography: 2026-04-27-discretisation-invariance.bib

toc:
  - name: Introduction and motivation
  - name: What is discretisation invariance?
  - name: Examples and non-examples
    subsections:
      - name: Fourier Neural Operator
      - name: Convolutional neural networks
      - name: Deep Operator Network with linear observations
      - name: PCA-Net
      - name: Neural fields
      - name: Graph neural networks
      - name: Graph kernel networks
  - name: Training and evaluation
  - name: The role of discretisation invariance
    subsections:
      - name: Arguments against discretisation invariance
      - name: Arguments in favour of discretisation invariance
  - name: Discussion

---
## Introduction and motivation
Partial differential equations (PDEs) are the gold standard in scientific modelling. With rare exceptions, PDEs are solved numerically, and the goal is always to develop a reliable, accurate algorithm that delivers a solution as cheaply as possible. Neural networks were first explored as a solution method starting at least from the 1990s <d-cite key="lee1990neural"></d-cite>, <d-cite key="lagaris1998artificial"></d-cite>. Over time, these methods converged to two dominant paradigms: physics-informed neural networks (PINNs, <d-cite key="lagaris1998artificial"></d-cite>, <d-cite key="raissi2017physics"></d-cite>) and operator learning <d-cite key="hesthaven2018non"></d-cite>, <d-cite key="kovachki2023neural"></d-cite>, <d-cite key="lu2021learning"></d-cite>. PINN is an unsupervised technique that directly aims to solve PDEs. Operator learning, the approach we consider here, is a supervised technique aiming to amortize the cost of parametric PDE solutions in the "multi-query setting". We explain the setup of operator learning below.

We start by specifying a model of interest in the form of a PDE. For the sake of example, we consider the stationary diffusion equation

$$
\begin{aligned}
&\frac{\partial}{\partial \boldsymbol{x}_{1}}\left(k(\boldsymbol{x})\frac{\partial}{\partial \boldsymbol{x}_{1}} u(\boldsymbol{x})\right)+\frac{\partial}{\partial \boldsymbol{x}_{2}}\left(k(\boldsymbol{x})\frac{\partial}{\partial \boldsymbol{x}_{2}} u(\boldsymbol{x})\right) = f(\boldsymbol{x}),\\
&\boldsymbol{x}\in\Gamma=(0, 1)^2,\,\left.u(\boldsymbol{x})\right|_{\boldsymbol{x}\in\partial\Gamma} = 0.
\end{aligned}
$$

This PDE naturally appears in modelling of multiphase fluid flow, heat conduction and electrostatic problems in heterogeneous media.

The PDE contains two parameters: diffusivity coefficient $k(\boldsymbol{x}) > 0$ and the source term $f(\boldsymbol{x})$. We assume that we need to solve the stationary diffusion equation repeatedly for a large set of parameters drawn from a joint probability distribution $k(\boldsymbol{x}), f(\boldsymbol{x}) \sim p_{f, k}$. One may simply call a classical solver for each new pair of parameters, but it can be more advantageous to exploit information recovered from already obtained solutions.

This can be done in a standard regression framework: collect a dataset $\left(f_1, k_1, u_1\right), \dots, \left(f_{M}, k_{M}, u_{M}\right)$, select a parametric model $\mathcal{N}_{\theta}$ and train it with $L_2$ loss function

$$
\begin{aligned}
\theta^{\star} &= \arg\min_{\theta}\mathbb{E}_{k, f\sim p_{k, f}}\left\|\mathcal{N}_{\theta}(k, f) - u\right\|_2^2 \\
&\simeq \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\left\|\mathcal{N}_{\theta}(k_i, f_i) - u_i\right\|_2^2.
\end{aligned}
$$

All standard techniques of machine learning apply: cross-validation, gradient descent methods, regularisation, etc.

When a PDE is discretised, the continuous physical fields $k(\boldsymbol{x}), f(\boldsymbol{x}), u(\boldsymbol{x})$ become matrices, e.g., $k\_{ij} = k(\boldsymbol{x}\_{ij})$ where $\boldsymbol{x}\_{ij} = \left((\boldsymbol{x}\_{1})_{i}, (\boldsymbol{x}\_{2})\_{j}\right)$ is a point on a regular grid. In this case, the regression problem is conceptually similar to classical image-to-image tasks: segmentation, denoising, and super-resolution.

Recently, a group of researchers suggested that the regression problem for PDEs involves more than learning an image-to-image map <d-cite key="li2020neural"></d-cite>, <d-cite key="li2020fourier"></d-cite>. They argued that the primal objects are functions themselves, not merely the particular way they are summarised with finite data. For example, one may represent the function $k(\boldsymbol{x})$ on a grid with $N\times N$ points, or on the grid with $2N\times 2N$ points, or as a set of coefficients $c_{i}$ in a finite series $k(\boldsymbol{x}) = \sum_{i, j=1}^{N} c_{i}\phi_{i}(\boldsymbol{x})$. While the particular representations are different, the underlying function $k(\boldsymbol{x})$ remains the same in all cases. This raises a key question: Is it possible to build a neural network that is, to a degree, agnostic to the choice of particular discretisation? The answer is positive, and the architectures exhibiting this desirable property are now called *discretisation invariant* or *discretisation agnostic*.

In this note, we address several questions about discretisation invariance architectures:
1. What is discretisation invariance? How to define it formally?
2. How are discretisation-invariant architectures built?
3. Why is discretisation invariance important?

## What is discretisation invariance?
Intuitively, discretisation-invariant architectures consistently map functions for different resolutions: when more details appear in the input, we expect to see more details in the output.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/waves.png" class="img-fluid" %}
<div class="caption">
    An example of a discretisation-invariant map $\psi = \mathcal{F}(\phi)$. When the input $\phi$ is available on a refined grid, the output $\psi$ is refined too.
</div>

To slightly formalise the illustration above, we define sampling and interpolation operators.

The sampling operator $\mathcal{S}\_{N}:\mathcal{C}\_{[0, 1]}\rightarrow \mathbb{R}^{N}$ takes a function $f$ from the space of continuous functions $\mathcal{C}\_{[0, 1]}$, and outputs its values on the uniform grid $x_{i} = i / (N-1),\,i=0,\dots,(N-1)$.

Interpolation operator $\mathcal{I}\_{N}:\mathbb{R}^{N}\rightarrow \mathcal{C}\_{[0, 1]}$ performs the inverse operation: from a set of samples $f(x\_i)$ it reconstructs the function $\widetilde{f}$ by linear interpolation

$$
\widetilde{f}(x) = \frac{f(x_{i}) (x_{i+1} - x) + f(x_{i+1}) (x - x_{i})}{x_{i+1} - x_{i}},
$$

where $x \in[x\_{i}, x\_{i+1}]$.

In general, $f(x) \neq \mathcal{I}\_{N}\left(\mathcal{S}\_{N}(f)\right)(x)$ but as $N$ grows, the composition $\mathcal{I}\_{N}\mathcal{S}_{N}$ becomes closer to identity in the standard $L_2$ norm: $\lim\limits\_{k\rightarrow\infty}\left\|f - \mathcal{I}\_{k}\mathcal{S}\_{k} f\right\|\_2 = 0$.

Having sampling $\mathcal{S}$ and interpolation $\mathcal{I}$ operators, we call a map $F\_{k}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{k}$ discretisation invariant if the sequence $\psi_k=\mathcal{I}\_{k}\left(F_{k}\left(\mathcal{S}\_{k}(\phi)\right)\right)$ converges to a unique element of the space $\mathcal{C}\_{[0, 1]}$ for each input $\phi$.

Our definition requires several clarifications:
1. We choose particular operators $\mathcal{S}\_{k}$, $\mathcal{I}\_{k}$ for the sake of example. In general, we ask for $\mathcal{S}\_{k}$ to extract a finite amount of information from function, $\mathcal{I}\_{k}$ to approximately restore the original function from this information, and for $\mathcal{I}\_{k} \mathcal{S}\_{k}$ to converge to the identity map.
2. The operator $\mathcal{S}\_{k}$ is analogous to an encoder and $\mathcal{I}\_{k}$ is analogous to a decoder. Unlike encoders and decoders, $\mathcal{S}\_{k}, \mathcal{I}\_{k}$ are not learned.
3. In current literature, $\mathcal{S}\_{k}$ is always a sampling operator. Given that, discretisation-invariant architectures are mainly architectures agnostic to the resolution of the input.
4. We select the $\mathcal{C}\_{[0, 1]}$ space with the $L\_2$ norm for the sake of example. The function space and norm should be tailored to the intended application.
5. When $\mathcal{S}\_{k}$ and $\mathcal{I}\_{k}$ are selected, we can have a family of maps $F\_{k}$ that always operate with finite amount of information for each $k$. Discretisation invariance is a requirement for the map $\mathcal{I}\_{k}F_{k}\mathcal{R}\_{k}$ to converge to a continuous operator $\mathcal{F}: \mathcal{C}\_{[0, 1]} \rightarrow \mathcal{C}\_{[0, 1]}$ between function spaces.

To show that discretisation-invariant operators exist, we provide a simple example from numerical analysis. For continuous functions, the integral $G(x_i) = \int_{0}^{x_i} g(x)dx$ can be approximated by the Riemann sum $G(x_i) \simeq \sum_{j=1}^{i}g(x_j)/(N-1)$. We can represent this approximation with a sampling operator and linear operators $F\_{k}$ given by $k\times k$ given by lower triangular matrices:

$$
\left(F_{k}\right)_{ij} = \left\{
    \begin{array}{ll}
        \frac{1}{k-1}, &  \text{if }i\leq j;\\
        0, & \text{ otherwise}.
    \end{array}
\right.
$$

For a continuous function $g$, the composition of sampling, the Riemann sum and interpolation $\mathcal{I}\_{k}\left(F_{k}\left(\mathcal{S}\_{k}g\right)\right)$ converges to the Riemann integral $G(x) = \int\_{0}^{x}g(y)dy$.

## Examples and non-examples
An example with antiderivative operator suggests a general strategy to design discretisation-invariant architectures: formulate all operations on functions in continuous form and use discretisation techniques from numerical analysis to process functions consistently on grids with different resolutions <d-cite key="berner2025principled"></d-cite>, <d-cite key="li2020fourier"></d-cite>, <d-cite key="li2020neural"></d-cite>. Most discretisation-invariant architectures that we describe in this section follow this general recipe.

### Fourier Neural Operator
The Fourier Neural Operator (FNO) is the most famous and successful example of discretisation-invariant architecture <d-cite key="li2020fourier"></d-cite>. FNO is a feedforward neural network that uses three operations:
1. Convolution with kernel size $1$. For input functions $v^{i}(x)$ with $N$ "channels", the output is $\sum_{j=1}^{N}A_{ij}v^{j}(x)$.
2. Pointwise nonlinear activation.
3. Spectral convolution <d-cite key="rippel2015spectral"></d-cite> with truncation.

Spectral convolution is the only operation with spatial transfer of information. It can be understood either as a parametrization of convolution in the Fourier domain, where the convolution operator becomes diagonal <d-cite key="rippel2015spectral"></d-cite>, or as an efficient evaluation of integral operator $\int \sum_{j} s_{ij}(y - x;\theta) v^{j}(y) dy$ with particular kernel convenient for implementation. The kernel is chosen to be a periodic, finite bandwidth function, so the whole integral operator can be implemented in three stages:
1. Fourier transform of the input with truncation $\hat{v}^{j}\_{k} = \mathcal{F}(v^{j}(x))\_{k},\,k=1,\dots,k\_{\max}$.
2. Linear operator diagonal in Fourier space $\hat{w}^{i}\_{m} = \sum\_{i} R\_{ijm} \hat{v}^{j}\_{m}$. Coefficients of tensor $R$ are learnable parameters.
3. Inverse Fourier transformation with padding to restore original spatial shape $\mathcal{F}^{-1}\left(\hat{w}^{i}_{m}\right)$.

Spectral convolution with truncation is discretisation invariant by construction, since it approximates continuous integral kernels with standard techniques from numerical analysis.

Many other architectures follow similar design pattern, e.g., <d-cite key="tripura2022wavelet"></d-cite>, <d-cite key="gupta2021multiwavelet"></d-cite>, <d-cite key="tran2021factorized"></d-cite>, by either modifying the parametrisation of spectral convolution or replacing Fourier with other fast transformations.

### Convolutional neural networks
Architectures based on convolutional neural networks (CNNs), especially ResNet <d-cite key="he2016deep"></d-cite> and U-Net <d-cite key="ronneberger2015u"></d-cite>, are highly successful for operator learning problems <d-cite key="stachenfeld2021learned"></d-cite>, <d-cite key="raonic2023convolutional"></d-cite>. They are often applied in the form of "image-to-image" mappings, with both images being physical fields of interest computed on uniform grids.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/receptive_field_conv.png" class="img-fluid" %}
<div class="caption">
    For a convolution operator with kernel size $5\times5$, a regular point collects information from $5$ neighbours along each dimension (shaded area). When the grid is refined the receptive field of convolution shrinks, so each point receives data from a smaller patch of $(x, y)$ space.
</div>

What makes convolutional architectures interesting in our context is their ability to process inputs of different resolutions. However, as illustrated in the image above, the receptive field of CNNs in coordinate space will decrease with an increase in resolution. We will see below, that when a CNN is trained on a fixed resolution, data on a refined grid appears as out-of-distribution, leading to a sharp drop in accuracy. As a result discretisation invariance is not observed.

### Deep Operator Network with linear observations
Deep Operator Network (DeepONet) is a meta-architecture <d-cite key="lu2019deeponet"></d-cite> based on the universal approximation results for operator learning <d-cite key="chen1995universal"></d-cite>. The architecture consists of two arbitrary neural networks: branch network and trunk network. For the input function $v(x)$, the output $u(x)$ is computed as follows:
1. Branch net $b$ takes whatever information about $v(x)$ is available (e.g., finite number of samples at selected points $v(x_1),\dots, v(x_d)$) and outputs a set of coefficients $c_{1}, \dots, c_{b}$.
2. The final layer of trunk net $t_{1}(x),\dots,t_{b}(x)$ provides a global basis that does not depend on the input $v(x)$.
3. The output of the architecture is constructed from branch and trunk nets $u(x) = \sum_{i=1}^{b} c_{i}t_{i}(x)$.

Readers familiar with reduced order modelling may recognise that the scheme closely resembles a non-intrusive proper orthogonal decomposition.

To make a discretisation-invariant DeepONet, we select a set of predefined basis functions $\psi_1(x),\dots,\psi_m(x)$, and use them to form linear observations $o_i = \int \psi_{i}(x) v(x) dx$ which are later supplied to branch net. To compute linear observations, any numerical integration can be applied, e.g., [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).

### PCA-Net
PCA-Net is an encoder-processor-decoder architecture based on proper-orthogonal decomposition (POD) or Karhunen–Loève expansion <d-cite key="hesthaven2018non"></d-cite>, <d-cite key="bhattacharya2021model"></d-cite>. For an input $v(x)$ we compute the output $u(x)$ as follows:
1. Encoder finds coefficients $c = \inf_{c} \left\|v(x) - \sum_{i=1}^{d} c_i\phi_i(x)\right\|_2^2$, where $\phi_i(x)$ are precomputed as explained below.
2. Processor is a standard feedforward architecture that transforms a vector of coefficients to another vector $d_i,i=1,\dots m$.
3. Similar to DeepONet, decoder computes a linear combination $u(x) = \sum_{i=1}^{m} d_i \psi_i(x)$, where $\psi_i(x)$ are computed similarly to $\phi_i(x)$.

For PCA-Net functions $\phi\_i(x)$, $\psi\_i(x)$ are computed using POD <d-cite key="volkwein2013proper"></d-cite>. Let $v_{j}(x),\,j=1,\dots,N_{\text{train}}$ be inputs from the train set. Basis functions are recursively defined as solutions to optimisation problems

$$
\begin{aligned}
&\psi_i(x) = \arg \min_{\psi} \sum_{j}\left\|v_{j} - \psi\left(\psi, v_{j}\right)\right\|_{2}^{2},\\
&\text{s.t }\left\|\psi\right\|_2 = 1,\left(\psi_{k}, \psi\right) = 0,\text{ for }k < i.
\end{aligned}
$$

That is, precisely the same way as the principal components in PCA, but in functional space.

### Neural fields
This is another example of encoder-processor-decoder architecture where both input and output are approximated by a form of implicit neural representation <d-cite key="serrano2023operator"></d-cite>. Operators based on neural fields work precisely as PCA-Net but use a different approach to represent functions by finite-dimensional vectors.

To illustrate how basis functions are built, suppose we collected a dataset of inputs $v_i(x),\,i=1,\dots,N_{\text{train}}$. We select a neural network with weights $\theta$, that take coordinate $x$ in the first layer, and, in addition, vector $z$ in some hidden layer. We find parameters of the resulting architecture $\phi_{\theta}(x;z)$ by optimising the loss

$$
\min_{\theta} \left(\sum_{i=1}^{N} \min_{z_i}\left\|\phi_{\theta}(x;z_i) - v_{i}(x)\right\|_{2}^2\right).
$$

As a result, for the entire dataset we will compute global parameters $\theta$ that are shared among samples $v_{i}$, and for each individual sample we find a coding vector $z_{i}$. This finite coding vector is used as a representation of function $v_i(x)$. For new inputs outside of the training set, the optimization problem above is solved with fixed $\theta$ to find finite-dimensional representation $z$. The same is done for the targets, and after that we are left with the problem of learning maps between finite-dimensional spaces.

Note, that all operations in this scheme are formulated with no explicit discretisation, ensuring that the whole architecture is discretisation agnostic.

### Graph neural networks
Unstructured grids are very common in scientific computing, especially when complex geometries are involved. Given that, graph neural networks (GNNs) are a natural choice for building neural PDE solvers <d-cite key="brandstetter2022message"></d-cite>. A GNN is an example of an architecture that can handle variations in grid and geometry, but is nonetheless not discretisation invariant.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/receptive_field_GNN.png" class="img-fluid" %}
<div class="caption">
    An example of the receptive field for GNN. Similarly to CNN, the receptive field shrinks when resolution is increased.
</div>

The reason GNNs are not discretisation invariant is precisely the same as for CNNs: the architecture can be applied on a refined grid, but the receptive field is going to shrink leading to out of distribution inputs. The change in receptive field is illustrated in the picture above.

### Graph kernel networks
The Graph kernel network (GKN) is a discretisation-agnostic version of a GNN <d-cite key="li2020neural"></d-cite>. It replaces message passing with an integral operator

$$
v_{i+1}(\boldsymbol{x}) = \int_{B(\boldsymbol{x})} k_{\phi}(\boldsymbol{x}, \boldsymbol{y}, u(\boldsymbol{x}), u(\boldsymbol{y})) v_{i}(\boldsymbol{y}) d\boldsymbol{y},
$$

where $v_{i+1}(\boldsymbol{x})$ is the output of the layer, $v_{i}(\boldsymbol{y})$ is an input, $B(\boldsymbol{x})$ is a ball of predefined radius around $\boldsymbol{x}$, and $u(\boldsymbol{x})$ is an input to the network, e.g., a diffusivity coefficient in stationary diffusion equation.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/receptive_field_KNN.png" class="img-fluid" %}
<div class="caption">
    For graph kernel networks, the receptive field, defined in $(x, y)$ space, is a hyperparameter of the architecture. When the grid is refined a graph of nearest neighbours is recomputed, ensuring discretisation invariance of the architecture.
</div>

A convenient way to approximate integral above is to use Monte Carlo method

$$
v_{i+1}(\boldsymbol{x}) = \frac{1}{N_{mc}}\sum_{l=1}^{N_{mc}} k_{\phi}(\boldsymbol{x}, \boldsymbol{y}_{l}, u(\boldsymbol{x}), u(\boldsymbol{y}_{l})) v_{i}(\boldsymbol{y}_{l}),
$$

where $\boldsymbol{y}\_{l}$ are points inside a ball $B(\boldsymbol{x})$ as shown in the figure above.

Importantly, the radius of the ball $B(\boldsymbol{x})$ is not related to discretisation used, and because of that when the grid is refined the finite sum approximates the same integral using more terms.

## Training and evaluation
We demonstrate discretisation invariance on three architectures and two PDEs. The code is available [in this repository](https://github.com/VLSF/discretisation_invariance). The software used includes JAX <d-cite key="deepmind2020jax"></d-cite> and Equinox <d-cite key="kidger2021equinox"></d-cite>.

The first PDE is the Burgers equation

$$
\frac{\partial u(x, t)}{\partial t} + \frac{1}{2}\frac{\partial \left(u(x, t)\right)^2}{\partial x} = \nu \frac{\partial^2 u(x, t)}{\partial x^2},
$$

with Dirichlet boundary conditions $u(0, t) = u(1, t) = 0$, the initial conditions sampled from a Gaussian random field, and a viscosity $\nu = 0.1$.

For the Burgers equation, neural networks were trained to predict $u(x, 0.3)$ from the initial condition $u(x, 0)$. Examples of features and targets are shown on the figure below.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/Burgers_samples_II.png" class="img-fluid" %}
<div class="caption">
    Samples from the Burgers dataset: initial conditions $u(x, 0)$ in the first row (features), and the solution $u(x, 0.3)$ in the second row (targets).
</div>

The second dataset is based on the stationary diffusion equation

$$
-\frac{d}{dx}\left(k(x) \frac{d \phi(x)}{dx}\right) = 1,\,\phi(0) = \phi(1) = 0,
$$

with $k(x)$ sampled from a Gaussian random field with a transformation that ensures: (i) $k(x) > 0$ and (ii) large spatial variability.

The task for this equation was to predict $u(x)$ from $k(x)$. Samples from the dataset are shown below.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/diffusion_samples_II.png" class="img-fluid" %}
<div class="caption">
    Samples from the diffusion dataset: diffusivity coefficients $k(x)$ in the first row (features) and solutions $\phi(x)$ in the second row (targets).
</div>

On these two datasets we train three architectures: FNO, U-Net and DeepONet with linear observations. The experimental setups is the standard one used to demonstrate discretisation invariance <d-cite key="li2020fourier"></d-cite>:
1. We generate a dataset on a grid with $N=512$ points.
2. The neural network is trained on a downsampled version with $N=64$ points.
3. After training it is evaluated on grids with $64, 128, 256, 512$ points.

The results of the experiments are reported below.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/discretisation_invariance.png" class="img-fluid" %}
<div class="caption">
    Relative test error for neural networks trained on grids with $64$ points and evaluated on grids with higher resolutions. Discretisation-invariant architectures, DeepONet and FNO, show only mild variations in relative error when resolution increases. In contrast, the relative error of U-Net instantly reaches $>100\%$ when the grid is refined.
</div>

It is quite clear that a discretisation-invariant architecture tolerates a resolution increase much better than a U-Net. The accuracy of DeepONet even slightly improves with resolution, while the accuracy of FNO slowly decreases.

## The role of discretisation invariance
We have seen the definition of discretisation invariance and typical numerical demonstration that shows the difference between classical and discretisation-agnostic architectures. It is not hard to find dozens of papers that develop novel discretisation-agnostic architectures and confirm their discretisation invariance through experiments like the one we reproduce above. Accuracy is still important, but discretisation invariance often becomes a goal in itself. Clearly, we want to develop discretisation-invariant architectures for some reason. Unfortunately, it is rarely explained *why* we need this property. We would like to invite the research community to a discussion on the purpose of discretisation invariance. Below, we provide our arguments both for and against this property.

### Arguments against discretisation invariance
The main problem is the tension between the infinite-dimensional setting, where discretisation invariance is relevant, and the fixed resolution setting used in practice. Several manifestations of this problem are described below.

**Discretisation invariance is an asymptotic property**

Both the formal definition and the typical evaluation strategies frame discretisation invariance as a statement about convergence <d-cite key="azizzadenesheli2024neural"></d-cite>. Convergence is, undeniably, crucial in classical scientific computing, as it theoretically allows one to reach arbitrary accuracy on paper and very high precision on a digital computer <d-cite key="bailey2012high"></d-cite>. However, the benefit is limited in operator learning. Test error rarely drops below a certain floor (e.g., $10^{-4}$), and given that training data always has finite resolution, it is unlikely that accuracy will increase significantly beyond what was achieved during training. Indeed, in our experiments above, we did not observe a palpable increase in accuracy  tied to this property. The related claim on "zero-shot superresolution" made in <d-cite key="li2020fourier"></d-cite> has similarly remained unverified in subsequent research <d-cite key="fanaskov2023spectral"></d-cite>, <d-cite key="sakarvadia2025false"></d-cite>.

**Functional Data Analysis and fixed-resolution setting**

Functional Data Analysis (FDA) is a long-established field <d-cite key="ramsay1991some"></d-cite>. FDA literature focuses on functions as objects of interest, examining classification, regression and interpolation within Banach spaces. This focus makes FDA highly related to both discretisation invariance and operator learning. Peculiarly, researchers in FDA and operator learning rarely reference each other. If we look at the modern application of machine learning, it becomes clear that the FDA is not widely popular. Some reasons for this are summarised in [the illuminating discussion on StackExchange](https://stats.stackexchange.com/a/564607). A key argument is that since all observations are finite, it is often possible to develop an algorithm for finite data that outperforms FDA algorithms.

Fixed resolution setting is so convenient and versatile that it is applied even when the problem has conspicuously multiresolution structure. A clear example is the deterministic weather forecast. The typical setup is fixed-resolution training and evaluation on the ERA5 dataset, where transformer-based models dominate the leaderboard <d-cite key="liu2024evaluation"></d-cite>. To be fair, discretisation-invariant architectures (like FourCastNet <d-cite key="pathak2022fourcastnet"></d-cite>) are also comparable or superior to classical weather prediction models, but whether discretisation invariance itself is a key driver of their success remains unclear.

**Discretisation invariance is not a good indicator of performance**

In <d-cite key="berner2025principled"></d-cite>, the authors argue that essentially any architecture can be made discretisation invariant. An interesting consequence of this finding is that discretisation invariance ceases to be a good guiding principle for architecture design. Architectures perform well or poorly independently of this property, since any design can be made discretisation invariant after mild adjustments. Let's consider FNO as a prime example.

FNO is certainly a high-performing architecture, but why? Is its success attributable to discretisation invariance? The main component of FNO is the spectral convolution - a convolution layer parametrised in Fourier space. In the seminal paper <d-cite key="rippel2015spectral"></d-cite> where it was introduced, the authors observed that for the fixed-resolution problem of image classification, architectures incorporating spectral convolutions and pooling converge $2$ to $5$ times faster and achieve improved accuracy compared to classical CNNs. What's the reason? Whatever the specific driver, this performance gain was demonstrated entirely within a fixed-resolution setting and therefore operates independently of discretisation invariance.

### Arguments in favour of discretisation invariance
Several failure modes we described above happen when one is trying to infer data at a higher resolution than was available during training. We argue that a more productive strategy is to leverage downsampling - that is, to intentionally decrease input resolution for computational gains. Several powerful applications based on this downsampling approach are described below.

**Hyperparameter optimization**

Discretisation invariance can significantly decrease the cost of a grid search <d-cite key="fanaskov2025deep"></d-cite>. The strategy involves first decreasing the dataset resolution to the smallest scale the architecture allows. A standard grid search is then performed, and the $K$ configurations with the highest validation accuracy are recorded. Finally, these $K$ configurations are tested again on the full resolution dataset. The authors of <d-cite key="fanaskov2025deep"></d-cite> report a reduction in grid search time by a factor of $7$, using this strategy.

**Multiresolution inference**

In <d-cite key="yao2025guided"></d-cite> authors combine denoising diffusion probabilistic model with discretisation-agnostic neural operator. A natural sampling approach for such a combined model is to perform most denoising steps in low resolution and then transition to high resolution only toward the end of the process. Authors report that such strategies speed up the overall process by a factor of $2$.

**Pretraining and finetuning**

Neural operators can be efficiently trained on multiresolution datasets <d-cite key="li2024multi"></d-cite>. For a given parametric PDE problem, one selects a desired high resolution $N\_x\times N\_y$ and generates a small set of train samples. These high-resolution samples are then supplemented by a large number of low resolution samples generated on a coarse grid $n\_x\times n\_y$. The neural operator is initially pretrained on the large low-resolution dataset and subsequently finetuned on the small high-resolution dataset, optimizing training efficiency.

## Discussion
We have discussed discretisation invariance from both theoretical and practical standpoints. We argue that the extrapolation from low to high resolutions does not work well in practice. A more productive direction is to select a desired resolution - a standard fixed-resolution setting pervasive in deep learning - and leverage downsampling. In our view, the benefits of the discretisation invariance are not yet completely understood, and in general, this property remains under-exploited.

---
