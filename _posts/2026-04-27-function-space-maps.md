---
layout: distill
title: "Learning Function Space Maps: A Red Herring?"
description: Much interest has been generated in the space of learning function space maps, such as in deep operator networks and neural operators. In this post, we explore whether viewing data in their underlying, infinite-dimensional form offers benefits in the manner professed or whether this is a fad.
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
#   - name: Yash Patel
#     url: "http://ypatel.io/"
#     affiliations:
#       name: Anthropic, University of Michigan

# must be the exact same name as your blogpost
bibliography: 2026-04-27-function-space-maps.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: The Prestine World of Mathematics
  - name: Partial Differential Equations
  - name: Neural Operators
  - name: Surrogates for Engineering Design
  - name: The Utility of Neural Operators
    subsections:
      - name: Call to Action

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

## The Prestine World of Mathematics

Arguments between the "discovery" and "invention" of math date back to the advent of the field. We believe, for instance, that math largely exists in the minds of people, existing in a platonic idealization of reality that lends itself to careful manipulation in an almost unprecedented manner that allows people to escape from their own intuitions. That is, by turning the crank of mathematics, people often discover surprising phenomena that they otherwise would never have predicted. This is not merely that humanity more generally are surprised by the findings uncovered by mathematical manipulation, but even the people who are *doing the cranking* are oftentimes themselves surprised.

Mathematics, however, often gets sullied, in the eyes of mathematicians, when brought down to the messiness of reality. Perfect geometric shapes are replaced by curves with tolerances from manufacturing defects. Clean, analytic manipulations give way to numerical approximations. What, then, is the use of these mathematically pure objects if they eventually require such degradation?

In particular, we focus this question on the recent rise of interest in using ML over function spaces to see whether this offers legitimate benefits over the traditional, discretized counterparts.

## Partial Differential Equations

Before diving into neural operators, let us first take a detour into the mature study of partial differential equations (PDEs). PDEs are a crowning achievement of physics, having produced remarkably predictive equations for many phenomena in nature. One notable example is the heat equation

$$ \frac{\partial u(x,t)}{\partial t} = \nabla u(x,t) $$

In the heat equation, $u(x,t)$ is a spatiotemporal field, meaning that it maps $u : \mathbb{R}^{d}\times\mathbb{R}\to\mathbb{R}$, a position in the spatial domain and time in the temporal domain to a temperatue reading. "Solving" this PDE then means, given a specification of the *initial* state of the system, can we predict the evolution of the system state? That is, given the initial temperature distribution $u(\cdot, 0)$, can we predict the temperature field at a later time $u(\cdot, T)$?

An interesting property of PDEs is that, while it may be feasible to analytically posit them, it is incredibly rare to find one that can be analytically solved. To circumvent this deficiency, we are often forced to approximate the dynamics and simulate the system evolution. Let us briefly switch to a simple ODE to understand this approach before returning to the PDE case. 

Suppose we has a basic ODE $\frac{dx}{dt} = f(x)$ and that we did not know how to solve this particular ODE. Recall that a derivative is just the limit of a difference:

$$ \frac{dx}{dt} = \lim_{\Delta t\to0} \frac{x(t + \Delta t) - x(t)}{\Delta t} $$

An intuitive idea, therefore, is to simply replace the derivative with this approximation:

$$ \frac{x(t + \Delta t) - x(t)}{\Delta t} = f(x(t)) $$

From this, we can now take any initial state $x(0)$ and evolve to a time $x(\Delta t)\approx f(x(0)) \Delta t + x(0)$. We can then use this to get $x(2 \Delta t)\approx f(x(\Delta t)) \Delta t + x(\Delta t)$ and so on. Naturally, there is a deep literature of numerical methods to solve ODEs. These approaches, however, fundamentally all propose a conceptually similar approach to that discussed above: discretizing the differential operator.

Returning to the space of PDEs, a similar approximation exists, in which the differential operators are replaced by their numerical counterparts and iterated to arrive at the answer. The time derivatives are treated analogously to in the ODE setting. To approximate spatial derivatives, we then need an analogous discretization to the $\Delta t$ used in the time domain. In a 1D spatial domain, a natural choice is to use an interval length $\Delta x$, in which case the $\nabla = \frac{\partial^2}{\partial x^2}$ becomes

$$ \nabla u(x, t) \approx \frac{u(x + \Delta x, t) - u(x - \Delta x, t)}{2 \Delta x}. $$

To characterize the full evolution of $u(\cdot, t)$, the intuitive strategy is to then alternate between solving for the spatial field at a fixed time step $t$ and evolving this field to the next time step.

## Neural Operators

A core issue with numerical solvers is that any new initial condition $u(\cdot, 0)$ or problem specification must be solved anew. This, however, is a failing of the method rather than something fundamentally true of PDE solutions. After all, analytic solutions are such a method that allows for the "instantaneous" solving of newly specified problems; if we have an analytic solution of a PDE $\mathcal{G}$, any new initial condition $u(\cdot, 0)$ can be plugged in to arrive at the solution $\mathcal{G}(u(\cdot, 0)) = u(\cdot, T)$. Given the success of, for example, AlphaFold in protein structure prediction in cases where we could not hand-specify an analytic folding function <d-cite key="abramson2024accurate"></d-cite>, a natural question to ask is whether such PDE solution maps can be learned from data.

In this vein, a slew of works have arisen, such as <d-cite key="li2020fourier,bonev2023spherical,liu2025difffno,lu2019deeponet,wang2021learning"></d-cite>. These maps that approximate the solutions of PDEs are termed "neural operators." An "operator" in math is simply a map between two function spaces $\mathcal{G} : \mathcal{U}\to\mathcal{V}$. Often, these function spaces are taken to be Hilbert spaces over a class of sufficiently smooth functions. <d-footnote>We will ignore the technical details of smoothness in this discussion, but for those interested, Sobolev spaces are typically employed to ensure the differential operators are well defined.</d-footnote> A "neural" operator, therefore, is simply a learned approximation of this mapping $\mathcal{G}_{\theta}$.

In the above example, we were interested in learning the mapping between initial and final conditions. A natural setup, therefore, would be to have a dataset $\mathcal{D} = \{(u^{(i)}(\cdot, 0), u^{(i)}(\cdot, T))\}_{i=1}^{N}$ pairs, where $u^{(i)}(\cdot, T) = \mathcal{G}(u^{(i)}(\cdot, 0))$ for the true operator $\mathcal{G}$, and to then train this neural operator in a typical MSE fashion:

$$ \theta^* := \mathrm{arg}\min_{\theta} \sum_{i=1}^{N} || \mathcal{G}_{\theta}(u^{(i)}(\cdot, 0)) - u^{(i)}(\cdot, T) ||^{2}_{\mathcal{U}} $$

However, therein lies the problem: unlike in typical machine learning tasks, the data here are fundamentally unobservable. This is in the sense we described earlier: a function only exists in the pristine world of mathematics. We, therefore, cannot construct ever truly produce such a dataset $\mathcal{D}$.

Instead, datasets consist of solved PDEs, in the manner described previously. That is, the dataset consists of *discretized* fields. What, then, does it mean for neural operators to treat these observations as functions if we only observe them as discretized fields? At their core, neural operators assume the discretized observations arise from an underlying field, thus allowing the discretization to be nonuniform across the dataset. In particular, if each datapoint is discretized with some grid $\mathcal{X}_{i}\,$ the function loss function is then given by the finite-dimensional norm induced on $\mathcal{U}$. For instance, in the common case of $\mathcal{U} = \mathcal{L}^{2}(\mathcal{X})$, this loss becomes

$$ \sum_{i=1}^{N} \sum_{x\in\mathcal{X}_i} || \mathcal{G}_{\theta}(u^{(i)}(x, 0)) - u^{(i)}(x, T) ||^{2} $$

The reason to leverage neural operators, therefore, is to take advantage of datasets with datasets with non-uniform grids. Returning to the original focus, is this property worthy of further investigation or is the field over-indexing on this property?

## Surrogates for Engineering Design

To answer this question, it is worthwhile investigating the use of neural operators. As mentioned, neural operators are intended to learn the solution maps of PDEs to allow for rapidly evaluation of different initial conditions to circumvent our inability to produce such a mapping analytically. Critically, this differs in a fundamental sense from the AlphaFold application. In the case of AlphaFold, we previously had *no* method to map from amino acid sequences to 3D structures. In contrast, we *do* have solution maps in the case of PDEs: numerical solvers. The only problem with numerical solvers is that they are incredibly costly, making solutions over collections of initial conditions slow. 

This makes the value proposition of neural operators fundamentally different from AlphaFold: the question is, are there settings where the rapid solution of PDEs justifies a potential loss in accuracy in using this learned surrogate in place of the numerical solver? A common application is in the rapid evalution of engineering designs. For instance, in the design of aircraft or vehicles, an engineer will propose a potential shape of the aircraft wing or car chassis, for instance, which then gets evaluated by running a computational fluid dynamics (CFD) solver to estimate the drag expected of the proposed design.

{% include figure.liquid
     path="assets/img/2026-04-27-function-space-maps/cfd.png"
     class="img-cfd"
%}

<p class="figcaption">
  CFD simulation of flow around an airfoil.
  <d-cite key="krishnamurthy2018meshCFD"></d-cite>
</p>

Naturally, such simulations will often reveal deficiencies in the proposed design, revealing regions of the original design producing especially high drag. With such insights, engineers can then improve the design, iteratively producing more and more efficient or cost-effective designs. This, however, brings us back to the major deficiency of numerical methods: since the cost of running a solver both high and is not amortized across runs, iterating over designs becomes slow. In contrast, neural operators allow for rapid estimation of the quality of a design, in turn allowing engineers to far more rapidly iterate on designs or even do so in an automated fashion.

For this reason, much of the practical use of such neural operators has focused on its surrogate use in design optimization, such as in materials <d-cite key="michaloglou2025physics,wang2026micrometer,jin2025characterization"></d-cite> and airfoil shape <d-cite key="shukla2023deep,shukla2024deep"></d-cite>. Formally, these surrogate models enable us to more efficiently solve a "PDE-constrained optimization" problem,

$$ \min_{\theta} J(u_{\theta}) \qquad \mathrm{s.t.} \quad D_{\theta} u = f_{\theta} $$

where $D_{\theta} u = f_{\theta}$ is the PDE that the field must satisfy (i.e., the Navier-Stokes equations in the case of the fluid flow around a car) and $J(u_{\theta})$ is the functional of such a field (i.e., the drag resulting from such a flow field).

## The Utility of Neural Operators

Notably, the above discussion regarding neural surrogates makes no explicit reference to discretization invariance. In other words, any surrogate model, learned or hand-crafted, could serve equally well for the above purpose. In that case, are discretization-invariant surrogate models useful for design optimization? It depends.

Even though we specifically discussed the map from initial to final conditions in the discussions above, neural operators are not limited to this application. For instance, the Poisson
equation can be used to model the stationary temperature distribution with a heat source function $f(x)$ as

$$ \nabla u(x) = -f(x) $$

In this case, there is no temporal dependency and one may wish instead to learn the map $\mathcal{G} : f\to u$. Such an approximation can be useful if, for instance, one has control over the source heat distribution (i.e., with local thermostats) and wishes to achieves a target thermal profile over a room. A discretization-invariant surrogate would then be useful if, across different design choices $f$, the ideal discretization would vary. While seemingly artificial, this phenomenon naturally arises when using numerical solvers. 

Suppose one has a spatial domain where a field is rapidly varying: using a fixed discretization $\Delta x$ for such regions as well as those that vary more gradually would result in poor approximations of the spatial gradient. For this reason, the spatial discretization is oftentimes non-uniform and adjusted to closely follow where it is expected more rapidly fluctuations of the solution field will exist, such as around sharp curves of a car or wing design in a fluids simulation.

{% include figure.liquid path="assets/img/2026-04-27-function-space-maps/mesh.jpg" class="img-mesh" %}

<p class="figcaption">
  Non-uniform discretization is often necessary to capture high frequency variation in the fields over the spatial domain.
  <d-cite key="centaur2025MeshRefinement"></d-cite>
</p>

The discretization invariance of the surrogate, therefore, is useful in settings where the meshing will need to adapt as a result of the design iteration. This is necessarily true in settings of shape optimization, where the domain shape is precisely the variable of optimiation, and even in many other cases of source property optimization, as such cases often require variable meshes to align well with each design iterate.

### Call to Action

While the above section highlights cases where a discretization-invariant surrogate could be useful, little work has yet concentrated on mesh adaptation concurrent to design iteration with neural operators. That is to say, little direct empirical evidence exists for the advantages afforded by neural operators for design iteration over fixed resolution surrogate maps. For this reason, a necessary step in the development of this field is to empirically validate whether this finding truly holds up to empirical scrutiny.