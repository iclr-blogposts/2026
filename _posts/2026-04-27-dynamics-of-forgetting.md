---
layout: distill
title: Dynamics of Forgetting
description: We analyze catastrophic forgetting through spectral decompositions of weights and updates, revealing when optimization refines existing circuits versus builds interfering new ones. Leveraging this, we design spectral techniques that suppress destructive update components while preserving structure.
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

# must be the exact same name as your blogpost
bibliography: 2026-04-27-dynamics-of-forgetting.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
# toc:
#   - name: Introduction
#   - name: Why do Models Forget
#   - name: Setting the Stage: the Stability vs. Plasticity Trade-off
#   - name: Observing Spectral Training Dynamics
#   - name: Two Types of Forgetting
#   - name: Towards (Spectrally) Graceful Updates
#   - name: Going Deeper
#   - name: Experimental Results
#   - name: Other Typography?
#     subsections:
#       - name: Toy Model: MLP on MNIST
#       - name: LLMs: Sequential SFT on Q&A tasks
#   - name: Final Notes
#   - name: Appendix
toc:
  - name: Introduction
  - name: Why Do Models Forget
  - name: Setting the Stage
  - name: Observing Spectral Training Dynamics
  - name: Two Types of Forgetting
  - name: Towards (Spectrally) Graceful Updates
  - name: Going Deeper
  - name: Experimental Results
    subsections:
    - name: Toy Model - MLP on MNIST
    - name: LLMs - Sequential SFT on Q&A Tasks
  - name: Final Notes
  - name: Appendix


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

## Introduction

{% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/cover.png" class="img-fluid" %}

Most of our understanding of the world is built through a continuous interaction with our surroundings. 

This marks a very distinct difference between Artificial and Human intelligence: while modern AI models are stuck in a training / inference dichotomy, human intelligence (and, more generally, intelligence found in nature) blurs the line between the two, integrating a subconscious process of accumulation and processing of new information throughout our interaction with the world. 

This makes continual learning, to us, the most pressing problem in modern Artificial Intelligence. 

Most existent continual learning methods (<d-cite key=kirkpatrick_overcoming_2017></d-cite><d-cite key=saha_gradient_2020></d-cite><d-cite key=wang_training_2021></d-cite><d-cite key=zenke_continual_2017></d-cite><d-cite key=aljundi_memory_2018></d-cite><d-cite key=farajtabar_orthogonal_2020></d-cite><d-cite key=lopez_paz_gradient_2017></d-cite><d-cite key=chaudhry_efficient_2019></d-cite><d-cite key=riemer_learning_2019></d-cite><d-cite key=shin_continual_2017></d-cite>) revolve around the idea of making a specific subset of the training data *harder to forget.* Through either replay buffers, reduced learning rates on the parameters that are more important to interpret *a specific* subset of data, or orthogonalization of the updates to task-specific directions, the model is strongly penalized whenever relevant information is forgotten.

All of these methods might work in specific multi-task scenarios, but the interactions that we, as humans, experience with the surrounding world are rarely ever categorizable in separate tasks.

To begin pondering the possibility of human-like lifelong learning, we can’t rely on manually “saving” a small amount of specific subsets of data from catastrophic forgetting: we need a learning paradigm that allows to simply *not forget anything,* and learn new things without damaging previous knowledge.

This blogpost will propose a strongly personal point of view on:

- Why do models forget, and whether we can distinguish between different types of forgetting;
- How does the core phenomenon of catastrophic forgetting look like from a spectral perspective;
- A possible approach towards more graceful optimization, to preserve learned structure. 

<!-- This post is also heavily inspired by a model merging perspective: a lot of model merging research is focused on keeping downstream capabilities of models that have been trained independently, when joined to form a single model. In a way, updating the model during training performs a small merge every time. Maybe, we could transfer insights on how to reduce task inference in MM to a different and exciting area. -->

## Why Do Models Forget


Catastrophic forgetting stems as an intrinsic consequence of how models are optimized. Gradient descent would like to minimize a loss function over a dataset of samples. Since this is usually computationally unfeasible, *Stochastic* Gradient Descent only concerns itself with a tiny, random portion of the data, and tunes the parameters to minimize the objective function on that subset of data. This means that each iteration has only one goal: optimize the parameters to predict the current batch as accurately as possible, as if no other batch ever existed.

This leads to a constant flow of *construction* and *destruction* of structure in the weight matrices, that hopefully balances itself over enough training steps with large enough batches. 

To mitigate the damage of this oversimplification of the learning paradigm, two main practices stuck around over the years:

- Learning rate schedulers apply a very simple yet effective heuristic: if the model has been updated a lot of times, then probably it has built a lot of structure in the parameters that is worth preserving. Therefore, it makes sense to lower the magnitude of the updates to avoid destroying that structure.
- Better optimizers, like Adam, add two improvements: i) stabilize the trajectory of the updates, by accelerating descent through consistent paths, and decelerating through inconsistent ones; ii) optimize more strongly parameters that have been optimized fewer times.


## Setting the Stage
This kind of techniques can be seen, from a continual learning point of view, as heuristics that try to balance the ability of a model to keep learning from new examples, while mitigating the amount of destruction that a new batch induces on the information that was previously learned. We like to think of this balance as a trade-off between *stability* and *plasticity* (<d-cite key=kim_achieving_2023></d-cite><d-cite key=chen_stabilityplasticity_2023></d-cite><d-cite key=jung_new_2023></d-cite><d-cite key=lu_rethinking_2025></d-cite>). The former represents the ability of a model to preserve learned structure (i.e:  circuits, patterns…), the latter stands for the potential to learn new structures when needed (for example, when a change in the distribution of training samples occurs).  

In the traditional optimization scenario, this trade-off looks like a zero-sum game: stronger optimization (higher learning rate), increases plasticity at the cost of heavier forgetting; more delicate optimization (lower learning rate), increases stability by reducing the model’s ability to learn new information.

We think that, to better navigate this tradeoff, a change in the optimization paradigm is needed. We like to think of this in terms of *gracefulness* of the updates. A graceful update is one that improves performance on the current batch without overwriting or degrading previously accumulated structure in the weights. There is a very fine line that keeps this from being an intrinsically contradictory statement: a good update should, by definition, update the model’s perception of the world (the training data) in a generalizable way; so how could it work without altering and refining previously learned connections and circuits?

## Observing Spectral Training Dynamics

{% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/cover2.png" class="img-fluid" %}

To answer, we need to go deeper into the dynamics of learning and refining knowledge. Each optimization step computes update directions that are entirely specific to the current batch. Intuitively, we can think of an update as a mixture of two macro-types of operations:


- *Refining present structure*: existing information encoded in the weights is updated according to its performance on the current batch;
- *Building new structure*: a new encoding is built ad-hoc for the data in this batch.

The first targets encoded structure that is strongly learned and optimized and tries to tweak it to make it work on the current batch. The second takes a weak, raw part of the network and begins to shape and refine it using the data in the current batch.

We can try to observe these behaviors by decomposing with SVD both a weight matrix

$$W = U_w \Sigma_w V^T_w$$

and the update applied to it at each optimization step,

$$ \Delta = U_\Delta \Sigma_\Delta V^T_\Delta $$

and plot the alignment between the $U_w$ vectors of the weight and the $U_\Delta$ vectors of the update.

$$ M_{i, j} = \frac{|U_w^i \cdot U_\Delta^j|}{\left\lVert U_w^i \right\rVert  \|U_\Delta^j\|} $$ 

An alignment between strong update directions and strong weight directions shows destructive interference of an already highly optimized structure in the weights. Alignment between top update directions and weak weight directions represents the second type of update, in which new structure is being built to encode new information.

Let’s now observe an EMA of this matrix while training a 2-layer MLP on MNIST with Adam.

<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-dynamics-of-forgetting/video1.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="400px"
    width="100%">
  </iframe>
</div>

<div class="caption">
    Left, top: singular values of the weight matrix;  
    Left, bottom: singular values of the update proposed by Adam for the same matrix;  
    Center: heatmap of alignment between left singular vectors of the update and left singular vectors of the weight;  
    Right: Validation Accuracy on MNIST.  

    The high values concentrated on the top left of the heatmap show strong alignment between the most important spectral components of the update and the weight. In other words, the optimizer focuses mostly on updating structures which are already very strong.
</div>

We clearly see that, after an initial stage of randomness, the training quickly converges to always updating a few directions in weight space. 

This behavior is even more visible when we reduce the hidden dimension of the linear layer analyzed to 2. The training clearly switches from a *structure building* state, in which updates are not necessarily aligned with the top directions in the weight matrix, to a *refining* state, in which the update assigns its focus proportionately to the importance of each direction. This switch also synchronized with the moment in which the validation accuracy’s derivative drops below 1 and the model enters a training phase of much slower improvement.

<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-dynamics-of-forgetting/video2.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="400px"
    width="100%">
  </iframe>
</div>

## Two Types of Forgetting

{% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/cover3.png" class="img-fluid" %}

So what causes forgetting, exactly? 

Our previous distinction between two macro-types of updates leads us to think about forgetting in terms of two easily distinguishable patterns. As an example, let’s think of the case of training a simple MNIST classifier. 

<!-- Note that we are not constraining our thought experiment to a sequential multi-task scenario: we believe that forgetting happens in every stochastic gradient descent training procedure. It’s just that, when we can treat each batch as a random sample from the same data distribution, these destructive updates even out over time.  -->

1. *The current sample destroys structure that was learned on different data.*

    In our example, this could look like the model encountering a new digit, and applying to it a series of weights built to recognize other digits, leading to a wrong prediction. The weight update based on this new sample will interfere with the existing circuits built from the previous digits, overwriting weights that were needed to recognize them.

2.  *The model learns entirely new structure on the current batch of data.* 

    Here, the model might see a new digit and build ad-hoc structure to encode / recognize it. These new paths might now incorrectly fire for other digits as well, interfering with their prediction.

In both cases, the problem consists in the model learning something on the current data which doesn’t apply to different data, but the approach to mitigate these two phenomena is drastically different. The first can be solved by applying updates that preserve the current structure. The second cannot. 

Most continual learning methods, as well as the rest of this blogpost, are only concerned with the first *(destructive)* type of forgetting, which is much easier to address.

We believe that analyzing and determining how much of each of these types of forgetting actually occurs at a given time in a continual learning procedure is one of the most interesting open problems in continual learning.

To observe spectral behavior in presence of strong forgetting, we can look at the alignment matrix when switching training tasks from MNIST to Fashion-MNIST.

<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-dynamics-of-forgetting/video3.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="400px"
    width="100%">
  </iframe>
</div>

It seems that, when switching to the new task of Fashion-MNIST:

- The top 3 components of the update switch from targeting already strong structures to building weaker ones (in the first 3 columns of the matrix, the “heat” disappears from the top and shifts downward). This would represent type 2, or *constructive* forgetting.
- The remaining important components of the update keep targeting already strong structure (a strong heat spot remains visible in the top-left sector of the matrix). This would seem to indicate type 1, or *destructive* forgetting.

So, are we able to do something to at least mitigate the *destructive* forgetting that happens at each optimization step?

## Towards (Spectrally) Graceful Updates
Given this (very raw and certainly incomplete) analysis, we can try to intervene by observing the spectral characteristic of each weight matrix, and manually tweaking each update to make it more graceful. Ultimately, we want to *refuse* components of the update that wrongfully penalize weights that were optimized for very different data.

Let’s start with an oversimplification and divide the spectral decomposition of a matrix into low-rank (or strong) components and high-rank (or weak) ones. Let us also assume that a low-rank update on a direction changes it strongly, while a high-rank update only refines it slightly.

Now we can define a policy that decides which components of an update can be allowed, and which ones need to be refused as potentially destructive, based on their alignment with the corresponding singular vectors of the weight matrix. 

A very simple example policy could look as follows:

- Strong updates on strong directions are clearly destructive of previously accumulated knowledge: we want to refuse them;
- Strong updates on weak directions are exactly what we want: they build new structure for the new data without destroying old ones;
- Weak updates on strong directions are also probably ok, as they only slightly refine strongly optimized weights;
- Weak updates on weak structures are hard to characterize: they might be noise, or very low-importance updates in general. For this example, we refuse them.


We can implement this policy and immediately see the structure we are enforcing in the alignment matrix: our policy divides the matrix into four quadrants and only allows updates belonging to two of the four possible combinations (top-right and bottom-left).


<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-dynamics-of-forgetting/video4.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="400px"
    width="100%">
  </iframe>
</div>

## Going Deeper

A matrix-wise intervention on modern deep learning models can be extremely impactful, with the [Muon optimizer](https://kellerjordan.github.io/posts/muon/) being the prime example. Yet, we can’t but notice that models will inevitably encode information in the composition of their layers, their *circuitry*, rather than just atomically in the weights of each. This exposes a clear weak point in our technique so far: it only tries to identify and preserve structure in each matrix individually, but it never looks at their composition.   

While we’re still far from having a practical solution to this, we have some initial approaches that seem to bring improvements over the previous update conditioning function (or *post-conditioner*, as we like to call it).

Our deeper approach shares the same core idea as the previous one, but it does so taking into account a key element: cross-layer alignment. 

Let

$$
W^{(l)} = U^{(l)}\Sigma^{(l)}V^{(l)T} , \qquad \Delta^{(l)} = U_\Delta^{(l)}\Sigma_\Delta^{(l)}V_\Delta^{(l)T} 
$$

be a weight matrix for layer $l$ in a model and the corresponding update proposed by a chosen optimizer, such as AdamW or Muon

In a matrix-wise post-conditioner, we use some function  $f(W^{(l)},\Delta^{(l)})$ to re-scale the singular sub-spaces spanned by $\Delta^{(l)}$, so that the update is not destructive.

For this post-conditioner, we instead re-scale them based on a function 


$$
f_{comp}(W^{(l-1)},\Delta^{(l-1)}, W^{(l)},\Delta^{(l)}, W^{(l+1)},\Delta^{(l+1)}). 
$$

Of course, using $l+1$ and $l-1$ implies a notion of ordering of the weight matrices, which is definitely not trivial to define in a more complex network, such as an LLM.

The intuition of $f_{comp}$ is that the subspaces of $\Delta^{(l)}$ are now also rescaled based on the alignment matrices

$$
 T^{(l+1)} = V^{(l+1)T}U^{(l)} , \qquad T^{(l-1)} = V^{(l)T}U^{(l-1)}.
$$

The core insight is as follows: if incoming and outgoing spectral spaces align, then there is sign of structure in the weights, and we don’t want to be updating it too aggressively, so we reduce the magnitude of the update based on how aligned they are. 

So, can these style of approaches actually mitigate destructive updates and help in the continual learning setting?

## Experimental Results
Truth is, we don’t really know yet. In some specific scenarios, our preliminary results seemed extremely encouraging; in others, lackluster. 

We present here a small collection of results, to be taken with a nice pinch of salt, which highlight the intrinsic difficulty of a proper navigation of the Stability-Plasticity trade-off.

We experiment with our post-conditioning technique across two distinct scenarios:

- **Toy model: 2-layer MLP trained on MNIST**
    
    We train a tiny 2-layer MLP from scratch on two subsequent classification tasks. This lets us study how our method’s hyperparameters behave across settings when training from scratch.
    
- **LLMs: sequential SFT across 3 datasets**
    
    We fine-tune a Transformer-based Large Language Model with Supervised Fine-Tuning on three different datasets in sequence. We evaluate accuracy on all three datasets throughout training, and measure forgetting and learning metrics for all checkpoints. This setting lets us monitor how the technique scales and how it interacts with sparsity in large models.
    

In our experiments, we call *softmask* the matrix-wise post-conditioner, and *compsoft* its circuit-aware counterpart.

### Toy Model - MLP on MNIST

We train on subsets of MNIST, where each task is binary classification over two digits. The first task is classifying digits $1$ vs. $2$, and the second task is classifying digits $3$ vs. $4$. Once accuracy reaches $0.95$ on the first task, the second task is activated and the model is trained on the new digit pair. We train with a constant learning rate.

Here we report **AccΣ := Acc1 + Acc2** , representing the sum of the accuracies of the two tasks at the end of the training steps.

{% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/fig1.png" class="img-fluid" %}


### LLMs - Sequential SFT on Q&A Tasks

We perform Supervised Fine-Tuning on **Qwen3 0.6B** (<d-cite key=yang2025qwen3></d-cite>) and **Llama 3.2 1B** (<d-cite key=grattafiori2024llama></d-cite>) on a sequence of three distinct datasets. As training progresses, we evaluate the model’s capabilities on all tasks for every checkpoint. The resulting accuracies let us gauge how well training navigates the robustness–plasticity tradeoff: an ideal model learns strongly while retaining high accuracy on previous tasks.

We train sequentially on three Q&A datasets (inspired by <d-cite key=shenfeld2025rl></d-cite>):

- Open Reasoner-zero (<d-cite key=hu2025open></d-cite>)
- SciKnowEval (<d-cite key=feng2024sciknoweval></d-cite>)
- HellaSwag (<d-cite key=zellers2019hellaswag></d-cite>)

Here, we plot the validation accuracy of the model on each of the three datasets throughout the training, when using a fixed learning rate of $4 \times 10^{-5}$:


{% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/fig2a.png" class="img-fluid" %}

And the same with a learning rate of $1 \times 10^{-5}$:

{% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/fig2b.png" class="img-fluid" %}

It is crucial to notice how the post-conditioning method strongly outperforms Adam, when used with a higher learning rate, but it starts losing when the learning rate is tuned to a level that is optimal for the task.

We can get an idea of the model’s ability to navigate the stability-plasticity tradeoff by plotting the last task’s accuracy (plasticity) vs an average of the accuracies for the previous tasks (stability) at the end of the training.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/fig3a.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-dynamics-of-forgetting/fig3b.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

## Final Notes

We are publishing this blogpost with the main intent of conveying our point of view on continual learning: how it is a core distinguishing factor between human and artificial intelligence, how catastrophic forgetting plays a central role in the optimization procedure of a model, even outside of traditional sequential multi-task scenarios, and how we believe that forgetting is made of two very different dynamics, one of which is commonly addressed by most CL methods, the other still largely unexplored.     

We argue that modern continual learning methods cannot rely on knowing what knowledge must be retained, and should instead be capable of preserving existing structure in the weights. 

We test a couple early implementations in two distinct settings, where we find some mixed results, which we hope will stimulate discussion in the community and help us develop on this first step towards producing gracefully updating, continually learning models. Overall, we think the point stands: we must find a new way to build non-destructive updates that respect the model’s learned structure. We can’t wait to know what you think about it.


<!-- ## Appendix

### SoftMask post-conditioner
Let $W \in \mathbb{R}^{m\times n}$ be a weight matrix and let  $\Delta \in \mathbb{R}^{m\times n}$ be the **proposed** optimizer update. SoftMask returns a **conditioned** update $\widetilde{\Delta}$.

1. SVD of the current weights (cache every N steps, for computational reasons): 

    $$
    W = U S V^\top
    $$

    where $S=\mathrm{diag}(s_1,\dots,s_r), s_1\ge\dots\ge s_r\ge 0$, and $r=\min(m,n).$

2. Express the update in the spectral basis

    $$
    C := U^\top \Delta V \in \mathbb{R}^{r\times r},
    \qquad
    u_{ij} := |C_{ij}|
    $$

3. Choose quantiles $q_s,q_u \in (0,1)$. Define

    $$
    \zeta_s := \mathrm{Quantile}\{q_s\}\big(\{s_i\}{i=1}^r\big)
    $$

    $$
    \zeta_u := \mathrm{Quantile}\{q_u\}\big(\{|C{ij}|\}_{1\le i,j\le r}\big)
    $$

4. Soft gates: let $\sigma(x)=\frac{1}{1+e^{-x}}$. Choose softness $\beta>0$  and small  $\varepsilon>0$, then set

    $$
    \beta_s := \beta\,\zeta_s + \varepsilon,
    \qquad
    \beta_u := \beta\,\zeta_u + \varepsilon
    $$

    Define two selectors (the “allowed quadrants”):

    $$
    A_{ij} := \sigma\!\left(\frac{u_{ij}-\zeta_u}{\beta_u}\right)\;
    \sigma\!\left(\frac{\zeta_s-s_i}{\beta_s}\right)
    $$

    $$
    B_{ij} := \sigma\!\left(\frac{s_i-\zeta_s}{\beta_s}\right)\;
    \sigma\!\left(\frac{\zeta_u-u_{ij}}{\beta_u}\right)
    $$

    Combine them into a soft mask:

    $$
    M_{ij} := \min\!\Big(1,\max\!\big(0, A_{ij}+B_{ij}\big)\Big)
    $$

5. Apply the mask and reconstruct the conditioned update

    $$
    \widetilde{\Delta} := U\,(M \odot C)\,V^\top
    $$

    where $\odot$ is elementwise multiplication.

    As  $\beta \to 0$, the sigmoids sharpen and SoftMask approaches a corresponding hard (binary) masking rule.

### CompSoft post-conditioner

1. Project the update into the current spectral basis

    $$
    C := U^\top \Delta W V \in \mathbb{R}^{r\times r}
    $$

2. Alignment strengths via softmax aggregation. Define
    
    $$
    T_{\text{in}} := |U_{\text{prev}}^\top V| \in \mathbb{R}^{r_-\times r}.
    $$

    Aggregation is a softmax-weighted mean over the prev-axis
 -->
