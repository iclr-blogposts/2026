---
layout: distill
title: "Elastic Weight Consolidation (EWC): Nuts and Bolts"
description: "A theoretical deep-dive into the Elastic Weight Consolidation method for continual learning, explaining the mathematical foundations and intuitions behind this influential approach to preventing catastrophic forgetting."
authors:
  - name: Anonymous
date: 2026-04-27
future: true
htmlwidgets: true
bibliography: 2026-04-27-elastic-weight-consolidation-nuts-bolts.bib
toc:
    - name: Abstract
    - name: Introduction
    - name: Elastic Weight Consolidation
      subsections:
        - name: Intractability of posterior of $\mathcal{A}$ and its approximation
    - name: Conclusion
toc_sticky: true
categories:
  - Continual Learning
  - Regularization Methods
  - Theoretical Analysis
tags:
  - EWC
  - Catastrophic Forgetting
  - Fisher Information Matrix
  - Bayesian Methods
  - Laplace Approximation
---

## Abstract

In this blogpost, we present a theoretical support of the continual learning method **Elastic Weight Consolidation**, introduced in the paper titled 'Overcoming catastrophic forgetting in neural networks' <d-cite key="kirkpatrick2017overcoming"></d-cite>. Being one of the most cited papers in regularized methods for continual learning, this blogpost disentangles the underlying concept of the proposed objective function. We assume that the reader is aware of the basic terminologies of continual learning.

## Introduction

Following are the notations used throughout this blogpost. Vectors and matrices are denoted in bold lowercase and bold uppercase, respectively. Superscript $^{\top}$ denotes matrix transpose. $\mathbb{E}[\cdot]$ denotes the expectation operator. An optimum value of a variable is denoted by adding a superscript $^{\star}$.

Continual learning is a much desired attribute for neural networks. For example, if we train a model to distinguish between images of a cat and a dog (task 1), and subsequently train it again to distinguish between images of chair and table (task 2), the model should be able to retain its knowledge on task 1 even after learning task 2. In simple terms, our network model should be able to perform equally well on all seen tasks, even after learning new ones. Any degradation of performance on the previous tasks after learning new ones is fittingly termed as *catastrophic forgetting*. This sub-research area has seen an insurgence in works in recent times <d-cite key="kirkpatrick2017overcoming,zenke2017continual,li2017learning,aljundi2018memory"></d-cite>. Briefly, the continual learning scenarios can be categorized into following <d-cite key="van2019three"></d-cite>:

- **Task-Incremental Learning**: For the given set of tasks, the task identity is known during testing.
- **Domain-Incremental Learning**: For the given set of tasks, task identity is not provided during testing, but need not infer the same.
- **Class-Incremental Learning**: For the given set of tasks, task identity is not provided during testing, but has to infer the same.

We highly recommend <d-cite key="van2019three,wiewel2019localizing"></d-cite> for a good overview of different methodologies to alleviate catastrophic forgetting as well as continual learning in general. The next Section describes the well studied regularization method of continual learning: Elastic Weight Consolidation. It presents a solution to the continual learning problem by making task-specific synaptic (*read* network parameters) consolidation. Based on the theory of plasticity of post-synaptic dendritic spines in the brain, this method presents a paradigm that marks how important is a network parameter to the previous tasks and penalizes any change made to it depending upon the importance, while learning new tasks.

## Elastic Weight Consolidation

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-elastic-weight-consolidation-nuts-bolts/fig1.png" title="Possible configurations of θ*_A" class="img-fluid rounded z-depth-1" width="75%" %}
    </div>
</div>
<div class="caption">
    <strong>Possible configurations of</strong> $\boldsymbol{\theta}^\star_{\mathcal{A}}$. The shaded region represents a space of optimum $\boldsymbol{\theta}_{\mathcal{A}}$ with acceptable errors w.r.t. $\boldsymbol{\theta}^\star_{\mathcal{A}}$ for task $\mathcal{A}$.
</div>

Denote parameters of layers of a deep neural network (DNN) with $\boldsymbol{\theta}$. Training DNNs generates a mapping between the input distribution space and target distribution space. This is done by finding out an optimum 

\begin{equation}
\boldsymbol{\theta} = \boldsymbol{\theta}^\star
\end{equation}

which results in the least error in the training objective. It has been shown in earlier works <d-cite key="sussmann1992uniqueness"></d-cite> that such a mapping can be obtained with many configurations of $\boldsymbol{\theta}^\star$, represented in the figure above. The term *many configurations* can be interpreted as a solution space around the most optimum $\boldsymbol{\theta}$ with acceptable error in the learned mapping. Note that in figures to follow, the shaded ellipses represent the solution of individual tasks where as the overlapping region of multiple ellipses, marked by diagonal lines, represents the common solution space for all tasks.

Let's begin with a simple case of two tasks, task $\mathcal{A}$ and task $\mathcal{B}$. To have a configuration of parameters that performs well for both $\mathcal{A}$ and $\mathcal{B}$, the network should be able to pick $\boldsymbol{\theta}$ from the overlapping region of the individual solution spaces (see Figure 2). This is with the assumption that there is always an overlapping region for the solution spaces of all tasks for the network to learn them sequentially. A case of four tasks has been illustrated in Figure 2. In the first instance, the network can learn any 

\begin{equation}
\boldsymbol{\theta} = \boldsymbol{\theta}_{\mathcal{A}}
\end{equation}

that performs well for task $\mathcal{A}$. But with the arrival of task $\mathcal{B}$, the network should pick up a 

\begin{equation}
\boldsymbol{\theta} = \boldsymbol{\theta}_{\mathcal{A}, \mathcal{B}}
\end{equation}

The next question that arrives is how can the network learn the a set of parameters that lies in this overlapping region. To this end, EWC presents a method of selective regularization of parameters $\boldsymbol{\theta}$. After learning $\mathcal{A}$, this regularization method identifies which parameters are important for $\mathcal{A}$, and then penalizes any change made to the network parameters according to their importance while learning $\mathcal{B}$.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-elastic-weight-consolidation-nuts-bolts/fig2.png" title="Overlap of possible configurations of θ*" class="img-fluid rounded z-depth-1" width="75%" %}
    </div>
</div>
<div class="caption">
    <strong>Overlap of possible configurations of</strong> $\boldsymbol{\theta}^\star$. The overlapping space represents an optimum parameter region where the network performs without any catastrophic degradation on previous tasks.
</div>

### Intractability of posterior of $\mathcal{A}$ and its approximation

To formulate the objective, we start by taking a Bayesian approach needed to estimate the network parameters $\boldsymbol{\theta}$. More specifically given the data $\boldsymbol{\Sigma}$, we want to learn the posterior probability distribution function $p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma})$. Following <d-cite key="lherranz2018rotating"></d-cite> and using Bayes rule, we can write 

\begin{equation}
\underbrace{p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma})}_{\text{posterior}} = \dfrac{\overbrace{p(\boldsymbol{\Sigma}\mid\boldsymbol{\theta})}^{\text{likelihood}}\overbrace{p(\boldsymbol{\theta})}^{\text{prior}}}{p(\boldsymbol{\Sigma})}
\end{equation}

Since maximizing a function is same as maximizing its logarithm, we take $\log(\cdot)$ of the above equation as follows:

\begin{equation}
\log(p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma})) = \log(p(\boldsymbol{\Sigma}\mid\boldsymbol{\theta})) +\log(p(\boldsymbol{\theta})) - \log(p(\boldsymbol{\Sigma}))
\end{equation}

To train the neural network on $\boldsymbol{\Sigma}$, the objective function to be optimized over the log-likelihood function:

\begin{equation}
\text{argmax}_{\boldsymbol{\theta}}{\ell(\boldsymbol{\theta}) = \log(p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma}))}
\end{equation}

For the case of given two independent tasks such that 

\begin{equation}
\boldsymbol{\Sigma} = \{\mathcal{A}, \mathcal{B}\}
\end{equation}

(with $\mathcal{B}$ appearing in sequence after $\mathcal{A}$), the log-posterior can be written as:

\begin{equation}
\log(p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma})) = \log(p(\mathcal{B}\mid\boldsymbol{\theta})) +\log(p(\boldsymbol{\theta}\mid\mathcal{A})) - \log(p(\mathcal{B}))
\end{equation}
where the independence of $\mathcal{A}$ and $\mathcal{B}$ is used. Following the Bayesian formulation, $p(\mathcal{B}\mid\boldsymbol{\theta})$ is the loss for current task $\mathcal{B}$, $p(\mathcal{B})$ is the likelihood for $\mathcal{B}$, and now posterior $p(\boldsymbol{\theta}\mid\mathcal{A})$ for $\mathcal{A}$ becomes prior for $\mathcal{B}$.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-elastic-weight-consolidation-nuts-bolts/fig3.png" title="Laplace approximation of true posterior pdf" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Laplace approximation of true posterior pdf.</strong> $\mathbb{I}_\mathcal{A}$ represents the Fisher Information matrix.
</div>

Referring to the log-posterior equation, it can be observed that we have to deal with the function $p(\boldsymbol{\theta}\mid\mathcal{A})$. This is the posterior function for $\mathcal{A}$ which contains the information about the parameters that explain $\mathcal{A}$ using the given network. As discussed in <d-cite key="kirkpatrick2017overcoming"></d-cite>, this posterior function is said to be intractable. Basically, the intractability of $p(\boldsymbol{\theta}\mid\mathcal{A})$ can be interpreted as the function not existing in some interpretable form. Hence, it is difficult to estimate its quantiles. See <d-cite key="tokdar2013lecture"></d-cite> for an example.

Next as the posterior is difficult to analyze in its present form, we aim to approximate it using Laplace approximation. In simple terms, Laplace approximation methodology is employed to find a normal distribution approximation to a continuous probability density distribution (see Figure 3). Assuming $p(\boldsymbol{\theta}\mid\mathcal{A})$ is smooth and majorly peaked around its point of maxima (i.e. $\boldsymbol{\theta}^\star_{\mathcal{A}}$), we can approximate it with a normal distribution with mean $\boldsymbol{\theta}^\star_{\mathcal{A}}$ and variance $[\mathbb{I}_{\mathcal{A}}]^{-1}$. This brings us to the question on how did we come to the conclusion on these particular values of mean and variance for the normal distribution.

To begin with, compute the second order Taylor expansion of $\ell(\boldsymbol{\theta})$ around $\boldsymbol{\theta}^\star_{\mathcal{A}}$ as follows:

\begin{equation}
\ell(\boldsymbol{\theta})\approx \ell(\boldsymbol{\theta}^\star_{\mathcal{A}}) +( \dfrac{\partial\ell(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}}) + \dfrac{1}{2}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})^\top(\dfrac{\partial^2\ell(\boldsymbol{\theta})}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}}) + \text{(higher order terms)}
\end{equation}

Neglecting higher order terms and noting that 

\begin{equation}
\dfrac{\partial\ell(\boldsymbol{\theta})}{\partial\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}} = 0
\end{equation}

(slope of tangent at peak), we have:
$$
\begin{equation}
\ell(\boldsymbol{\theta})\approx \ell(\boldsymbol{\theta}^\star_{\mathcal{A}}) + \dfrac{1}{2}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})^\top\underbrace{(\dfrac{\partial^2\ell(\boldsymbol{\theta})}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})}_{\text{Hessian}}(\boldsymbol{\theta} -\boldsymbol{\theta}^\star_{\mathcal{A}})
\end{equation}
$$
Using the log-posterior equation, we can write the above for task $\mathcal{A}$ as following:

\begin{equation}
\log(p(\boldsymbol{\theta}\mid\mathcal{A})) = \log(p(\boldsymbol{\theta}^\star_{\mathcal{A}}\mid\mathcal{A})) + \dfrac{1}{2}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})^\top(\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}}) + \Delta
\end{equation}

where 

\begin{equation}
\Delta = \log(p(\boldsymbol{\theta}^\star_{\mathcal{A}}\mid\mathcal{A}))
\end{equation}

Next, write 

\begin{equation}
(\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}}) = -((-\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})^{-1})^{-1}
\end{equation}

and replace it back in the equation to express the same in the standard form of normal distribution function:

\begin{equation}
p(\boldsymbol{\theta}\mid\mathcal{A}) = \epsilon\exp(-\dfrac{1}{2}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})^\top((-\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})^{-1})^{-1}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}}))
\end{equation}

where 

\begin{equation}
\epsilon = \exp(\Delta)
\end{equation}

is a constant. From this equation, it can be concluded that we have obtained the Laplace approximation of posterior pdf as:


\begin{equation}
p(\boldsymbol{\theta}\mid\mathcal{A})\sim\mathcal{N}(\boldsymbol{ \theta}^\star_{\mathcal{A}}, (-\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})^{-1})
\end{equation}

Notice the variance of the estimated normal distribution of $p(\boldsymbol{\theta}\mid\mathcal{A})$. Given $\boldsymbol{ \theta}^\star_{\mathcal{A}}$, the term $\log(p(\boldsymbol{\theta}\mid\mathcal{A}))$ represents the log-likelihood of posterior pdf $p(\boldsymbol{\theta}\mid\mathcal{A})$. Clearly, the term represents the inverse of **Fisher information matrix** (FIM), 
$$
\begin{equation}
\mathbb{I}_{\mathcal{A}} = \mathbb{E}[-\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}}]
\end{equation}
$$

Note that we obtain $\mathbb{I}_{\mathcal{A}}$ by using the Bayesian equation and treating the prior $p(\boldsymbol{\theta})$ and $p(\mathcal{A})$ constant. This makes derivative of log of the Bayesian equation posterior and likelihood equal. More on this in **Appendix A.2** of <d-cite key="van2019three"></d-cite>. Finally, we get 

\begin{equation}
p(\boldsymbol{\theta}\mid\mathcal{A})\sim\mathcal{N}(\boldsymbol{ \theta}^\star_{\mathcal{A}}, [\mathbb{I}_{\mathcal{A}}]^{-1})
\end{equation}

Further, as FIM can also be computed from first order derivatives, we can avoid the Hessian computed in the Taylor expansion using the following property <d-cite key="kay1993fundamentals"></d-cite>:$$
\begin{equation}
\mathbb{I}_{\mathcal{A}} = \mathbb{E}[-\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}}] = \mathbb{E}[(\dfrac{\partial(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial\boldsymbol{\theta}})(\dfrac{\partial(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial\boldsymbol{\theta}})^\top\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}}]
\end{equation}
$$
Now, we can write the log-posterior equation as:
$$
\begin{equation}
\log(p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma})) = \log(p(\mathcal{B}\mid\boldsymbol{\theta})) +\dfrac{\lambda}{2}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})^\top(\dfrac{\partial^2(\log(p(\boldsymbol{\theta}\mid\mathcal{A})))}{\partial^2\boldsymbol{\theta}}\mid_{\boldsymbol{\theta}^\star_{\mathcal{A}}})(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}}) + \epsilon'
\end{equation}
$$
where $\epsilon'$ accounts for all constants and $\lambda$ is a hyper-parameter introduced to have a trade off between learning $\mathcal{B}$ and not forgetting $\mathcal{A}$. Simplifying more, we have:
$$
\begin{equation}
\log(p(\boldsymbol{\theta}\mid\boldsymbol{\Sigma})) = \log(p(\mathcal{B}\mid\boldsymbol{\theta})) - \dfrac{\lambda}{2}(\boldsymbol{\theta} - \boldsymbol{ \theta}^\star_{\mathcal{A}})^\top \mathbb{I}_{\mathcal{A}} (\boldsymbol{\theta} -  \boldsymbol{\theta}^\star_{\mathcal{A}}) + \epsilon'
\end{equation}
$$

This implies:
$$
\begin{equation}
\underbrace{\ell(\boldsymbol{\theta})}_{\text{overall loss}} = \underbrace{\ell_\mathcal{B}(\boldsymbol{\theta})}_{\text{loss for \mathcal{B} }} - \underbrace{\dfrac{\lambda}{2}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})^\top\mathbb{I}_{\mathcal{A}}(\boldsymbol{\theta} -\boldsymbol{ \theta}^\star_{\mathcal{A}})}_{\text{weight regularizer}} + \epsilon'
\end{equation}
$$

Further simplification can be found in <d-cite key="van2019three"></d-cite>. Before we end this Section, let's discuss how does the FIM indicates the **importance** of the parameters for the previous tasks.

We say a network has learnt a task when its objective has reached a minimum in the loss surface. We know that the curvature of such surfaces represent the sensitivity of the network with respect to the optimum $\boldsymbol{\theta}^\star$. This sensitivity can be determined by looking at the direction along which $\boldsymbol{\theta}^\star$ changes. This implies the curvature is inversely proportional to change in $\boldsymbol{\theta}^\star$. Hence, if the more the curvature, a '$\delta$' increment can result in large increase in the loss. Curvature of a curve is denoted by its Hessian and hence in our case, as the second derivative is of the log likelihood function of the posterior pdf, the FIM $$\mathbb{I}_{\mathcal{A}}$$ comes into picture. Thus, $$\mathbb{I}_\mathcal{A}$$ can tell us which parameter is important to the the previous task as its corresponding element in $$\mathbb{I}_\mathcal{A}$$ will have a large value, indicating higher importance. See <d-cite key="maltoni2019continuous"></d-cite> for more.

## Conclusion

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-elastic-weight-consolidation-nuts-bolts/fig4.png" title="Sequential training on task B after task A" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    <strong>Sequential training on task</strong> $\mathcal{B}$ <strong>after task</strong> $\mathcal{A}$. Left: Train the network as it is: results in 'Forgetting', Middle: Make no change in the parameters of previous tasks, Right: Make changes in the parameters of the previous tasks depending on their importance.
</div>

In this blogpost, we have presented a theoretical support of the EWC method. We have shown how the intractable posterior function can be approximated using Laplace approximation, and how the Fisher Information Matrix can be used to identify the importance of parameters for previous tasks. The EWC method provides a principled approach to continual learning by selectively regularizing parameters based on their importance to previously learned tasks.
