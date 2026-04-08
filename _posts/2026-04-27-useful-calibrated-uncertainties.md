---
layout: distill
title: What (and What Not) are Calibrated Probabilities Actually Useful for?
description: This blogpost clarifies the practical usefulness of having a model with calibrated probabilities, something that is not often clearly stated in the calibration literature. We show that a calibrated model can be relied on to estimate average loss/reward, however, good calibration does not mean that a model is useful for per-sample decision making. 
date: 2026-04-27
future: true
htmlwidgets: true

# camera-ready author list
authors:
  - name: "Guoxuan Xia"
    affiliations:
      name: "Department of Electrical and Electronic Engineering, Imperial College London"

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Author Name
#     url: "https://example.com"
#     affiliations:
#       name: Institution Name

# must be the exact same name as your blogpost
bibliography: 2026-04-27-useful-calibrated-uncertainties.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: People Have a Fuzzy Impression of Calibration (in AI)
  - name: A General Explanation of Calibration
    subsections:
      - name: Confidence Calibration
  - name: What Do Calibrated Probabilities Allow You to Do?
    subsections:
      - name: Interactive Example
  - name: What Calibrated Probabilities DO NOT Guarantee
    subsections:
      - name: Interactive Example Continued
  - name: Misunderstandings from the Literature
    subsections:
      - name: A Brief Retrospective
  - name: Closing Thoughts and Takeaways
---
<style>
  .takeaway-box {
    margin: 1.4rem 0 1.6rem 0;
    padding: 1rem 1.1rem 1rem 1.1rem;
    border-left: 5px solid #2c7a7b;
    border-radius: 0 10px 10px 0;
    background: linear-gradient(135deg, rgba(44, 122, 123, 0.1), rgba(44, 122, 123, 0.03));
  }

  .takeaway-box .takeaway-title {
    margin: 0 0 0.45rem 0;
    font-size: 1.4rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    font-weight: 700;
  }

  .takeaway-box p {
    margin: 0 0 0.7rem 0;
    font-size: 1.2rem;
    line-height: 1.55;
  }

  .takeaway-box p:last-child {
    margin-bottom: 0;
  }

  .highlight-box {
    margin: 1.2rem 0 1.4rem 0;
    padding: 0.9rem 1.1rem;
    border-left: 4px solid #2c7a7b;
    border-radius: 0 10px 10px 0;
    background: linear-gradient(135deg, rgba(44, 122, 123, 0.1), rgba(44, 122, 123, 0.03));
  }

  .highlight-box p,
  .highlight-box ol {
    margin-top: 0;
    margin-bottom: 0;
  }

  .equation-box {
    margin: 1.2rem 0 1.5rem 0;
    padding: 0.8rem 1rem 0.35rem 0;
    border: 1px solid rgba(70, 70, 90, 0.25);
    border-radius: 10px;
    background: rgba(140, 140, 180, 0.06);
  }

  .equation-box p {
    margin-top: 0;
  }

  .equation-box .katex-display,
  .equation-box mjx-container[display="true"],
  .equation-box d-math {
    margin-left: 0 !important;
    padding-right: 1rem;
    box-sizing: border-box;
  }
</style>

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/confused.png" class="img-fluid z-depth-1" %}
## People Have a Fuzzy Impression of Calibration (in AI)

The goal of this blogpost is to provide an intuitive and helpful guide on understanding the *practical usefulness* of a *well-calibrated* model, for those both familiar and unfamiliar with calibration. Anecdotally, when I tell researchers and practitioners from other domains that I work in uncertainty estimation, they often give remarks along the lines of "Oh so like calibration?" or "You mean probabilistic and Bayesian stuff?". They seem to know of the research domain and associate it with vague motivations of "safety", "trustworthiness" and "reliability", but often have an unclear sense of what it actually entails. Even amongst uncertainty researchers, people often have fuzzy impressions of the motivation for calibration. Papers on the subject are often written with one or two motivating paragraphs, before promptly moving onto the main algorithmic meat. Reading the literature thus becomes a way to learn about *how to better calibrate models* without really engaging concretely with *how, why and when calibrated models are useful*. 

Many motivating paragraphs in papers, in order to communicate with the reader more intuitively, end up being imprecise and can be easily misinterpreted. Below I've included an excerpt from the introduction of *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite> which is the go-to introductory paper for calibration in deep learning. It is broadly representative of how calibration is typically motivated in AI research recently, and we will unpick it gradually over the course of this blogpost.
<div id="excerpt"></div>
> In real-world decision making systems, classification networks must not only be accurate, but also should indicate when they are likely to be incorrect. As an example, consider a self-driving car that uses a neural network to detect pedestrians and other obstructions. If the detection network is not able to confidently predict the presence or absence of immediate obstructions, the car should rely more on the output of other sensors for braking. Alternatively, in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low. Specifically, a network should provide a calibrated confidence measure in addition to its prediction. In other words, the probability associated with the predicted class label should reflect its ground truth correctness likelihood. 

A reader is likely to come away from this paragraph thinking that calibration is important for *mitigating risk* from errors during *individual scenarios* (i.e. for a specific road object or individual patient). The reality is more nuanced than this; calibration is actually somewhat orthogonal to a model's ability to detect its own errors. As such, it is easy to come to misunderstandings over calibration after reading passages like the above in the literature. After reading this blogpost, hopefully the reader will have a clearer understanding and intuition of what a calibrated model actually is important for, and in what use cases calibration is actually insufficient. 

<div class="highlight-box">
<p><strong>In this blogpost we will:</strong></p>
<ol>
  <li>Provide a clear and neutral explanation of calibration.</li>
  <li>Demonstrate, with examples and interactive widgets, how calibration allows a model to be relied upon to estimate <em>expected</em> loss/reward without ground truth labels, thereby enabling the setting of decision rules that depend on said loss/reward.</li>
  <li>Show that calibration does not guarantee performance for decision making on <em>individual</em> samples; instead what matters is the model's discrimination ability.</li>
  <li>Review how the calibration literature has, over time, blurred the understanding of the above.</li>
</ol>
</div>

## A General Explanation of Calibration

Calibration, as it is defined in deep learning, is not an intuitive concept to understand. Thus, paper authors tend to first motivate it using imprecise, but more intuitive natural language, like in the example above. We will instead start with a clear and general explanation of calibration to the reader as a basis for the rest of the blogpost.

For a model to be calibrated, when it predicts a certain probability distribution over different outcomes, the real-world frequency of said outcomes should match that probability distribution.<d-footnote>We will focus on distributions over discrete outcomes in this blogpost as it is the most common setting for calibration.</d-footnote> A widely used example is weather forecasting: over a long run of days, it should rain on 70% of the days which have 70% chance of rain forecasted. Explicitly, in natural language, a calibrated model is one where,

$$
\begin{array}{l}
\text{empirical frequency of outcomes when } \\
\text{probabilities for those outcomes are predicted}
\end{array}
= \text{ those predicted probabilities},
$$

or for a **binary** $\text{event}$ (e.g. rain or no rain), a simpler version is

<div class="equation-box">
$$
\begin{array}{l}
\text{empirical frequency of event occurring when } \\
\text{a given probability of event is predicted} 
\end{array}
= 
\begin{array}{l}
\text{ that predicted}\\
\text{ probability}
\end{array}
\tag{1}\label{eq:calib-natlang}
$$
</div>
which can be expressed mathematically as,

<div class="equation-box">
$$
\underbrace{P(\text{event} \mid  \pi) = \pi}_\text{actual probability of event is $\pi$},\quad \text{for }\underbrace{\pi={P}_{\theta}(\text{event}\mid \text{conditions}) }_\text{when model predicts probability $\pi$}\text{ in } [0,1], \tag{2}\label{eq:calib-math}
$$
</div>

where $\pi$ is a random variable that captures the value of the probability output by our model with parameters $\theta$. Note that $\pi(\text{event},\text{conditions})$ is a function of $\text{event}$ and $\text{conditions}$. If $\text{event}$ is "rain" then $\text{conditions}$ could be "temperature of the previous day", for example (i.e. the model input). We will be relying on Eq. \ref{eq:calib-math} a fair amount in this blogpost, however, it is simply formalising the intuition in Eq. \ref{eq:calib-natlang}.

In this blogpost we will focus on binary events; however, the intuitions/ideas naturally extend to the general case. We leave the general version here for reference:

$$
\begin{align*}
&P(\text{outcome}_k \mid  \boldsymbol\pi  ) = \pi_k,\quad \forall k, \\
&\pi_k={P}_{\theta}(\text{outcome}_k\mid \text{conditions}), \quad \boldsymbol\pi\text{ is a probability vector}.
\end{align*}
$$

Importantly, $P(\text{event} \mid  \pi  )$ is **not** given $\text{conditions}$. It is the probability of the $\text{event}$ *on average* over the distribution of possible $\text{conditions}$,
<div class="equation-box">
$$
P(\text{event} \mid  \pi)
= \sum_{\text{conditions}}
P(\text{event} \mid  \text{conditions})\,
P\bigl(\text{conditions} \mid  \pi\bigr).
\tag{3}\label{eq:calib-math-avg}
$$
</div>
**Calibration is concerned with average behaviour over many inputs with the same predicted probability $\pi$, and not with any single individual prediction.**

In the binary case, how well calibrated a model is can be visualised using a reliability diagram <d-cite key="degroot1983comparison,NiculescuMizilCaruana2005"></d-cite> which plots the empirical frequency of $\text{event}$ (on average over $\text{conditions}$) given the predicted model probability $\pi$, i.e. visually comparing the left and right hand sides of Eq. \ref{eq:calib-natlang}. Practically, by binning predicted probabilities $\pi$, $P(\text{event} \mid  \pi  )$ can be estimated using the empirical frequency of $\text{event}$ within each bin.

<div id="reliability-diagram-figure"></div>
{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/reliability_diagrams.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>Left: Reliability diagram showing over and under-confident models. Right: Practical reliability diagram where empirical frequencies are estimated using bins.</strong>
</div>

A well-calibrated model's reliability diagram will align with $y=x$ (Eq. \ref{eq:calib-natlang} is satisfied). Deviation below indicates "overconfidence" (real frequency < predicted probability); deviation above indicates "underconfidence" (real frequency > predicted probability). Note that the notion of over/under-confidence is much more specific here than in colloquial usage; we will return to this point later. Also, a model can be overconfident for a certain range of probabilities and underconfident on a different range (i.e. both at the same time), however, for simplicity's sake we'll limit discussion in this blog to models that are either over- or under-confident. The results naturally extend to more general cases.
### Confidence Calibration
The above general presentation of calibration doesn't immediately connect with the idea of mitigating risk in safety-critical scenarios that is present in our [quoted excerpt](#excerpt) from *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite>, as well as generally throughout the literature. In fact, in *On Calibration of Modern Neural Networks*, Guo et al. choose a binary $\text{event}$ **"model prediction is correct"** from the scenario of multi-class classification, which then inherently ties calibration to prediction errors and their associated risks/costs. This specific case of calibration is known as *confidence calibration* and is the primary form that is found in the deep learning literature <d-cite key="Guo2017Calibration,Minderer2021Revisiting,Zhang2020MixnMatchCalibration,param_temp_scaling_eccv2022,Xiong2023ProCal,LeCoz2024EfficientCalibration"></d-cite>. In this case, Eq. \ref{eq:calib-natlang} becomes:
<div class="equation-box">
$$
\begin{array}{l}
\text{classification accuracy when a given } \\
\text{probability is predicted for the top class}
\end{array}
= 
\begin{array}{l}
\text{ that predicted}\\
\text{ probability}
\end{array}
\tag{4}\label{eq:conf-calib-natlang}
$$
</div>
or mathematically for true label $y$, input $x$ ($\text{conditions}$) and predicted label (top class) $\hat y=f_\phi(x)$ of classifier $f_\phi$ with parameters $\phi$,
<!-- =\arg \max_\omega P_{\theta}(\omega\mid x)$, -->
<div class="equation-box">
$$
\underbrace{P\bigl(y=\hat y \mid  \pi\bigr) = \pi}_\text{actual classification accuracy is $\pi$},\quad \text{for }\underbrace{\pi=P_{\theta}(y=\hat y\mid x) }_\text{predicts probability $\pi$} \text{ in } [0,1]. \tag{5}\label{eq:conf-calib-math}
$$
</div>
In words: amongst all test points where the model says "the probability of the predicted class label is 80%", the prediction should actually be right 80% of the time.

One nuance the reader should be aware of is that for the purposes of confidence calibration, the model for the binary $\text{event}$ "model prediction is correct" $P_\theta(y=\hat y\mid x)$ is conceptually separate from the underlying classifier $f_\phi(x)$, and does not necessarily have to share parameters. However, when the classifier is a cross-entropy-trained softmax model (i.e. the vast majority of classifiers in deep learning), ${P}_{\theta}(y=\hat y\mid x)$ can be, and almost always is, extracted from the probability vector over classes output by the classifier <d-cite key="LeCoz2024EfficientCalibration"></d-cite>. In this case,

$$
P_{\theta}(y=\hat y\mid x) =  \text{softmax}\bigl(\text{logits}_\phi(x)\bigr)_\hat{y}~,
$$

$$
\hat y = \arg\max_k \text{logits}_\phi(x)_k~.
$$

Thus, for this scenario, $P_\theta(y=\hat y\mid x)$ can be viewed as the confidence of the classifier $f_\phi$ in its own prediction, hence the name *confidence calibration*.

For further discussion of other realisations of calibration (e.g. with different definitions of the binary $\text{event}$), as well as more specifics on how calibration is measured in practice (e.g *Expected Calibration Error* <d-cite key="Naeini2015BBQ"></d-cite>) I recommend this [excellent blogpost](https://iclr-blogposts.github.io/2025/blog/calibration/) by Maja Pavlovic <d-cite key="pavlovic2025understanding"></d-cite>.
## What Do Calibrated Probabilities Allow You to Do?

Now that we've understood what a calibrated model is in an abstract sense, we can start to understand what practical benefit it provides. Consider the calibration equation (Eq. \ref{eq:calib-math}) again,

$$
P(e \mid  \pi) = \pi, \quad \text{for }\pi={P}_{\theta}(e\mid c) \in [0,1].
$$

where we've abbreviated $\text{event}$ and $\text{conditions}$ to random variables $e$ and $c$ respectively. Intuitively, this means that for the predictions with $\pi$ equal to some value, say $0.7$, we can reliably expect $\text{event}$ to occur $70\%$ of the time. Consider a loss (or reward) function $\mathcal{L}(e,\pi)$ that depends on $\text{event}$ and the model's predicted probability. For example, in the case of confidence calibration, the 0-1 multiclass classification loss is 0 when $\text{event}$ occurs (correct prediction) and 1 when $\text{event}$ doesn't occur (misclassification). Intuitively, we can now rely on $\pi$ to calculate the *expected/average loss*. 

Consider confidence calibration where for some image classifier our binary model is predicting $\pi=0.6$ probability of being correct for half of the input images $x$ and $\pi=0.4$ for the other half. If our model is well calibrated we can then reliably claim that it will have an average accuracy of $60\%$ on the first half and $40\%$ on the second half, leading to an overall accuracy of $50\%$, *without needing to compare its predictions to any ground truth labels $y$*. This is the sense in which a calibrated model is *reliable*.

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/calibration_illust.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
Illustration of how a confidence-calibrated model allows estimation of expected accuracy from only inputs/conditions without needing events/labels.</strong>
</div>

Once we have a reliable estimate of the expected loss/reward we can make downstream decisions, e.g. for a certain deployment scenario the image classifier is not accurate enough and will lead to profit loss from misclassifications, so we decide not to deploy it. Additionally, humans can often reason intuitively about probabilistic information <d-cite key="CosmidesTooby1996IntuitiveStat"></d-cite>,<d-footnote>Imagine you are told a weighted coin returns heads 80% of the time. It's easy to visualise and reason about.</d-footnote> so human decision making can be performed naturally and reliably if model outputs are calibrated probabilities.

Finally, we note that in order to calibrate an uncalibrated model, or indeed to measure the calibration of the model in the first place, we need access to ground truth $\text{event}$ data <d-cite key="Guo2017Calibration,NiculescuMizilCaruana2005"></d-cite>. However, in problem scenarios where access to data at *deployment* time is different to *development* time, e.g. due to privacy or cost constraints, it can be important at deployment to have a model that was calibrated during development.<d-footnote>Incidentally this raises the challenging open problem of maintaining calibration under deployment time distribution shift <d-cite key="Ovadia2019Shift"></d-cite>.</d-footnote>

<div class="takeaway-box" id="key-takeaway-1">
<p class="takeaway-title">Key Takeaway 1</p>
<p><em>A well-calibrated model enables reliable estimation of average loss/reward from only inputs/conditions without any ground truth events/labels. These reliable estimates can then be used to effectively inform downstream decisions that depend on average behaviour.</em></p>
</div>


<!-- Now let's show this mathematically (the reader is free to skip to the interactive widget in the next section if they are happy with the above intuition). The expected loss of the model over the distribution of $\text{conditions}$ and $\text{event}$ (i.e. the data) is,

$$
\begin{align*}
\mathbb{E}[\mathcal{L}(e,\underbrace{\pi(e,c)}_{\pi={P}_{\theta}(e\mid c)})]
&= \sum_{c,e}  \mathcal{L}(e,\pi(e,c)) P(e,c) \\
&= \sum_{c,e,\pi}  \mathcal{L}(e,\pi) P(e ,\pi,c) \\
&= \sum_{e,\pi}  \mathcal{L}(e,\pi) P(e \mid \pi) P(\pi)\\
&=  \sum_{e,\pi} \mathcal{L}(e,\pi) P(e \mid \pi) \sum_{c} P(\pi, c) \\
&=  \sum_{e,\pi} \mathcal{L}(e,\pi) P(e \mid \pi) \sum_{c} \underbrace{\delta(\pi(e,c)) P(c)}_\text{Dirac $\delta$ for $\pi={P}_{\theta}(e\mid c)$}. \\
&=  \sum_{e} \mathcal{L}(e,\pi) P(e \mid \pi(e,c)) \sum_{c}P(c)\\
\end{align*}
$$

If the model is well calibrated we can substitute $\pi$ in for $P(e\mid \pi)$ and then approximate the expected loss by sampling model predicted probabilities $\{\pi^{(n)}={P}_{\theta}(e=1\mid c^{(n)})\}_n$ over $N$ samples of $\text{conditions}$ drawn from the data distribution $c\sim P(c)$,

$$
\begin{align*}
\mathbb{E}[\mathcal{L}(e,\pi(e,c))]&=  \sum_{e,\pi} \mathcal{L}(e,\pi) P(e \mid \pi) \sum_{c} \delta(\pi(e,c)) P(c) \\
&=  \underbrace{\sum_{e} \mathcal{L}(e,\pi) \sum_{c}\pi(e,c) P(c)}_{\text{no sum over }\pi\text{ as it is a fixed function of }e,c} \\
&= \underbrace{\sum_{c} P(c) \sum_{e} \mathcal{L}(e,\pi) \pi(e,c)}_{\text{no need for }P(e,c)\text{ only the distribution }P(c)}\\
&\approx \frac{1}{N} \sum_{n} \mathcal{L}(e=1,\pi^{(n)}) \pi^{(n)} +\mathcal{L}(e=0,\pi^{(n)}) (1-\pi^{(n)})  \tag{6}\label{eq:expected-loss-est}
\end{align*}
$$ -->
Now let's show this mathematically. If the intuition is already clear, feel free to [skip to the interactive widget in the next section](#interactive-example). Nothing later in the blog relies on following this derivation step-by-step. The expected loss of the model over the joint distribution of $\text{conditions}$ and $\text{event}$ (i.e. the data) is

<!-- $$\begin{align*}
\mathbb{E}[\mathcal{L}(e,\underbrace{\pi(e,c)}_{\pi = P_{\theta}(e\mid c)})] &= \sum_{c,e} \mathcal{L}(e,\pi(e,c))\,P(e,c) \\
&= \sum_{c} P(c)\sum_{e} \mathcal{L}(e,\pi(e,c))\,\underbrace{P(e\mid c)}_{\substack{\text{ground-truth}\\\text{conditional}}}.
\end{align*}$$

We can also rewrite the expected loss as
$$\begin{align*}
\mathbb{E}[\mathcal{L}(e,\pi(e,c))]
&= \sum_{\pi} \underbrace{P(\pi)}_{\substack{\text{marginal over}\\\text{predicted}\\\text{probabilities}}}\sum_{e} \mathcal{L}(e,\pi)\,\underbrace{P(e\mid \pi)}_{\substack{\text{frequency of}\\\text{events among}\\\text{examples with }\pi}}.
\end{align*}$$ -->

$$
\begin{align*}
\mathbb{E}[\mathcal{L}(e,\underbrace{\pi(e,c)}_{\pi={P}_{\theta}(e\mid c)})]
&= \sum_{c,e}  \mathcal{L}(e,\pi(e,c)) P(e,c) \\
&= \sum_{c,e,\pi}  \mathcal{L}(e,\pi) P(e ,\pi,c) \\
&= \sum_{e,\pi}  \mathcal{L}(e,\pi) P(e \mid \pi) P(\pi)\\
&=  \sum_{e,\pi} \mathcal{L}(e,\pi) P(e \mid \pi) \sum_{c} P(\pi, c) \\
&=  \sum_c P(c)\sum_{\pi}P(\pi\mid c) \sum_e \mathcal{L}(e,\pi) P(e \mid \pi) \\
\end{align*}
$$

For the binary case $e\in\{0,1\}$ we now make a minor abuse of notation and write

$$
\pi(c) := P_{\theta}(e=1\mid c)\in[0,1],$$
so that the full predicted distribution over $e$ is identified by the single scalar $\pi(c)$: $P_{\theta}(e=1\mid c)=\pi(c)$ and $P_{\theta}(e=0\mid c)=1-\pi(c)$.

If the model is *well calibrated*, then for any $\pi\in[0,1]$ we have

$$
P(e=1\mid \pi) = \pi,\qquad P(e=0\mid \pi) = 1-\pi,
$$

so the sum over $e$ simplifies to

$$
\sum_{e} \mathcal{L}(e,\pi)\,P(e\mid \pi) = \mathcal{L}(1,\pi)\,\pi + \mathcal{L}(0,\pi)\,(1-\pi).
$$

Moreover, since $\pi$ is a deterministic function of $c$, the conditional $P(\pi\mid c)$ puts all its mass on the single value $\pi(c)$. Thus, we can replace the sums over $e$ and $\pi$,

$$
\begin{align*}
\mathbb{E}[\mathcal{L}(e,\pi(e,c))]
&=  \sum_c P(c)\sum_{\pi}P(\pi\mid c) \sum_e \mathcal{L}(e,\pi) P(e \mid \pi) \\
&= \sum_{c} P(c)\sum_{\pi} \underbrace{P(\pi\mid c)}_{\substack{\text{degenerate since}\\\pi=\pi(c)}}\bigl[\mathcal{L}(1,\pi)\,\pi + \mathcal{L}(0,\pi)\,(1-\pi)\bigr] \\
&= \underbrace{\sum_{c} P(c)\bigl[\mathcal{L}(1,\pi(c))\,\pi(c) + \mathcal{L}(0,\pi(c))(1-\pi(c))\bigr]}_{\substack{\text{now only an expectation over conditions }c}}.
\end{align*}
$$

In practice we approximate this expectation using $N$ unlabeled samples $c^{(n)}\sim P(c)$ and their predicted probabilities $\pi^{(n)} = P_{\theta}(e=1\mid c^{(n)})$:

$$
\mathbb{E}[\mathcal{L}(e,\pi(e,c))] \approx \frac{1}{N}\sum_{n=1}^{N}\bigl[\mathcal{L}(1,\pi^{(n)})\,\pi^{(n)} + \mathcal{L}(0,\pi^{(n)})(1-\pi^{(n)})\bigr]. \tag{6}\label{eq:expected-loss-est}
$$

That is to say, under calibration, we can compute expected loss or reward using only the distribution of model-predicted probabilities $\pi$ on unlabeled inputs $c$.

And so, we have mathematically demonstrated [Key Takeaway 1](#key-takeaway-1).

### Interactive Example

To make this concrete, let’s look at a simple but realistic decision problem where average loss/reward is important. A bank wants to decide on a decision threshold for issuing loans based on predicted default probabilities. The problem setup is as follows:
- A bank wants to issue loans to customers based on their predicted probability of default (failure to pay the loan back).
- On average 35% of customers default on their loans.
- The bank incurs a loss of 240 dollars for each defaulted loan and gains 18 dollars for each successfully repaid loan.
- The bank uses a machine learning model to predict the probability of default for each customer.
- If the predicted probability of default is above threshold $\tau$ the bank decides not to lend (no loss or profit for that customer).
- The bank wants to choose the threshold $\tau$ that maximises profit, but the responsible department doesn't have access to default event data and so relies on the model probabilities assuming they are calibrated.
- Given calibrated predicted probability of default $\pi$, the expected profit from issuing a loan is $18 \times (1-\pi) - 240 \times \pi$.

Explore $\tau$ by varying it to maximise the estimated profit. You should observe that by optimising $\tau$ based on the calibrated model probabilities, the bank can achieve near-optimal profit even without access to default $\text{event}$ data, i.e. the model can be trusted. Now toggle to the uncalibrated models. The average profit estimates become unreliable, leading to choices of $\tau$ that squander profit or even incur losses on the actual customers. The model that is overconfident about the probability of defaulting leads to overly pessimistic estimates of profit and underlending, whilst the underconfident model is overly optimistic and leads to overlending. 

Note that intuitively, the optimal threshold is on the boundary where expected profit given $\pi$ turns to zero for calibrated $\pi$, i.e. reject loans for all $\pi$ that lose money in expectation. Interestingly, this decision rule can be in fact set directly, without any $\text{conditions}$ or $\text{event}$ data, if the model is well calibrated, since the aforementioned boundary value of $\pi$ is independent of the distribution over $\text{conditions}$.

<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-useful-calibrated-uncertainties/loan_widget.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="500px"
    width="100%"
  ></iframe>
</div>

## What Calibrated Probabilities DO NOT Guarantee
Recall the motivating paragraph from *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite>, one part of which states:

> ... in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low. 

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/selective_classification.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
Illustration of automated medical diagnosis with selective classification where a prediction should be deferred to an expert if it is likely to be incorrect.
</strong>
</div>

The task the authors are describing here is selective classification <d-cite key="Geifman2017SelectiveClassification,Jaeger2023FailureDetection,Xia2024SelectiveOOD"></d-cite> (or classification with abstention). In selective classification, if a classifier is likely to make an error on a specific input, then a useful uncertainty estimate should reflect this by indicating low confidence, triggering abstention, e.g. deferring a diagnosis to a human doctor. Here, **per-sample decisions** are made based on the model $P_\theta(\text{event}\mid \text{conditions})$, and we want it to be able to separate inputs where $\text{event}$ occurs (correct prediction) from inputs where $\text{no event}$ (error) occurs, i.e. **good discrimination**. This is an intuitively desirable property for uncertainty estimates, however, **a well-calibrated model is not one that is necessarily better for per-sample decision making**, in tasks such as selective classification. 

Consider again our previously discussed well-calibrated image classifier. Here we are focusing on the model of $P_\theta(\text{event of correct prediction}\mid \text{image})$ rather than the multi-class classifier $f_\phi(x)$ itself, which we assume has a fixed overall accuracy of $50\%$. If we set abstention threshold $\tau = 0.5$, then we would abstain on the images with accuracy $40\%$ and boost the accuracy on the selected images to $60\%$. 

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/calibration_discrimination.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
Illustration of how abstention can improve accuracy by rejecting uncertain predictions.
</strong>
</div>

However, if we consider another well-calibrated model that predicts the average accuracy $\pi=0.5$ for *all* images, then all predictions have tied confidence and we have no way to discriminate between correct and incorrect predictions, even though for selective classification we would like our model to be more uncertain on errors. **The model is well-calibrated but its uncertainties are *useless* for abstaining on potential errors!** The key here is that *calibration is related to the accuracy* ***averaged*** *over different inputs* $x$, and does not interrogate the model for each input individually. It only examines $\pi$ with respect to true probability $P(\text{event}\mid \pi)$, not with respect to the per-sample true probability $P(\text{event}\mid \text{conditions})$ (Eq. \ref{eq:calib-math-avg}).


{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/cal_bad_disc.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
Illustration of how a well-calibrated model with no discrimination ability cannot improve accuracy via abstention.
</strong>
</div>

Conversely, if a different well-calibrated model for $\text{event}$ "correct prediction" is able to predict $\pi=0.8$ for a subset of half of the images where the classifier has accuracy $80\%$ (different to previous subsets) and $\pi=0.2$ for the other half where accuracy is $20\%$, then the selected images will have an accuracy of $80\%$ when setting $\tau=0.5$. **This model makes better per-sample decisions as it is better at discriminating between correct and incorrect events.**

**You can think of calibration and discrimination as two orthogonal axes**: calibration controls the relationship between predicted probabilities and average frequencies, whilst discrimination controls how well the model is able to *rank* individual inputs by their true likelihood of $\text{event}$ occurring.

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/cal_better_disc.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
Illustration of how a well-calibrated model with better discrimination ability can improve accuracy via abstention.
</strong>
</div>
 
Finally, if we consider an uncalibrated (overconfident) model that predicts $\pi=0.9$ for the same subset of size 50% as above where the classifier has accuracy $80\%$ and $\pi=0.7$ for the other half where accuracy is $20\%$, then the selected images will have an accuracy of $80\%$ when setting $\tau=0.8$. In this case, despite being *uncalibrated*, the model is still able to effectively discriminate between correct and incorrect predictions and abstain on samples it is more likely to be incorrect on. 

However, in order to estimate the accuracy of selected images and then reliably choose a threshold $\tau$ for deployment, we would need access to ground truth labels from a validation set. If we wanted 80% accuracy on selected images and relied on the uncalibrated model's $\pi$, rather than validation labels, we would have selected all images and ended up with an accuracy of only 50%. If you don't trust the calibration of your model, the only way to reliably determine its performance is to validate it against ground truth labels.
 
 
{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/uncal_better_disc.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
Illustration of how an uncalibrated model with good discrimination ability can improve accuracy via abstention, if there is access to validation labels to choose the threshold and validate the accuracy. 
</strong>
</div>

We note that the above example is purposely reductive as generally speaking $\pi$ will take many possible values rather than just two, and that downstream decisions may involve many possible actions. However, the intuition extends naturally to the general case.

We can now see how the motivating paragraph from *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite> can lead to misunderstandings. It suggests that confidence calibration is important for selective classification, but it does not clarify that calibration is orthogonal to the model's ability to discriminate between correct and incorrect predictions on individual inputs. Rather, good calibration instead allows the abstention decision rule (threshold) to be set reliably without validation labels. Crucially, calibration is not actually necessary to achieve good selective classification performance if validation labels are available.

<div class="takeaway-box" id="key-takeaway-2">
<p class="takeaway-title">Key Takeaway 2</p>
<p><em>A well-calibrated model does not in any way guarantee good performance for downstream decision making for individual inputs. What fundamentally determines performance is the ability to <strong>discriminate</strong> between $\text{event}$ and $\text{no event}$ for each individual set of $\text{conditions}$. </em></p>
<p><em>
Calibration enables the reliable setting of decision rules based on estimated average loss/reward without access to ground truth events/labels.  However, if validation labels are available, the same rules can be equivalently determined with uncalibrated models by directly estimating loss/reward.</em></p>
</div>

We’ve just seen, at an intuitive level, that a model can be perfectly calibrated yet useless for decision making, and vice-versa. Now we will write out the above discussion more mathematically. Again, if the intuition is sufficient, you can [skip to the interactive widget below](#interactive-example-continued). Consider again the calibration equation (Eq. \ref{eq:calib-math}) and Eq. \ref{eq:calib-math-avg},

$$
P(e \mid  \pi) = \pi, \quad \text{for }\pi={P}_{\theta}(e\mid c) \in [0,1],
$$

$$
P(e \mid  \pi)
= \sum_{c}
P(e \mid  c)\,P\bigl(c \mid  \pi\bigr).
$$

Recall that $P(e \mid  \pi)$ is **not** given $\text{conditions}$ $c$. As such, calibration does not guarantee for any specific $\text{conditions}$ $c$ that the model's predicted probability $\pi$ matches the actual probability of $\text{event}$ occurring, i.e.

$$
\underbrace{P(e \mid  c)}_{\text{true prob. given }c} \approx \underbrace{\pi(e,c)={P}_{\theta}(e\mid c)}_{\text{model pred. prob.}}.
$$

In Bayesian decision theory <d-cite key="Chow1970OptimumReject,Bishop2006PatternRecognition"></d-cite>, the optimal decision rule *requires* knowledge of the true probability $P(e \mid  c)$ for each individual $c$ in order to minimise expected loss/reward on individual inputs, 

$$
\begin{aligned}
\text{optimal decision}(c)
&\in
\arg\min_{d(c)}
\mathbb{E}_{P(e \mid c)}
\bigl[
  \mathcal{L}\bigl(e,\pi(e,c),d(c)\bigr)
\bigr] \\
&=
\arg\min_{d(c)}
\sum_{e \in \{0,1\}}
\mathcal{L}\bigl(e,\pi(e,c),d(c)\bigr)\,P(e\mid c).
\end{aligned}
$$

<!-- Thus calibration alone does not tell you whether or not you will be able to make good decisions on individual inputs. Now to concretely discuss our examples of selective classification and loan issuance, for a binary thresholding decision rule, intuitively, the optimal threshold can be obtained using any monotone mapping of the true probability $P(e \mid  c)$ (uncalibrated value) as long as $\text{event}$ and $\text{conditions}$ can be sampled to estimate the expected loss. 

Consider a binary decision with $\text{event } e \in \{0,1\}$. Define

$$
p(c) := P(e=1 \mid c), \qquad s(c) := f\bigl(p(c)\bigr),
$$

where $f:[0,1]\to\mathbb{R}$ is strictly increasing (so $s$ is a possibly uncalibrated but monotone score). A binary threshold decision rule on $s$ is

$$
d(c;t) :=
\begin{cases}
1, & s(c) \ge t,\\[4pt]
0, & s(c) < t.
\end{cases}
$$

For any threshold $t$ there exists a unique

$$
\tau = f^{-1}(t)
$$

such that the decision regions coincide:

$$
\{c : s(c) \ge t\}=\{c : f(p(c)) \ge t\}=\{c : p(c) \ge \tau\}.
$$

The expected loss of the threshold rule can therefore be written either in terms of $t$ or $\tau$:

$$
R(t):= \mathbb{E}_{(e,c)\sim P}\bigl[\mathcal{L}\bigl(e,\pi(e,c),d(c;t)\bigr)\bigr]=
\mathbb{E}_{(e,c)\sim P}
\bigl[
  \mathcal{L}\bigl(e,\pi(e,c),\mathbf{1}\{p(c)\ge \tau\}\bigr)
\bigr],
$$

with $\tau = f^{-1}(t)$. If we can sample $(e,c) \sim P(e,c)$, we can estimate $R(t)$ empirically:

$$
\hat R(t)=\frac{1}{N}\sum_{n=1}^N
\mathcal{L}\bigl(e^{(n)},\pi(e^{(n)},c^{(n)}),d(c^{(n)};t)\bigr),
$$

and choose the threshold $t$ that minimises $\hat R(t)$. Since every $t$ corresponds to some posterior threshold $\tau$, any strictly monotone (possibly uncalibrated) mapping of $P(e \mid c)$ is sufficient for learning the optimal binary thresholding rule from validation data. -->
Thus, calibration alone does not tell you whether or not you will be able to make good decisions on individual inputs. As an aside, one might ask, why not define calibration in terms of $P(\text{event}\mid \text{conditions}) = \pi(e,c)$ directly? The reason is that this is not typically possible to evaluate in practice, as it would require the sampling of multiple $\text{event}$ outcomes for the same $\text{conditions}$. For the rain forecasting example, it would be like being able to rewind time and replay the same day over and over to see if it rains or not.

Now to concretely discuss our examples of selective classification and loan issuance, we aim to show that what matters for decision making performance is how well we can order or distinguish different $\text{conditions}$ using $P_{\theta}(e\mid c)$, not the absolute value of the predicted probability. Note that the true conditional distribution $P(e\mid c)$ achieves maximal discrimination as any ambiguity it contains is irreducible.

Consider again a binary event $e \in \{0,1\}$ and define the true event probability

$$
p(c) := P(e=1 \mid c).
$$

Suppose the model (or some post-processing) produces a scalar score

$$
s(c) := f\bigl(p(c)\bigr),
$$

where $f:[0,1]\to\mathbb{R}$ is strictly monotone. In particular, $f$ is invertible $p = f^{-1}\bigl(s\bigr)$. Assume decisions are taken by some rule $d(c;\eta)$ with parameters $\eta$ that depends on $c$ only through $s(c)$, i.e.

$$
d(c;\eta) = g\bigl(s(c);\eta\bigr) = g\bigl(f(p(c));\eta\bigr).
$$

Because $f$ is invertible and strictly monotone, there is a one-to-one reparameterisation of the decision rule in terms of $p(c)$: for every $\eta$ there exists an $\eta'$ such that

$$
d(c;\eta) = g\bigl(f(p(c));\eta\bigr) = \tilde g\bigl(p(c);\eta'\bigr),
$$

so the resulting actions (and hence decision regions) are identical whether we work with $s(c)$ or with $p(c)$. The expected loss of the decision rule can therefore be written as

$$
R(\eta) := \mathbb{E}_{P(e,c)}\bigl[\mathcal{L}\bigl(e,\pi(e,c),d(c;\eta)\bigr)\bigr] = \mathbb{E}_{P(e,c)}\bigl[\mathcal{L}\bigl(e,\pi(e,c),\tilde g(p(c);\eta')\bigr)\bigr].
$$

If we can sample $(e,c)\sim P(e,c)$ (i.e. validation dataset), we can estimate $R(\eta)$ empirically as

$$
\hat R(\eta) = \frac{1}{N}\sum_{n=1}^N \mathcal{L}\bigl(e^{(n)},\pi(e^{(n)},c^{(n)}),d(c^{(n)};\eta)\bigr),
$$

and choose the parameter $\eta$ that minimises this estimate. Since there is a one-to-one correspondence between $\eta$ and $\eta'$, any strictly monotone (possibly uncalibrated) mapping of $P(e\mid c)$ is sufficient for learning the optimal decision rule within this family from validation data. In other words, if you have validation labels, you can tune your decision rule using any reasonable score that preserves the ordering of examples—whether or not that score is calibrated. Although a calibrated model is not required, in the absence of validation data it allows the estimation of $R(\eta)$ without access to events $e^{(n)}$ as discussed previously (Eq. \ref{eq:expected-loss-est}).

In the particular cases we consider in this blog (selective classification and loan issuance), the decision rule is a simple binary threshold of the score,

$$ 
d(c;\tau) := \begin{cases} 0, & s(c) \ge \tau,\\[4pt] 1, & s(c) < \tau, \end{cases}
$$

and the loss $\mathcal{L}$ does not depend on $\pi$, so what determines performance $R$ is how well the score/model *orders* examples by their true $\text{event}$ probability, i.e. its discrimination ability.<d-footnote>For many practical scenarios the loss is independent of $\pi$. For losses where $\pi$ matters, given $\pi$ from the model, the optimal decision rule can still be found with any order-preserving map of the true distribution.</d-footnote>

Finally, if the reader is interested in exploring further theory related to the above, <d-cite key="PerezLebel2023GroupingLoss,Chidambaram2025reassessing"></d-cite> discuss the limitations of calibration from the perspective of proper scoring rules. 

### Interactive Example Continued

Extending our previous interactive example of the bank deciding whether to issue loans, we introduce three different levels of discrimination for the predicted default probabilities, i.e. how useful $\pi$ is for separating defaulting customers from non-defaulting customers. We use the standard Receiver Operating Characteristic (ROC) curve to show binary discrimination ability. Observe how the maximum actualisable profit when varying $\tau$ depends on the discrimination ability of the model, not its calibration. A well-calibrated model with no discrimination ability is able to reliably estimate that it can't make any money, but it still can't make any money! Also observe how having access to validation labels (i.e. reading the actualised profit directly) means you don't need to pay attention to the estimated profit (and thus the calibration of the model). To summarise: **how well the model separates defaulters from non-defaulters controls how much money you can make; calibration only controls how well you can estimate that profit without labels**.

<div class="l-page">
  <iframe
    src="{{ 'assets/html/2026-04-27-useful-calibrated-uncertainties/loan_discrimination_widget.html' | relative_url }}"
    frameborder="0"
    scrolling="no"
    height="500px"
    width="100%"
    style="display:block; margin: 0 auto; max-width: 100%;"
  ></iframe>
</div>


## Misunderstandings from the Literature

Looking at our motivating paragraph from *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite> again:

> In real-world decision making systems, classification networks must not only be accurate, but also should indicate when they are likely to be incorrect. As an example, consider a self-driving car that uses a neural network to detect pedestrians and other obstructions. If the detection network is not able to confidently predict the presence or absence of immediate obstructions, the car should rely more on the output of other sensors for braking. Alternatively, in automated health care, control should be passed on to human doctors when the confidence of a disease diagnosis network is low. Specifically, a network should provide a calibrated confidence measure in addition to its prediction. In other words, the probability associated with the predicted class label should reflect its ground truth correctness likelihood.

We can see that in an effort to convey the importance of calibration intuitively, the authors have blended calibration together with selective classification without explicitly communicating the nuances that we've discussed in this blogpost. That is to say, a calibrated model allows the reliable estimation of accuracy and setting of decision rules without validation labels, however, good calibration does not say anything about how well a model is able to separate its own errors from correct predictions on a per-sample basis. From the excerpt, one can see how **a reader might mistakenly associate good calibration with good per-sample decision making**. This sort of motivating language, that doesn't explicitly clarify the relationship between calibration and selective classification, has become commonplace in the subsequent calibration literature  <d-cite key="Minderer2021Revisiting,Zhang2020MixnMatchCalibration,param_temp_scaling_eccv2022,Xiong2023ProCal,Nixon2019CVPRW,Meronen2024fixoverconfidence"></d-cite>, which can contribute to a fuzzy understanding of calibration’s practical importance in the research community. 

Another factor that leads to confusion is the use of the word "overconfident". A key empirical finding of *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite> is that certain neural networks are overconfident in the *confidence calibration sense* ([see reliability diagram](#reliability-diagram-figure)).  This overconfidence result is oft-cited in papers <d-cite key="Minderer2021Revisiting,param_temp_scaling_eccv2022,LeCoz2024EfficientCalibration"></d-cite> as motivation without clarifying that overconfidence in calibration *is orthogonal to discriminating individual failures/errors* as we've discussed. Colloquially, overconfidence typically refers to a person's attitude to a given situation (specific $\text{conditions}$): *"He overconfidently decided to choose the hard option only to fail"*. It is therefore not hard to see how someone might conflate the colloquial and calibration meanings of "overconfident" together when engaging with the literature, confusing calibration and discrimination. 

Below I've included a few more examples from the literature of imprecise use of language when motivating calibration:

From *Proximity-Informed Calibration for Deep Neural Networks* <d-cite key="Xiong2023ProCal"></d-cite>:

> ...suppose a dark-skinned individual has a high risk of 40% of having the cancer. However, due to their underrepresentation within the data distribution, the model overconfidently assigns them 98% confidence of not having cancer. As a result, these low proximity individuals may be deprived of timely intervention.

Here the authors discuss overconfidence and intervention for an *individual* case, mixing together calibration with per-sample discrimination and decision making.

From *Revisiting the Calibration of Modern Neural Networks* <d-cite key="Minderer2021Revisiting"></d-cite>:

> Model calibration refers to the accuracy with which the scores provided by the model reflect its predictive uncertainty. For example, in a medical application, we would like to defer images for which the model makes low-confidence predictions to a physician for review. Skipping human review due to confident, but incorrect, predictions, could have disastrous consequences.

Here the authors define calibration in natural language, but then go on to describe selective classification as a motivating example, blurring the distinction between the two concepts. 

From *Fixing Overconfidence in Dynamic Neural Networks* <d-cite key="Meronen2024fixoverconfidence"></d-cite>:

> ... adaptive early exiting is typically based on the confidence scores at the individual exits or learned gating functions. Thus, it is crucial that the predictive densities or gating functions are robust and allow for trustworthy decision-making. However, current approaches are problematic as typical neural architectures are: (i) miscalibrated, (ii) overconfident, and (iii) their predictions do not capture epistemic uncertainties...

In this case the authors have introduced calibration in the context of dynamic neural networks whose performance depends on their ability to make decisions on a per-sample basis.

Although not all papers mix calibration and discrimination/per-sample decision making together, examples like the above are commonly found throughout the literature. I hope that after this blogpost the reader is now equipped to avoid the misunderstandings that may arise from such examples.
### A Brief Retrospective

Readers mainly interested in practical takeaways can skim this section; it’s here to provide some historical context to the rest of the blogpost.

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/timeline.png" class="img-fluid z-depth-1" %}

<div class="caption">
<strong>
A summarised timeline of calibration in the literature.
</strong>
</div>

So, how did we get here? Originally calibration was very much grounded in the idea that it was a property of models averaged over many predictions. Early papers (around the 80s) <d-cite key="degroot1983comparison,Dawid1982"></d-cite> pose calibration as simply one potential way to evaluate the quality of probabilistic forecasting models and don't link it to per-sample decision making or notions of safety. Moreover, good calibration was already emphasised as not being sufficient, with the example of a useless but well-calibrated model that always predicts the base rate already present <d-cite key="degroot1983comparison"></d-cite>.

Around the 2000s, with the increase in popularity of machine learning models such as SVMs, decision trees and neural networks, calibration begins to be emphasised as important for enabling reliable downstream decision making <d-cite key="NiculescuMizilCaruana2005,Platt1999,ZadroznyElkan2001"></d-cite>. The motivation here is generally to do with being able to trust the probabilities to set decision rules based on average performance (without validation labels), e.g. <d-cite key="NiculescuMizilCaruana2005"></d-cite> define a fixed decision threshold that assumes calibrated probabilities and evaluate downstream profits much like in our interactive example. Interestingly, the distinction between discrimination and calibration is not really emphasised, however, since research at this time generally focuses on binary models, the importance of discrimination is already implicitly baked in as a key part of the performance of the model. The line of thought is "by default we already measure discrimination via accuracy/AUROC, we should also measure calibration".

Into the 2010s, as deep learning becomes more popular, Guo et al.'s *On Calibration of Modern Neural Networks* <d-cite key="Guo2017Calibration"></d-cite> firmly associates calibration with notions of safety, uncertainty estimation and per-sample decision making. This motivation is then adopted by the uncertainty community and propagated throughout the literature. Calibration often appears as a key desideratum for uncertainty estimation methods <d-cite key="Ovadia2019Shift,Nixon2019CVPRW,Lakshminarayanan2017Ensembles,NeurIPS2020TutorialUncertainty,malinin2020ensemble"></d-cite>, frequently without clarifying the distinction between calibration and discrimination. Crucially, with the advent of large-scale multi-class classification and the definition of the binary $\text{event}$ as "correct prediction", the discrimination of the binary (error detection) model is no longer implicitly emphasised, as the default performance measure is the base classifier's multi-class accuracy. Meanwhile, selective classification research that explicitly focuses on said binary model appears as a separate thread <d-cite key="Geifman2017SelectiveClassification,geifman2018biasreduced,pmlr-v97-geifman19a"></d-cite>.

Finally, in the 2020s, calibration becomes firmly established as a benchmarked performance metric. Method papers keep motivation brief and often imprecise, instead focusing on improving performance <d-cite key="Minderer2021Revisiting,Zhang2020MixnMatchCalibration,param_temp_scaling_eccv2022,Xiong2023ProCal,LeCoz2024EfficientCalibration"></d-cite>. As an established "measure of reliability", in the era of large language models, calibration finds itself as a small part of many broader evaluations <d-cite key="Srivastava2022BIGBench,Liang2022HELM"></d-cite>, without much accompanying clarification, even featuring in OpenAI's GPT-4 tech report <d-cite key="OpenAI2023GPT4"></d-cite>.

Now one can see how the nuances behind calibration have been somewhat lost over time, leading to the "fuzzy impression" we've discussed in this blogpost. Interestingly, there have been recent works that have started to re-emphasise the distinction between calibration and discrimination <d-cite key="Jaeger2023FailureDetection,PerezLebel2023GroupingLoss,Chidambaram2025reassessing,zhu2022rethinking"></d-cite>, however, this nuance is still not widely appreciated in the community (which is what partially motivated the writing of this blog). In particular, some works highlight that methods that improve calibration, such as label smoothing, may degrade discrimination for selective classification <d-cite key="zhu2022rethinking,xia2025towards"></d-cite>.
## Closing Thoughts and Takeaways

The aim of this blogpost was to clarify the usefulness of calibrated probability estimates in machine learning. We showed that a calibrated model allows reliable estimation of average loss/reward, and consequent decisions based on those estimates, without access to validation labels. However, it does not in any way guarantee good performance on tasks that involve per-sample decision making. For the latter, good discrimination is required instead. We also discussed and highlighted how confidence calibration and selective classification have been mixed together in the literature, leading to potential misunderstandings about the nature of calibration with respect to per-sample decisions. As a practical rule of thumb, whenever you see (or design) a calibration experiment, it’s worth asking: is the downstream decision about average behaviour, about individual cases, or both? The answer should determine whether calibration (or labels at deployment time) or discrimination or both are actually the bottleneck. We hope that the above blogpost has been helpful to the reader and that they now have a clearer understanding of what calibrated probabilities are actually useful for.

{% include figure.liquid path="assets/img/2026-04-27-useful-calibrated-uncertainties/understands.png" class="img-fluid z-depth-1" %}
