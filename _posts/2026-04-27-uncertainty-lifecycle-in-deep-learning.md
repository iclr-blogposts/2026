---
layout: distill
title: Uncertainty Lifecycle in Deep Learning
description: Uncertainty modeling in deep learning has different attributes such as uncertainty propagation, uncertainty estimation, uncertainty decomposition, uncertainty attribution and uncertainty sensitivity, that are extensively discussed in literature. However, there is no proper structure explaining how these different components interact with each other at different stages of Deep Learning pipeline. We propose to structure the flow and transformation of uncertainty from input to prediction through the model, by appropriately positioning them. And we call this structure as “Uncertainty Lifecycle”. The “Uncertainty lifecycle” can be represented as a structured process for handling, quantifying, analyzing, and interpreting uncertainties at different stages of Deep Learning pipeline.

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
bibliography: 2026-04-27-uncertainty-lifecycle-in-deep-learning.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Uncertainty Lifecycle
    subsections:
      - name: 1. Uncertainty Propagation
      - name: 2. Uncertainty Estimation
      - name: 3. Uncertainty Decomposition
      - name: 4. Uncertainty Attribution
      - name: 5. Uncertainty Sensitivity
  - name: Conclusion


# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#     font-size: 16px;
#   }
---

## Introduction
Uncertainty in Machine Learning arises due to <d-cite key="koller2009probabilistic"></d-cite>:
- Limitations in our ability to observe the world
- Limitations in our ability to model it
- Innate non-determinism itself

These uncertainties exist either in data or in model or both.The uncertainty in **data** arises due to measurement noise, inherent randomness, and label/feature ambiguity. These uncertainties are called as data uncertainty or **Aleatoric Uncertainty** <d-cite key="der2009aleatory"></d-cite> and are often irreducible even with more data. The uncertainty in the **model** arises due to limited training data, model misspecification, optimization imperfections, or out-of-distribution inputs. These uncertainties are called as model uncertainty, knowledge uncertainty or **Epistemic Uncertainty** <d-cite key="der2009aleatory"></d-cite>, and are often reducible with more data or better modelling. Both uncertainties together are called as **Total Uncertainty (or Predictive Uncertainty)**.

The different concepts in uncertainty as shown in Figure 1, are discussed and studied in the literature extensively <d-cite key="gawlikowski2023survey, abdar2021review, loquercio2020general, wang2024uncertainty, mucsanyi2024benchmarking"></d-cite> either individually or as a combination of one or two component. But there is no such work that highlight and represent their connection and flow in a single framework. 

{% include figure.liquid path="assets/img/2026-04-27-uncertainty-lifecycle-in-deep-learning/Uncertainty_Concepts_1.png" class="img-fluid" %}
<div class="caption">
  Figure 1: Different Uncertainty Concepts in Deep Learning
</div>

In this blogpost, we have focused to provide a holistic framework and call it as **"Uncertainty Lifecycle"**, to show how different concepts of uncertainty are applicable at different stages in the uncertainty modeling. 


_"Uncertainty lifecycle” can be represented as a structured process for handling, quantifying, analyzing, and interpreting uncertainties at different stages of Deep Learning pipeline_

## Uncertainty Lifecycle

Uncertainty Lifecycle represents the holistic framework to connect and position the major identified concepts of uncertainty modeling which are
  1. Uncertainty Propagation
  2. Uncertainty Estimation 
  3. Uncertainty Decomposition
  4. Uncertainty Attribution
  5. Uncertainty Sensitivity

In Figure 2, we see the complete lifecycle of uncertainty flow. The inherent uncertainties present in the data(aleatoric) and model(epistemic) propagates through the model to the output. This propagated uncertainty is measured by the uncertainty estimation techniques and further decomposed into aleatoric and epistemic uncertainty. To estimate the propagated uncertainty, we need to model it through uncertainty propagation module either at intermediate layer or at the output layer as per the requirement. The output of this modeling will be the distribution which will be fed to the uncertainty estimation module. 

Further, the obtained total uncertainty or the decomposed uncertainty can be attributed by the uncertainty attribution concept, to its actual root cause which could be any specific input feature or the specific model layer. The attributed/identified region causing this uncertainty is obtained in form of an explanation/attribution map or importance score. Also, the sensitivity of these estimated uncertainties to the input or the model is analyzed by the uncertainty Sensitivity analysis concept. The sensitivity analysis can be further performed only for specific region in the input/model causing the uncertainty, as identified by the uncertainty attribution.

{% include figure.liquid path="assets/img/2026-04-27-uncertainty-lifecycle-in-deep-learning/Uncertainty_Lifecycle_4.png" class="img-fluid" %}
<div class="caption">
  Figure 2: Uncertainty Lifecycle; This figure shows applicability of different concepts of uncertainty at different stages: Uncertainty propagation, Uncertainty Estimation, Uncertainty attribution and uncertainty Sensitivity.  It explains how data and model uncertainty propagate through the model and captured at the output in the form of total uncertainty by estimating the predictive distribution over output (sometimes you may get decomposed uncertainty directly). The total uncertainty can be decomposed into epistemic and aleatoric uncertainty. Uncertainty Attribution and Uncertainty sensitivity method explains the predictive uncertainty or decomposed uncertainty. It also shows how uncertainty attribution and uncertainty sensitivity method interact with each other.
</div>

Once the cause and sensitivity of uncertainty is identified, that is which region or which feature is causing the uncertainty, appropriate measures can be taken to reduce these uncertainties.

This cycle of propagating, measuring, decomposing and attributing the uncertainty can be repeated to gain further insight of the model, and improve the model performance accordingly.

One more thing to notice in Figure 2 is that there is no clear demarcation between some of the components. For e.g. uncertainty attribution and sensitivity can be part of same analysis or treated separately as they both observe how uncertainty changes with change in input or model parameters. Further, many methods implemented to do estimation does propagation implicitly. And, many methods directly give decomposed uncertainty, and you may not need separate decomposition. 
Although they may not be completely separable, this structure gives a framework for positioning our actions, a bit more clearly.
Let’s discuss each of the concept in Uncertainty Lifecycle

### 1. Uncertainty Propagation:
**• What it is:** The process of tracking how uncertainties in inputs, model parameters, or intermediate layers flow through a neural network to affect its predictions.

**•	What it does:** Simulates the transformation of uncertainties (e.g., noisy data or uncertain weights) through the model’s computations (e.g., convolutions, activations), producing a distribution or samples of outputs. It tries to answer - “How does uncertainty flow through the model?”

**•	Input:** Data with uncertainties (e.g., noisy sensor readings modeled as distributions) and model elements like uncertain weights (e.g., from limited training data).

**• Output:** A distribution, set of samples, or statistical moments (e.g., mean and variance) representing the uncertainty in predictions.

**•	Role in the Flow:** The foundational step, initiating uncertainty analysis by propagating uncertainties through the model, providing raw data for subsequent steps.

**•	Example Approaches:** Monte Carlo Sampling <d-cite key="blundell2015weight"></d-cite>, BNN <d-cite key="tishby1989consistent, graves2011practical, goan2020bayesian"></d-cite>, Analytical methods (variance propagation rules), probabilistic layers, ADF <d-cite key="gast2018lightweight"></d-cite>.

Generally, propagation refers to observing how the uncertainty changes at the output layer. However, many other works also refer propagation as observing the uncertainty changes in intermediate layer. For e.g. ADF method in <d-cite key="gast2018lightweight"></d-cite> propagate uncertainty via moment matching and modeling the activations also as distribution, and not just the output. 

These recent methods enrich the explanation of uncertainty propagation by emphasizing approximation and efficiency for deep networks, where exact propagation (e.g., full Bayesian integrals) is often infeasible. They all operate within the same framework: starting with input/model uncertainties (e.g., distributions), transforming them through layers (via approximations like moment-matching in ADF <d-cite key="gast2018lightweight"></d-cite>, subspace projection in WGMprop <d-cite key="monchot2023input"></d-cite>, or message-passing in factor graphs <d-cite key="daruna2023uncertainty"></d-cite>, and outputting uncertainty representations (e.g., variances or distributions)

### 2. Uncertainty Estimation/Quantification
**• What it is:** The process of quantifying how confident or uncertain a model is about its predictions, often summarizing it into a single metric or distribution. (Estimation and quantification are synonymous, referring to measuring uncertainty magnitude). 

**•	What it does:** Distills propagated uncertainties into a concrete measure of confidence (e.g., variance, entropy), enabling assessment of prediction reliability. It tries to answer – “How uncertain is the prediction?”

**•	Input:** The output from uncertainty propagation, such as a distribution, sampled predictions, or intermediate variances.

**• Output:** A quantified uncertainty metric, e.g., variance, standard deviation, entropy, confidence intervals, or a probability distribution (e.g., “Prediction: 0.85 with variance 0.02”).

**•	Role in the Flow:** The second step, bridging propagation’s raw distributions to interpretable metrics, enabling decomposition and further analysis for the next steps.

**•	Example Approaches:** Measuring the Entropy, Mutual Information of the output distribution, Predictive variance as measure of uncertainty, computing variance over ensemble output. These methods are applicable to output distributions or sampled predictions obtained via Ensemble <d-cite key="lakshminarayanan2017simple, huang2017snapshot, valdenegro2019deep, wenzel2020hyperparameter"></d-cite>, BNN <d-cite key="kendall2017uncertainties, tishby1989consistent, graves2011practical,goan2020bayesian"></d-cite>, Monte Carlo Dropout <d-cite key="gal2016dropout"></d-cite>, Test Time Augmentation <d-cite key="lyzhov2020greedy"></d-cite>.

**•	How it is Measured:** The total uncertainty is quantified using statistical or information-theoretic metrics such as variance/standard deviation, entropy, confidence intervals, mutual information. For e.g., for a classifier, entropy of 0.7 bits indicates high uncertainty, while a regression model’s standard deviation of 2.5 units quantifies spread.

Uncertainty propagation and uncertainty estimation are not seen separately in many implementations. The output of propagation is distribution or sampled predictions and output of estimation is an uncertainty score computed from this distribution. 

**•	Propagation = mechanism** (push uncertainty through).   
**•	Estimation = measurement** (summarize how much uncertainty there is).

Uncertainty Propagates first, and then it is estimated on the prediction.
In a few approaches like ADF <d-cite key="gast2018lightweight"></d-cite>, the uncertainty is estimated at the intermediate layer as well. In research community, however the term uncertainty quantification/estimations mostly refer to measuring the uncertainty at the output prediction.

**3. Uncertainty Decomposition**

**• What it is:** A quantitative technique which split the total uncertainty into distinct components, such as aleatoric (data-related) and epistemic (model-related) uncertainty, or contributions from specific features or layers.

**•	What it does:** Splits the quantified uncertainty to reveal the contribution of each source, enabling targeted diagnostics (e.g., distinguishing data noise from model gaps). It tries to answer - “What type of uncertainty do we have, and how much of each?”

**•	Input:** The quantified total uncertainty (e.g., total variance) and supporting data from propagation (e.g., samples, distributions).

**•	Output:** A breakdown of uncertainty contributions into its components.

**•	Role in the Flow:** The third step, refining total uncertainty into components, providing input simultaneously for uncertainty attribution and uncertainty sensitivity analyses.

**•	Example Approaches:** Variance Decomposition <d-cite key="depeweg2018decomposition, kendall2017uncertainties, kwon2020uncertainty, depeweg2017sensitivity, mucsanyi2024benchmarking"></d-cite>, Entropy-based Decomposition <d-cite key="depeweg2018decomposition, smith2018understanding, hullermeier2021aleatoric"></d-cite>, Mutual Information Decomposition <d-cite key="depeweg2018decomposition, smith2018understanding"></d-cite>, Kendall and Gal’s Heteroscedastic approach<d-cite key="kendall2017uncertainties, kwon2020uncertainty"></d-cite>.

**•	How it is Measured:** Decomposition is measured by the proportion or magnitude of uncertainty attributed to each component. For e.g. a total variance of 0.05, decomposition might yield 0.03 (60%) aleatoric and 0.02 (40%) epistemic.

In literature, uncertainty decomposition means splitting the predictive uncertainty into aleatoric and epistemic uncertainty.

### 4. Uncertainty Attribution
**• What it is:** The process of explaining and localizing which parts of the input or model contribute most to the uncertainty in a given prediction.

**•	What it does:** Assigns uncertainty to specific causes, providing human-readable explanations. It tries to answer the question - “Where is the uncertainty coming from? Which parts of the input or model contributed to this uncertainty?”

**•	Input:**  Predictive uncertainty or decomposed uncertainty components (e.g., aleatoric/epistemic breakdowns)

**•	Output:** Explanations of uncertainty sources, e.g., attribution maps or scores that localize/quantify the root cause of uncertainty.

**•	Role in the Flow:** A parallel final step, focusing on explaining and interpreting uncertainty cause for decision-making and debugging. 

**•	Example Approaches:** Techniques adapted from XAI (Explainable AI) field for uncertainty – Perturbation based attribution giving feature importance <d-cite key="wood2024model, watson2023explaining, phillips2018interpretable"></d-cite>, Gradient-based attribution giving saliency map <d-cite key="amanova2024finding, bley2025explaining, wang2023gradient, perez2022attribution"></d-cite>, Counterfactual based attribution giving counterfactuals <d-cite key="antoran2020getting, ley2021delta, ley2022diverse"></d-cite>.

**•	How it is Measured:** Attribution is measured by qualitative or semi-quantitative assignments of uncertainty back to sources. For e.g. the region near nose in the input image of a cat is responsible for the uncertainty in prediction or feature number 3 in the tabular data is responsible for the uncertainty.

Uncertainty decomposition tells whether the source is from data or model, whereas uncertainty attribution gives more fine grain and localize root cause analysis of the uncertainty - “Pixel noise in region X contributes 60% to aleatoric uncertainty.”

### 5. Uncertainty Sensitivity
**•	What it is:** An analysis of how sensitive the model’s output uncertainty is to perturbations in inputs, parameters, or other factors.

**•	What it does:** Quantifies the influence of specific factors on uncertainty, revealing which elements (e.g., input noise) most affect uncertainty when changed.

**•	Input:** Total or decomposed uncertainty components (e.g., aleatoric/epistemic variances) and supporting data from attribution.

**•	Output:** Sensitivity metrics, e.g., “Reducing noise in feature Y by 10% decreases uncertainty by 15%,” or rankings of influential factors.

**•	Role in the Flow:** A parallel final step focusing on “what-if” scenarios and robustness, complementing attribution’s explanatory focus.

**•	Example Approaches:** Sobol Sensitivity Analysis <d-cite key="fel2021look"></d-cite>, Gradient-Based Methods<d-cite key="depeweg2017sensitivity"></d-cite>, Perturbation Analysis.

**•	How it is Measured:** Sensitivity is measured by metrics quantifying the change in uncertainty due to perturbations. Relative change in uncertainty, e.g., “10% noise reduction in feature Y lowers variance by 0.01.”

In the literature, uncertainty sensitivity and uncertainty attribution are related but they are not the same - attribution tells you “Where uncertainty is coming from”, sensitivity tells you “How fragile uncertainty is to small input changes.”


In summary, the uncertainties in data and model flows through the model to the output, where **Propagation** tracks them, **Estimation/Quantification** measures them, **Decomposition** splits them into components, **Attribution** explain and localize uncertainty’s root cause and **Sensitivity** test robustness with respect to the input. Table 1 summarizes the same along with an example

{% include figure.liquid path="assets/img/2026-04-27-uncertainty-lifecycle-in-deep-learning/Uncertainty_Summary_Table.png" class="img-fluid" %}
<div class="caption">
  Table 1: Summary of Uncertainty Lifecycle along with usecase and example
</div>

## Conclusion
We discussed the following concept in detail and how they flow through the pipeline from input to output
- Propagation: how uncertainty flows through the system.
- Estimation: Quantifies the associated uncertainty as some score
- Decomposition: why the model is uncertain (aleatoric vs epistemic).
- Attribution: which features cause uncertainty.
- Sensitivity: how uncertainty changes under small input changes.

And in this work, we have proposed an **“Uncertainty Lifecycle”** which presents a holistic framework for connecting and positioning different studied uncertainty concepts in literature. It shows how they are interconnected with each other, where output of each serve as an input for the next like a flow. The entire uncertainty lifecycle can be repeated to gain insight about model and data, debug the model and improve the performance.  It can further provide explanation and reliability measures to end user for appropriate decision-making. 
