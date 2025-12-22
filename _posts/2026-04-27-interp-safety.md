---
layout: distill
title: 'Charting the Depths: Interpretability Tools to Enhance LLM Safety'
description: "Motivated by the increasing deployment of LLMs for safety-critical applications, we provide an accessible introduction to a practical suite of interpretability tools useful for understanding LLMs’ behavior during safety-critical decisions. Previous discussions of interpretability are often heavily focused on these methods' technical aspects, rather than giving practical guidance for their immediate use; here, we provide practitioners with an overview of a range of methods for understanding LLM behavior. For each method covered, we highlight what it can and cannot tell us, and how this can help inform deployment decisions."
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
bibliography: 2026-04-27-interp-safety.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: The Challenge of Model Interpretability
  - name: 'Exploring the Surface: Model Behavior Analysis'
  - name: 'Diving Deeper: Model Introspection'
    subsections:
      - name: 'Method One: Probing'
      - name: 'Method Two: Sparse Autoencoders'
  - name: What's Next?
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

Let’s say we have an AI model that’s making an important decision: given a student’s college recommendation letter, output a numerical score (1-9) representing the student’s admissibility. LLMs and other AI models are on track to become part of admission processes. But how can we be confident in their behavior? In this article, we take the perspective of AI safety (rather than AI policy). Thus, instead of defining safe or ideal behaviour for this use case, our goal is to help readers understand some of the methods available to them, given such a definition. More specifically, we’ll describe a range of methods from the rapidly developing field of **interpretability** that can help model developers and deployers enhance safety within AI systems.

Assume that we built some kind of recommendation letter scoring model consisting of a fine-tuned LLM that inputs a recommendation letter and outputs a score as the next token. How do we make sure that this model does not do anything that we don’t want it to do, like basing students’ scores on whether or not their letters mention their love of Taylor Swift? 

<div class="row justify-content-center mt-3">
  <div class="col-10 col-sm-8 col-md-6 col-lg-4 mt-3 mt-md-0">
{% include figure.liquid path="assets/img/2026-04-27-interp-safety/taylor_no.gif" class="img-fluid" %}
  </div>
</div>

At a high level, there are three steps where we could intervene.

First, we could try to prevent the model from learning that information in the first place. For example, we could filter the training data to make sure that Taylor Swift appeared in both positive and negative contexts. 

Second, we could evaluate our model to see how it was doing after being trained. Do we actually observe a difference in scoring for students who do or don’t mention Taylor Swift? Since we don’t want a student’s musical taste to have any impact on their recommendation letter score, we’d also need to look at its internals. Here, internals refer to the mechanisms by which the model goes from inputs to outputs. In particular, we’re interested in the particular features or pieces of information that affect the scores being generated. Ideally, we wouldn’t detect music fandoms having any influence on our model's predictions! 

Third, if we did find a negative behavior (or a problematic internal association), we would need to either a) perform additional training to correct it, b) take additional steps at deployment time to override the model, or c) simply not deploy the model. 

Dataset filtering, runtime monitors, and other techniques address the first and third; however, the second remains highly challenging. For this reason, our article focuses on the second step, exploring some ways we can begin to “crack” model internals.

## The Challenge of Model Interpretability

When it comes to assessing a model’s behavior, we want to know not only _what_ it will do, i.e., directly relating inputs to outputs, but also _why_ it will do this, i.e., understanding the mechanisms behind how this relation occurred. For example, we may find that the model scores fans of Taylor Swift differently from students with other favorite artists. Is this because the model uses being a fan of Taylor Swift as a relevant feature for its output score, or is this merely a statistical artifact?

<div class="row justify-content-center mt-3">
  <div class="col-12 col-sm-10 col-md-8 col-lg-6 mt-3 mt-md-0">
{% include figure.liquid path="assets/img/2026-04-27-interp-safety/billie_why.gif" class="img-fluid" %}
  </div>
</div>

The idea of separating _what_ and _why_ questions naturally allows us to categorize interpretability methods. Generally speaking, methods that work with model inputs and outputs give us our _what_ answers. Methods at this level are precise, but don’t let us go beyond specific data points. 

We have to dive deeper to learn about why the model does what it does. In this post, we present examples of methods that give general insights about a model’s internal representations, at the expense of precision.

Alright, get your snorkels! We’re going to start at the surface with a brief overview of what the input and output spaces can tell us.

## Exploring the Surface: Model Behavior Analysis

Without diving deeply into model internals, we can learn a lot about a model’s behavior by simply examining _what it says_ as a function of _what we put in_. Treating models like black-box functions (which is our only option for most proprietary models) allows us to ask and answer questions without making any assumptions about what’s happening under the hood. 

Back to our college admissions example. We were worried that the model we’re using to rate recommendation letters was treating Taylor Swift fans unfairly. How do we know if this might be the case? As a _first_ step, let’s take a look at the relationship between input space and output space.

What if all we have is a model’s output? It turns out that, as discussed by <d-cite key="morris2024inversion"></d-cite>, this can actually tell us quite a bit _about the input!_  Let’s take one of our recommendation letters as an example input: “They are an excellent mathematics student and a huge Taylor Swift fan.” Suppose our model has generated the output: “4.” One experiment we can do is to swap out different musical artists to see how much the likelihood of “4” changes. 

{% include figure.liquid path="assets/img/2026-04-27-interp-safety/fig_1.png" class="img-fluid" %}

Using this measure, we can test different inputs and explore how much the output distribution changes. Which part of our recommendation letter mattered most in producing a final score? The artist “Taylor Swift”? The subject “mathematics”? The adjective “excellent”?

This approach can give us a sense of which words are most impactful on a particular outcome, but it doesn’t tell us whether or not the original input was within the model’s training distribution. <d-cite key="xiong2024confidence"></d-cite> presents a range of strategies to help LLMs calibrate their own uncertainty, including using human-inspired prompts, strategically sampling multiple responses, and defining an aggregation method that allows computing consistency among responses. These options give us additional insights into a model’s confidence about a particular input. Practically speaking, this allows us to handle cases flagged as OOD differently, for example by routing to a human expert for verification.

{% include figure.liquid path="assets/img/2026-04-27-interp-safety/fig_2.png" class="img-fluid" %}

So, what _can’t_ methods examining the input and output space tell us? A lot, actually. Because they require starting from inputs that either are specified by us pre-deployment or created by the users post-deployment, they don’t generalize to things that we haven’t considered or observed.  Although they allow us to approximate the impact of a given word on an outcome, or a model’s certainty about an input, they don’t tell us anything about _why_ the model’s predictions happen.

If you’ve made it this far, grab your fins and an oxygen tank! It’s time to go beneath the surface.

## Diving Deeper: Model Introspection

Let’s suppose that through some computational magic, we were able to measure our model’s output for every possible music preference.

If it turned out that all artists were treated equally, i.e., had the same distribution of scores, could we conclude that our model was behaving in the way we wanted it to?

Yes… Well, yes, if we only care about _what_ the model is doing and not _why_.

Here are two reasons “why” is important. First, understanding a model’s “inner model” can allow us to make predictions about how it will behave out of distribution. For example, <d-cite key="zhou2024fourier"></d-cite> shows that LLMs use Fourier features when solving addition problems suggesting that they will be able to generalize to problems that weren’t contained in their training data, since their internal model of arithmetic aligns with ours. Within a problem space of clear correct answers, a good inner model suggests good performance in the future, even if the input distribution changes.

Our second reason for caring about why comes from the space of problems that don’t have a right/wrong answer and are evaluated based on preferences. You can imagine many different recommendation letter scoring models that end up with the same score distribution. However, one that was basing its decisions on grades and classroom behavior would be much better than one basing them on musical preferences.

<div class="row justify-content-center mt-3">
  <div class="col-12 col-sm-10 col-md-8 col-lg-6 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/2026-04-27-interp-safety/bad_bunny_better.gif" class="img-fluid" %}
  </div>
</div>

Given LLMs’ text-likelihood-maximizing pre-training objective and capacity for memorization, it’s reasonable to think they would do nothing more than model statistical patterns and make (sometimes spurious) correlations <d-cite key="bender2021parrots"></d-cite>. In fact, they have been shown to meaningfully model higher-level concepts. In one notable example, <d-cite key="li2023othello"></d-cite> asked a GPT model to predict moves in the board game Othello, which it had never seen before. Using probes (which we’ll describe in detail momentarily), they discovered that the model had an internal representation of the board state that could be interpreted geometrically.

Works like these established LLMS do, in fact, have internal concepts worth exploring. How exactly can you do that? We’re glad you asked!

### Method One: Probing
Probing methods are based on a very sensible assumption: if we can train a simple (linear) model to predict some complex attribute (e.g., a student's favorite artist) based on the internal activations of another model (e.g., our recommendation letter scorer), then the latter model must be encoding that attribute since the linear model isn’t powerful enough to discover this attribute on its own. 

The method proposed by <d-cite key="hewitt-manning-2019-structural"></d-cite> tests whether LLMs encode syntax dependency parses by doing just that. They successfully train a model to predict syntax tree distance between all pairs of words in all sentences of a corpus, giving the model only an LLM’s hidden representations of the sentences. It turns out the model really was representing syntax!

{% include figure.liquid path="assets/img/2026-04-27-interp-safety/fig_3.png" class="img-fluid" %}

How exactly does this work? <d-cite key="hewitt-manning-2019-structural"></d-cite> begins by producing input pairs (x,y) where x is the hidden state of an LLM after inputting a particular sequence and y is the trait being probed. Given a dataset in this form, they fit a matrix to the data with the objective of minimizing the difference between the true label and the one assigned by the matrix. This allows us to answer the question: how much can a simple model learn about y from x alone? 

Instead of searching over all possible favorite colors and favorite artist inputs, probing allows us to determine whether our model has a (linear) internal representation of these attributes. Depending on the attributes identified, we may decide the model isn’t able to perform the prediction task in an unbiased way. If it’s easy for a classifier to guess a student’s favorite artist from the internal states of a model scoring their recommendation letter, then that information is probably way too heavily used. 

Although this is a great thing to be able to check, there are a few notable things that probing doesn’t tell us. Since we need to know what we’re looking for, probing doesn’t give us any information about concepts we hadn’t thought to test. We also don’t know where within the model that information might be encoded, so to localize it we would need to probe every layer. Probing also assumes that the information we’re looking for within the LLM is linearly encoded. Although it has worked well in practice <d-cite key="wu2025axbench"></d-cite>, , it seems reasonable that significant amounts of information are not linearly encoded. If we observe that a probe succeeds, it’s reasonable to conclude that our model encoded the information we were probing for. However, independent of the probing model’s complexity, we can’t use this approach to make conclusions about the inverse: if the probe fails, our model may _still know the thing we were probing for!_

<div class="row justify-content-center mt-3">
  <div class="col-10 col-sm-8 col-md-6 col-lg-4 mt-3 mt-md-0">
    {% include figure.liquid path="assets/img/2026-04-27-interp-safety/ariana_know.gif" class="img-fluid" %}
  </div>
</div>

### Method Two: Sparse Autoencoders
This brings us to the second method for this section: Sparse Autoencoders.

Maybe our recommendation letter scoring model is using a concept we hadn’t thought of, or maybe (and by maybe, we mean it’s extremely likely) we simply want to understand its internal processes before deploying it in the world. 

Sparse Autoencoders (SAEs), or more general disentangling methods, offer us one lens for looking deeper inside LLMs. The sparsity of an SAE is created by a constraint: for any given input, the majority of latent neurons should be inactive. As a result, the SAE has to be _very_ selective about the features it considers. 

This approach to interpretability is based on _sparse dictionary learning_. We can represent the internal activations of our LLM as a set of vectors $$X$$. Assuming that this set can be created through a sparse, linear combination of unknown vectors $$G$$ (which is exactly the assumption that SAEs make!), then we can learn a “dictionary” of vectors $$F$$, where each network feature $$g_i$$ has a corresponding dictionary feature $$f_j$$ that approximates it. The goal of SAEs is to learn this dictionary set $$F$$, since once we have it we can make inferences about the model’s internal feature set $$G$$. <d-cite key="huben2024sparse"></d-cite> approaches this by training an autoencoder that inputs an LLM’s hidden activations given input $$x$$ and outputs a reconstruction $$\hat{x}$$ of that input. To achieve the desired dictionary property, <d-cite key="huben2024sparse"></d-cite> adds a sparsity penalty to the autoencoder. The model must then learn which features of the hidden activations are _most important_ for the current input.

How does this help us understand what’s going on inside an LLM? It turns out that when the authors of <d-cite key="huben2024sparse"></d-cite> examined which neurons were activated for different inputs, they were able to find highly interpretable features like names and the use of legal terms. 

{% include figure.liquid path="assets/img/2026-04-27-interp-safety/fig_4.png" class="img-fluid" %}

Does that mean if we applied this to our college admissions example, we could find all of the features that our model considered most important? Sadly, it does not. Interpreting the features the SAE identifies can be difficult (and in fact, recent work <d-cite key="hewitt2025position"></d-cite> proposes we need to understand AI models as having distinct concepts that we humans don’t have!). It’s also not clear how well these representations truly reflect what’s going on inside the LLM. More complex interactions could be obscured by the SAE. In short: SAEs do

In short: SAEs do not provide comprehensive information about an LLM’s dynamics. The good news is that when we are able to interpret the features they locate, we can gain insight into what our model is using for its predictions. We can use these to assess how effectively our model is disentangling traits, e.g., favorite musical artist and gender. Additionally, SAEs allow us to explore concept-based interventions, e.g., by zeroing out neurons in the model associated with a concept we want it to ignore when making a prediction.

## What's Next?

If we had applied all of the methods described in this post, what could we have learned? 

- Making discrete changes to a specific input and measuring the effect on the output score distribution would tell us how impactful certain terms were on our observed outcome.
  - **Example:** we could measure how important swapping favorite artists was vs swapping GPAs.
- Confidence elicitation would give us more information about how certain the model was in its predictions on specific inputs.
  - **Example:** we could estimate how confident the model was scoring each student and route the least confident predictions to a human for validation.
- Probing would let us assess whether our model had internal representations of particular topics we cared about.
  - **Example:** we could train a probe to classify favorite music genres to see how clearly encoded this was within our scoring model.
- Sparse autoencoders would allow us to explore the space of topics especially relevant to the model, provided we could interpret them once we found them. They also let us make interventions on parts of the model associated with these concepts.
  - **Example:** we could train an SAE on our recommendation scoring model and then search its embeddings for identifiable features like music preferences and favorite colors.

Generalizing beyond this example, we present the table below with (non-exhaustive) descriptions of what the methods can and can’t tell us, and how this information might practically be used.

| **Method**      |      **What can it tell us?**      |  **What can’t it tell us?** | **What can we do with this information?**|
| ------------- | -------------  | ------------- | ------------- |
| Exploring discrete changes  | For a **specific** set of inputs and a **specific** set of features, how does changing those features affect a model’s outputs?| What happens if I have [insert input not in the set here] or [insert feature not in the set here]? <br> <br> How do the features (e.g., an artist’s name) relate to higher-level concepts (e.g., gender) within the model? | If we identify features that should have a low (or non-existent) impact on the final prediction (e.g., gender), then we can re-train the model or choose not to deploy it. |
|Confidence elicitation| What is our **estimate** of how confident the model is in its prediction on a given input? | How confident _should_ the model be (i.e., how representative is this input of its training data)? <br> <br> How close is the model’s prediction (independent of confidence) to the behavior that we want it to have? |If our model has low confidence for an input, we can route it to an alternative model (or a person!)|
|Probing | Are we able to find a **specific** concept in a model’s internal representations?|Does the model have an internal representation of this concept? (A negative result doesn’t mean that the model isn’t representing the concept in some way, since we’re reliant on the quality of our classifier).<br><br>Exactly how well is the model encoding this trait? (We can get a relative sense of this, but not an absolute one)<br><br> How is this concept (compared and in addition to other ones) being used to make our model’s final prediction?| If there is a concept that we know the model shouldn’t be using in its final prediction (e.g., if we want it to be agnostic to favorite artist), and we can effectively predict it with a probe, then we can choose not to deploy the model and attempt to train one that does not encode this concept. As a caveat here, we recommend checking out this paper <d-cite key="gonen-goldberg-2019-lipstick-pig"></d-cite>. Attempts to remove bias may end up obscuring it superficially!|
|Sparse autoencoders|What are some **interpretable** concepts that our model has an internal representation of? <br> <br> When (i.e., for which inputs and outputs) are the identified concepts used?| Is the model using [insert concept the SAE didn’t find here]?<br><br>To exactly what degree are the identified concepts A and B entangled?| If we identify interpretable concepts, we can look at which of these were used for particular predictions and update our decisions about deployment accordingly. For example, finding out that our model learned to specifically watch for red car drivers in college admission maybe a bad sign.<br><br>We note that if an SAE does not identify a particular concept, this does not a guarantee that it isn’t being used. The concept could be encoded in a more complex way, and by nature of sparsity, many less-used concepts are being dropped by the SAE (but not by the LLM).|

Yes, there’s a lot we can’t know! Depending on the application, that may be reason enough for not deploying an AI system. However, there is also a lot we can know and, given that information, a lot we can do. 

Since work on identifying circuits <d-cite key="elhage2021mathematical"></d-cite> and causal mechanisms <d-cite key="geiger2021causal"></d-cite> within LLMs is still at the frontier and is not yet actionable under many circumstances, we’re currently limited to assessing concepts, rather than full pathways and processes. There are cases where we’d want to have both. Existing methods in the input/output space give us information about _what_ an LLM is doing, mechinterp approaches tell us more about _why_, but there’s a remaining gap for _how_. For many human processes (like college admissions), we care about all three!

College admissions is only one of the many critical decision-making domains where LLMs are being used. In order to avoid unsafe deployment or deploy as safely as possible, model developers should answer as many questions about their models as they can. This post has covered a few of the many promising methods available. We hope it’s inspired you to take the dive! 