---
layout: distill
title: UnigramLM - An Attempt at Writing the Missing Manual
description:  This post is my attempt to write down the UnigramLM tokenization algorithm cleanly and explicitly because no such derivation appears to exist and I think understanding the theory behind the method could help us improve it. I'll formalize the generative model around which the algorithm is based, derive the EM updates, explain why pruning is needed (and how it's done), and point out the spots where the practical implementation defined by the SentencePiece library diverges from the pretty mathematical models. I hope this post provides a new lens through which to look at the UnigramLM tokenization algorithm while pointing out some interesting potential extensions/revisions to the current implementation.
date: 2026-04-27
future: true
htmlwidgets: true
enable_math: true

# anonymize when submitting
authors:
  - name: Clara Meister
    url: https://cimeister.github.io/
    affiliations:
        name: EPFL

# do not fill this in until your post is accepted and you're publishing your camera-ready post!
# authors:
#   - name: Albert Einstein
#     url: "https://en.wikipedia.org/wiki/Albert\_Einstein"
#     affiliations:
#       name: IAS, Princeton
#   - name: Boris Podolsky
#     url: "https://en.wikipedia.org/wiki/Boris\_Podolsky"
#     affiliations:
#       name: IAS, Princeton
#   - name: Nathan Rosen
#     url: "https://en.wikipedia.org/wiki/Nathan_Rosen"
#     affiliations:
#       name: IAS, Princeton

# must be the exact same name as your blogpost
bibliography: 2026-04-27-unigramlm-manual.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Intro and motivation
  - name: Tokenization Background and Notation
  - name: What you came here for - UnigramLM
    subsections:
      - name: Generative model
      - name: Inference
      - name: Learning Model Parameters
      - name: The UnigramLM Algorithm
        subsections:
          - name: Concise Pseudocode 
  - name: Implementation in the SentencePiece library
  - name: Conclusion
---

## Intro and motivation
*(feel free to [skip](#sec:background) straight to the exposition)*

These days, tokenization is basically synonymous with Byte-pair Encoding (BPE). If you ask someone "do you know how tokenization works?", there's a decent chance you'll get an answer like: "Yeah yeah, I know BPE."  But tokenization != BPE. There are numerous (arguably better motivated) algorithms one could use for segmenting text into tokens. This post focuses on UnigramLM (the SentencePiece "unigram" model), which is a relatively far departure from the BPE approach... 

### Why look at UnigramLM now (and not just "make BPE better")?
Recent work keeps showing that tokenizers themselves can induce [unfairness](https://arxiv.org/abs/2305.15425) and [uneven performance](https://arxiv.org/abs/2305.17179) across languages, dialects, and writing systems. A lot of the community response has (reasonably!) focused on patching BPE: adding constraints, regularizers, or parity-aware merges. Those are valuable, but there's a risk in treating "tokenization = BPE + tweaks" as the whole design space. UnigramLM is a widely deployed alternative (T5, XLNet), and it comes from a fundamentally different modeling viewpoint. Instead of greedily merging pairs to compress text, it posits a probabilistic model and asks: which pieces best explain the corpus? The pieces that survive are those with high likelihood under a (simple) language model. I would a priori guess that choosing pieces to maximize corpus likelihood is more likely to recover recurring morphological patterns and meaningful substructures in a language than choosing pieces purely on the basis of compression; there is empirical evidence suggesting as much ([Bostrom and Durrett, 2020](https://aclanthology.org/2020.findings-emnlp.414/); [Vemula et al., 2025](https://aclanthology.org/2025.ijcnlp-srw.20/)). Lastly, if the fairness problems we're seeing turn out to be partly a consequence of BPE's specific assumptions/inductive biases, then isn't it worth exploring an algorithm built on entirely different foundations? Not only does UnigramLM provide another such option, its probabilistic formulation gives us principled places to add our own biases/constraints: through priors, through the objective, or through the set of segmentations considered during inference. To me, this approach to tokenization sounds at least as promising as coming up with more and more methods to patch BPE's outputs.  

### Why this blog post

With the above motivation in mind, I figured I should actually understand the algorithm. I first went to the [original 2018 paper](https://aclanthology.org/P18-1007/). The paper gives a high-level story of the algorithm but leaves most of the mathematical machinery implicit. So then I went to the SentencePiece repo---the original paper's implementation of the method. Turns out that reverse-engineering [C++ code](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc) into a coherent mathematical derivation isn't so easy. Existing tutorials turned out to not be much help either:


> *The HuggingFace documentation* [on UnigramLM] *describes a
> tokeniser that doesn't exist. It should not be relied on as an
> explanation for UnigramLM, because it doesn't even come close.*  
> –Claude

In short, I couldn't find a single place that spells out UnigramLM's full generative model, why the algorithm is mathematically sound, or how the engineering details (like pruning and vocabulary initialization) fit into that picture. This post is my attempt to provide an approachable but rigorous walkthrough of UnigramLM as a probabilistic model, showing why EM is a reasonable tool here, what the posterior over segmentations actually looks like, and how the SentencePiece implementation approximates or diverges from all of this in practice. If you've ever felt that UnigramLM is "clear enough to use, but not clear enough to explain on a whiteboard," my hope is that this takes you the rest of the way—and maybe even toward extending it. Because at least I think its a pretty cool algorithm that deserves some of BPE's limelight. 

<a id="sec:background"></a>
## Tokenization Background and Notation

So that we're on the same page, let's start with a formal definition of tokenization.

Let $$\mathbf{s}=\langle s_1, s_2,\dots\rangle$$ be a string---a sequence of characters (or bytes) such that $$s_t\in\Sigma$$ for a base alphabet $$\Sigma$$. Let $$\mathcal{V}$$ be a finite set, where each $$v\in\mathcal{V}$$ consists of a sequence of symbols from $$\Sigma\cup\Gamma$$, where $$\Gamma$$ denotes a finite set of reserved symbols (e.g., whitespace markers, start/end tokens, etc.); we refer to $$\mathcal{V}$$ as our **vocabulary** and to $$v$$ as **pieces**.[^2] During tokenization, we wish to convert the sequence of characters/bytes $$\mathbf{s}$$ into a sequence of tokens $$\mathbf{v}=\langle v_1,\dots,v_m\rangle$$, each of which is a piece in the set $$\mathcal{V}$$. We refer to this token sequence as a **segmentation** of $$\mathbf{s}$$, and it can informally be seen as just a different way of representing the original string.

A tokenization algorithm defines a mapping $$h: \Sigma^* \rightarrow \mathcal{V}^*$$ and the method for learning the parameters of this mapping. The application of $$h$$ (which we'll call our **tokenization function** here) to a string is sometimes referred to as inference, although perhaps more commonly people just call this process "tokenizing a string." For example, the byte-pair encoding (BPE) algorithm defines a $$h$$ that is parameterized by a list of *merge* pairs $$\boldsymbol{\mu}=\langle(v_1, v_1'),(v_2, v_2'), \dots \rangle$$ and the algorithm for learning $$\boldsymbol{\mu}$$. At inference, starting from the representation of $$\mathbf{s}$$ as just a sequence of symbols from the base vocabulary $$\Sigma$$, $$h_{\boldsymbol{\mu}}$$ goes through the text $$i=1, \dots \lvert\boldsymbol{\mu}\rvert$$ times. At step $$i$$, it replaces all co-occurrences of the pair $$(v_i, v_i')$$ with a new merged token (typically, of the form $$v_i\circ v_i'$$).[^3]

Importantly, we assume that $$\mathbf{s}$$ can be reconstructed from $$\mathbf{v}$$ via a **detokenization function** $$g: \mathcal{V}^* \rightarrow \Sigma^*$$; often $$g$$ is a simple mapping like string concatenation with some special symbol handling, e.g., $$g(\mathbf{v}) = v_1\circ\dots \circ v_m$$. In what follows, we consider $$g$$ fixed and treat it as part of the model specification. All probabilities over strings and segmentations are defined with respect to this fixed choice of $$g$$. Notably, given just the vocabulary $$\mathcal{V}$$, there are often multiple valid $$\mathbf{v}$$ for which the application of our simple detokenization function $$g$$ would lead to the same $$\mathbf{s}$$. In other words, $$g$$ is generally non-injective. We use $$\mathcal{T}_{\mathcal{V}}(\mathbf{s}) \mathrel{\stackrel{\textnormal{def}}{=}}g^{-1}(\mathbf{s}) = \{\mathbf{v}\in\mathcal{V}^* : g(\mathbf{v}) = \mathbf{s}\}$$ to refer to the set of all valid token sequences that produce $$\mathbf{s}$$, i.e., the set-valued inverse of $$g$$.
<div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">

**Example 1** (A concrete example of the non-injectivity of $$g$$.). *Consider a toy string $$\mathbf{s}= \text{hat}$$ and a small vocabulary $$\mathcal{V}= \{\text{h},\text{a},\text{t},\text{ha},\text{at}\}$$. Under our fixed detokenization function $$g$$ (simple concatenation of token symbol sequences), the set of all valid segmentations of $$\mathbf{s}$$ is*

$$
\begin{aligned}
    \mathcal{T}_{\mathcal{V}}(\mathbf{s})
    =\{
        \langle \text{h}, \text{a}, \text{t} \rangle,
        \langle \text{ha}, \text{t} \rangle,
        \langle \text{h}, \text{at} \rangle\}.
\end{aligned}
$$

*where all three segmentations detokenize to the same string $$\mathbf{s}= \text{hat}$$ under $$g$$.*
</div>

While it might not seem notable, the non-injectivity of $g$ is
actually an interesting property of most tokenization schemes. For one,
it's motivated several variants of different tokenization algorithms in
which the inference rule — the mapping
$h:\Sigma^\*\rightarrow\mathcal{V}^\*$ that selects a particular
element of $\mathcal{T}_{\mathcal{V}}(\mathbf{s})$ — is replaced
or redefined, for example by sampling from a posterior over
segmentations<d-cite key="kudo-2018-subword"></d-cite> or by changing the inference
objective to something like minimizing token sequence length<d-cite key="hofmann-etal-2022-embarrassingly"></d-cite><d-cite key="schmidt-etal-2024-tokenization"></d-cite>. It
also means that we should distinguish between the **canonical tokenization** 
of $\mathbf{s}$, which is $h(\mathbf{s})$, and any other valid segmentation
$\mathbf{v} \in \mathcal{T}\_{\mathcal{V}}(\mathbf{s})$ with
$\mathbf{v} \neq h(\mathbf{s})$,
which are typically called **non-canonical tokenizations**.
The existence of non-canonical
tokenizations has implications for how one should actually compute the
probability of a string under a language model using a given vocabulary. 
See Cao and Rimell (2021)<d-cite key="cao-rimell-2021-evaluate"></d-cite> for a more detailed discussion of
non-canonical tokenizations and why they matter in practice.

## What you came here for - UnigramLM

The UnigramLM tokenization algorithm<d-cite key="kudo-2018-subword"></d-cite> takes a probabilistic-modeling approach to string tokenization. It defines an $$h$$, together with an algorithm for learning its parameters, by treating tokenization as inference in a latent-variable generative model over strings---in particular, a unigram generative model.

**Few Sentence Description of UnigramLM**: UnigramLM is basically what it sounds like: a unigram language model. The only parameters of the tokenization scheme are a unigram probability distribution. When learning the tokenizer, we learn both the vocabulary and piece probabilities of this unigram model that (approximately) maximize corpus log-likelihood. At inference time, given a string, UnigramLM chooses the segmentation (sequence of pieces) that has the highest probability under this learned unigram model. In contrast to BPE's greedy merge story, UnigramLM's behavior is really "whichever segmentation makes the whole corpus most probable under this unigram model wins."

<a id="sec:gen_model"></a>
### Generative model

The UnigramLM tokenization algorithm assumes that each observed string $$\mathbf{s}$$ arises from a latent sequence of tokens $$\mathbf{v}$$, where tokens are drawn independently from a fixed probability distribution, i.e., token sequences are produced by a unigram language model. The data-generating distribution can thus be defined in terms of the unigram probabilities $$\boldsymbol{\phi}\in \Delta^{\lvert\mathcal{V}\rvert - 1}$$, where $$\Delta^{k} = \{\mathbf{x} \in \mathbb{R}^{k+1}_{\geq 0} : \sum_i x_i = 1\}$$ denotes the $$k$$-dimensional probability simplex, i.e., the set of valid probability distributions over $$k+1$$ outcomes. Notably, the idea of treating text segmentation as inference over a unigram model has a longer history than is commonly acknowledged. Bimbot et. al. (1995)<d-cite key="Bimbot95"></d-cite> similarly proposed modeling language as independent draws from a categorical distribution over variable-length units (what they called *multigrams*) and to estimate probabilities via EM with marginalization over all valid segmentations. Kudo (2018) does not reference this earlier line of work, but the core generative assumption (and much of the machinery) is essentially the same.

Before we get to the definition of the data-generating distribution though, we have to establish some other definitions.

**Warning about notation:** To reduce the number of nested subscripts (and other similarly offensive notational choices), I'm going to primarily use random variables to describe this problem. As is standard, uppercase letters will denote random variables (e.g., $$X$$, $$Z$$), and bold uppercase letters will denote sequences of them (e.g., $$\mathbf X$$, $$\mathbf Z$$).

Formally, let $$V$$ be our token-valued random variable: a categorical random variable on $$\mathcal{V}$$ with $$\sum_{v\in\mathcal{V}}P(V=v;\boldsymbol{\phi})=1$$. Occasionally for shorthand, we'll use $$\phi_v= P(V=v;\boldsymbol{\phi})$$ to refer to the unigram probability of the piece $$v$$. Let $$\mathbf{V}$$ be a random variable taking values in the space of token *sequences* $$\mathbf{v}\in \mathcal{V}^*$$. For the distribution of $$\mathbf{V}$$ to be a valid probability distribution on $$\mathcal{V}^*$$, we must specify a length prior, i.e., a random variable $$M$$ on $$\mathbb{N}$$ with $$\sum_{m=0}^\infty P(M=m)=1$$.[^4] The UnigramLM algorithm then assumes token sequence $$\mathbf{v}=\langle v_1,\dots,v_m\rangle$$ are generated as

$$
m\sim M,\quad v_t\stackrel{\text{i.i.d.}}{\sim} {\small\mathrm{Categorical}}(\boldsymbol{\phi}) \quad (t=1,\dots,m)
\tag{1}
$$

We can thus define the distribution of $$\mathbf{V}$$ as

$$
P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi}) \mathrel{\stackrel{\textnormal{def}}{=}} P(M=\lvert\mathbf{v}\rvert)\prod_{t=1}^{\lvert\mathbf{v}\rvert}\boldsymbol{\phi}_{v_t}
\tag{2}
$$

The likelihood of a sequence conditional on a given length $$m$$ is then simply the product of its piece probabilities, i.e., Eq. (2) where the length prior term cancels out:

$$
P(\mathbf{V}=\mathbf{v}\mid M=m;\boldsymbol{\phi}) = \prod_{t=1}^m \boldsymbol{\phi}_{v_t},
\tag{3}
$$

#### A sidebar on the length prior.

The original paper does not make use of the $$M$$ formulation. Rather, it computes token sequence probabilities using Eq. (3). In this respect, one could argue that the name "Unigram LM" is slightly misleading: it doesn't define a valid language model over strings without the length prior. I'll keep using the length prior in my definitions throughout this exposition so its effect on the algorithm is visible. Note that while the parameters of $$P(V;\boldsymbol{\phi})$$ are completely specified by $$\boldsymbol{\phi}$$, this isn't the case with $$P(\mathbf{V};\boldsymbol{\phi})$$, for which the parameters of $$M$$ must also be known to fully specify an actual Unigram LM distribution. I won't add any additional notation to $$P(\mathbf{V};\boldsymbol{\phi})$$ to specify the parameters of $$M$$, though, to avoid clutter and because it's often ignored anyway.
**Potential research direction: Exploring the effect of re-including the length prior in the UnigramLM model.** Given that compression is such a desirable property of tokenization, a parameter that biases segmentations toward shorter lengths seems worth investigating. The length prior is a concrete way to achieve this.

Given the deterministic mapping $$g$$ from tokens to strings, we can derive the distribution over strings---our data-generating distribution---as a pushforward of the distribution over tokens. Let $$\mathbf{S}$$ be a random variable on $$\Sigma^*$$. The following relationship holds:

$$
\begin{aligned}
    P(\mathbf{S}=\mathbf{s};\boldsymbol{\phi}) \mathrel{\stackrel{\textnormal{def}}{=}}\sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})
\end{aligned}
\tag{4}
$$

#### Some useful relationships between $$\mathbf{V}$$ and $$\mathbf{S}$$.

We can see from Eq. (4) that distribution of $$\mathbf{S}$$ is simply the marginal probability distribution over valid segmentations of $$\mathbf{s}$$ under $$\mathcal{V}$$. Applying Bayes' rule then gives us the posterior over segmentations for a fixed $$\mathbf{s}$$:

$$
P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}; \boldsymbol{\phi}) = 
\begin{cases}
    &\frac{P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})}{P(\mathbf{S}=\mathbf{s};\boldsymbol{\phi})} \quad \text{if } \mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s})\\ &0 \quad \quad \text{ otherwise.}
\end{cases}\tag{5}

$$

By just moving some terms in Eq. (5) around, we also get the definition of the joint distribution over strings and token sequences:

$$
P(\mathbf{S}=\mathbf{s}, \mathbf{V}=\mathbf{v};\boldsymbol{\phi}) = P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})\mathbb{1}\{\mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s})\},
$$

<div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">
**Example 2** *(Posterior over segmentations for "hat")*. Continuing from our running example, suppose we have piece probabilities $$\phi_{\text{h}}=0.3$$, $$\phi_{\text{a}}=0.1$$, $$\phi_{\text{t}}=0.25$$, $$\phi_{\text{ha}}=0.2$$, $$\phi_{\text{at}}=0.15$$. Dropping the length prior, the probability of each segmentation is:



$$
\begin{aligned}
P(\langle \text{h}, \text{a}, \text{t} \rangle;\boldsymbol{\phi}) &= 0.3 \times 0.1 \times 0.25 = 0.0075\\
P(\langle \text{ha}, \text{t} \rangle;\boldsymbol{\phi}) &= 0.2 \times 0.25 = 0.05\\
P(\langle \text{h}, \text{at} \rangle;\boldsymbol{\phi}) &= 0.3 \times 0.15 = 0.045
\end{aligned}
$$
​


The marginal probability of the string is $$P(\mathbf{S}=\text{hat};\boldsymbol{\phi}) = 0.0075 + 0.05 + 0.045 = 0.1025$$. 




Applying Eq. (5), the posterior is:


$$
\begin{aligned}
P(\langle \text{h}, \text{a}, \text{t} \rangle \mid \text{hat};\boldsymbol{\phi}) &= 0.0075 / 0.1025 \approx 0.073\\
P(\langle \text{ha}, \text{t} \rangle \mid \text{hat};\boldsymbol{\phi}) &= 0.05\phantom{00} / 0.1025 \approx 0.488\\
P(\langle \text{h}, \text{at} \rangle \mid \text{hat};\boldsymbol{\phi}) &= 0.045\phantom{0} / 0.1025 \approx 0.439
\end{aligned}
$$


Two things to notice. First, the three-piece segmentation $$\langle \text{h}, \text{a}, \text{t} \rangle$$ gets very little posterior mass even though each of its pieces is individually common. This should make intuitive sense: the product of three probabilities is penalized relative to two. Second, the posterior is sensitive to the parameters: if we were to increase $$\phi_{\text{at}}$$ relative to $$\phi_{\text{ha}}$$, the ranking of the two-piece segmentations would flip. We will revisit this example when we discuss inference and EM.
</div>

### Inference

For the moment, let's assume that we know $$\boldsymbol{\phi}$$, or at least have estimates for these parameters. At inference time (i.e., when segmenting text into tokens), the UnigramLM tokenization algorithm aims to find the most likely segmentation of $$\mathbf{s}$$ into tokens $$\mathbf{v}= \langle v_1, v_2, \dots\rangle$$ under the generative model (defined above) with these parameters. To this end, it uses a Viterbi-style algorithm:

$$
\begin{aligned}
h_{\boldsymbol{\phi}}(\mathbf{s})&= \arg\max_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}; \boldsymbol{\phi})\\ 
&= \arg\max_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})\\ 
&= \arg\max_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} P(M=\lvert\mathbf{v}\rvert)\prod_{t=1}^{\lvert\mathbf{v}\rvert}\boldsymbol{\phi}_{v_t}\\ 
&\overset{?}{=} \arg\max_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} \prod_{t=1}^{\lvert\mathbf{v}\rvert}\boldsymbol{\phi}_{v_t}
\end{aligned}
\tag{6}
$$

where the second line follows from the relationship in Eq. (5) ($$P(\mathbf{S}=\mathbf{s};\boldsymbol{\phi})$$ does not depend on $$\mathbf{v}$$ and so it doesn't affect the argmax).

<div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">
**Example 3** *(Viterbi inference on "hat")*. Using the same parameters as Example 2, the Viterbi segmentation is the one with the highest posterior probability, or equivalently, the highest 
$$P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})$$: 

$$h_{\boldsymbol{\phi}}(\text{hat}) = \arg\max\{0.0075,\; 0.05,\; 0.045\} = \langle \text{ha}, \text{t} \rangle$$

Now suppose we change the parameters so that $$\phi_{\text{at}} = 0.25$$ and $$\phi_{\text{t}} = 0.15$$ (swapping the probabilities of $$\text{t}$$ and $$\text{at}$$), keeping everything else the same. The segmentation probabilities become:

$$
\begin{aligned}
P(\langle \text{h}, \text{a}, \text{t} \rangle;\boldsymbol{\phi}') &= 0.3 \times 0.1 \times 0.15 = 0.0045\\
P(\langle \text{ha}, \text{t} \rangle;\boldsymbol{\phi}') &= 0.2 \times 0.15 = 0.03\\
P(\langle \text{h}, \text{at} \rangle;\boldsymbol{\phi}') &= 0.3 \times 0.25 = 0.075
\end{aligned}
$$

The Viterbi segmentation flips to $$\langle \text{h}, \text{at} \rangle$$. This illustrates the core property of UnigramLM inference: the tokenization of a string is entirely determined by the piece probabilities, and even small parameter changes can alter which segmentation wins. It also shows why learning good $$\boldsymbol{\phi}$$ matters—the parameters don't just score segmentations, they effectively choose them.
</div>

#### Another sidebar on the length prior.

As we can see in Eq. 6, the length prior ($$M$$) is part of the posterior distribution and should thus affect the Viterbi segmentation; intuitively speaking, it biases the distribution towards token sequences of certain lengths.

<div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">
**Example 4** (Effect of the length prior on Viterbi segmentation). *Suppose $$\mathbf{s}$$ admits two segmentations $$\mathbf{v}^{(1)}$$ and $$\mathbf{v}^{(2)}$$ with $$\lvert\mathbf{v}^{(1)}\rvert = 1$$ and $$\lvert\mathbf{v}^{(2)}\rvert = 3$$, and that the two have identical unigram products $$\prod_t \phi_{v_t}$$. Without a length prior, these segmentations tie. But if $$P(M=1) \gg P(M=3)$$, the model strongly prefers $$\mathbf{v}^{(1)}$$. This illustrates that the length prior can have a non-trivial effect on the inference result, which is why dropping it is a modeling choice worth being explicit about.*
</div>


As hinted at earlier, SentencePiece (and all other implementations of UnigramLM that I've seen) drop the length prior term. Unless otherwise specified, when talking about inference, we will assume no length prior is used in segmentation probability computations for faithfulness to the original algorithm.

The true parameters of the generative process $$\boldsymbol{\phi}$$ are unknown, however; this includes both the piece probabilities $$\phi_v$$ and the underlying vocabulary $$\mathcal{V}$$ over which they are defined. The UnigramLM tokenization algorithm (described next) proposes a method for coming up with an estimate of these parameters from text data.

### Learning Model Parameters

Maximum likelihood estimation (MLE)---a standard approach to estimating model parameters---aims to find the model parameters that maximize the log-likelihood of our data. Under the UnigramLM assumptions about the generative process of strings, each observed string $$\mathbf{s}$$ was produced by some token sequence $$\mathbf{v}$$. But we never actually observe $$\mathbf{v}$$. It is a latent variable: the "hidden" segmentation that gave rise to the string we see. In EM terminology, the observed data augmented with these latent variables ($$\mathcal{X} = \{(\mathbf{s}_i,\mathbf{v}_i)\}_{i=1}^K$$) is referred to as the **complete** dataset. If we had access to this complete data (i.e., if someone told us the ground-truth segmentation for every string), estimating $$\boldsymbol{\phi}$$ would be straightforward---we'd just count tokens and normalize as if performing MLE for a standard categorical distribution. But this isn't the case. To arrive at our ultimate method, we'll start from the complete-data log likelihood, which is defined as:

$$
\begin{aligned}
    \mathcal{L}(\mathcal{X}; \boldsymbol{\phi}) &\mathrel{\stackrel{\textnormal{def}}{=}}\log\prod_{i=1}^KP(\mathbf{S}=\mathbf{s}_i, \mathbf{V}=\mathbf{v}_i;\boldsymbol{\phi})\\ 
    &= \sum_{i=1}^K\log P(\mathbf{S}=\mathbf{s}_i, \mathbf{V}=\mathbf{v}_i;\boldsymbol{\phi})
\end{aligned}
\tag{7}
$$

Of course, we don't have the complete data---we only observe the strings $$\mathbf{s}_i$$, not the segmentations $$\mathbf{v}_i$$ that produced them. For any given string, the true segmentation could be any element of $$\mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)$$, and we don't even know $$\mathcal{V}$$. So Eq. (7) is not something we can optimize directly. What we *can* work with is the observed-data log-likelihood: the probability of the strings alone, obtained by marginalizing over all possible latent segmentations. This is the quantity we'll try to maximize instead, and it's defined using the marginal from Eq. (4):

$$
\begin{aligned}
    \mathcal{L}(\mathcal{C}; \boldsymbol{\phi}) &\mathrel{\stackrel{\textnormal{def}}{=}}\log\prod_{i=1}^KP(\mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi})\\ 
    &= \log\prod_{i=1}^K\sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)} P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})
    \\ 
    &= \sum_{i=1}^K \log\sum_{\mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)} P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})
\end{aligned}
\tag{8}
$$

where $$\mathcal{C}= \{\mathbf{s}\mid \mathbf{s}, \_ \in \mathcal{X} \}$$ is simply our observed set of strings, i.e., our corpus. Unfortunately, Eq. (8) is a difficult quantity to maximize directly due to the log--sum structure. Luckily, the expectation-maximization (EM) algorithm provides us a route for working with this situation.

### The Expectation-Maximization Algorithm in the Context of UnigramLM

EM was designed for exactly the use case where wish to get MLE estimates for a data-generating process in which only part of the data is unobserved.

**TL;DR of the application of the EM algorithm to UnigramLM**: EM is an iterative algorithm for approximating MLE estimates. The E step computes the expected complete data log-likelihood under current beliefs about model parameters (in our case, $$\boldsymbol{\phi}^{(n)}$$); this quantity serves as a proxy for the observed data log-likelihood, which is much harder to compute directly. The M step then solves for the free parameters (in our case, $$\boldsymbol{\phi}$$) that maximize this quantity, and then updates our current beliefs to the new quantity.

In more detail now: the EM algorithm uses Jensen's inequality to relate the *expected value* of the complete data log-likelihood to the *observed* data log-likelihood, i.e., relating the expected value of Eq. (7) to Eq. (8). This is exactly the connection made by Kudo (2018)<d-cite key="kudo-2018-subword"></d-cite> (even if not explicitly) when introducing their algorithm for approximating the parameters $$\boldsymbol{\phi}$$.

**Expected complete-data log-likelihood under observed data and current parameters.**

Let $$\boldsymbol{\phi}^{(n)}$$ denote our current belief about what the unigram parameters might be (more discussion on how we can initialize this distribution coming up!). For now, we will assume that the vocabulary is fixed. These random variables adhere to our original definitions in [4.1](#sec:gen_model). Note that when we use simply $$\boldsymbol{\phi}$$, we are referring to the distributions (and corresponding random variables) induced by a generic $$\boldsymbol{\phi}$$; these are the entities for which our parameters are free variables that we are optimizing.

The expected complete data log-likelihood under $$\boldsymbol{\phi}^{(n)}$$---which we denote as $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$---follows simply from taking the expectated value of Eq. (7), given our observed data $$\mathcal{C}$$ and our current model parameters $$\boldsymbol{\phi}^{(n)}$$, i.e., the expected value under the posterior $$\mathbf{V}\mid \mathbf{S};\boldsymbol{\phi}^{(n)}$$.

$$
\begin{aligned}
\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})
&\mathrel{\stackrel{\textnormal{def}}{=}} \mathop{\mathrm{\mathbb{E}}}
\big[\mathcal{L}(\mathcal{X}; \boldsymbol{\phi}) \mid \mathcal{C}, \boldsymbol{\phi}^{(n)}\big]\\ 
&= \underset{ \mathbf{V}\mid \mathbf{S};\boldsymbol{\phi}^{(n)}}{\mathop{\mathrm{\mathbb{E}}}}\big[\sum_{i=1}^K \log P(\mathbf{S}, \mathbf{V};\boldsymbol{\phi})\mid \mathcal{C}\big]\\ 
    &= \sum_{i=1}^K\underset{\mathbf{v}\sim\mathbf{V}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)}}{\mathop{\mathrm{\mathbb{E}}}}\big[\log P(\mathbf{S}=\mathbf{s}_i, \mathbf{V}=\mathbf{v};\boldsymbol{\phi})\big]
\end{aligned}
$$

In words, we can think of $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$ as the expected complete data log-likelihood where the (latent) segmentations are induced by the posterior with parameters $$\boldsymbol{\phi}^{(n)}$$, while the log-likelihood inside is evaluated using the candidate parameters $$\boldsymbol{\phi}$$.

Now we will show how this quantity relates to the observed data log-likelihood.

#### Observed data log-likelihood and Jensen's inequality.

We start with a reminder of Jensen's inequality, applied to our definition of $$P(\mathbf{S}=\mathbf{s};\boldsymbol{\phi})$$. For any valid distribution probability $$P(\mathbf{V}=\mathbf{v})$$, Jensen's inequality tells us

$$
\begin{aligned}
\log P(\mathbf{S}=\mathbf{s};\boldsymbol{\phi})
&= \log \sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} P(\mathbf{V}=\mathbf{v})\,\frac{P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})}{P(\mathbf{V}=\mathbf{v})}\\ 
&\ge \sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s})} P(\mathbf{V}=\mathbf{v})\,\log \frac{P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})}{P(\mathbf{V}=\mathbf{v})}
\end{aligned}
$$

If we choose $$P(\mathbf{V}=\mathbf{v})$$ to be $$P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}= \mathbf{s};\boldsymbol{\phi}^{(n)})$$---the posterior under our current parameter beliefs for a fixed $$\mathbf{s}$$---and apply this to our definition of the observed data log-likelihood from Eq. (8), we get

$$
\begin{aligned}
\mathcal{L}&(\mathcal{C};\boldsymbol{\phi})= \sum_{i=1}^K \log\sum_{\mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)} P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})\\ 
&\ge \sum_{i=1}^K\sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)}P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})
\big[\log P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})-\log P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})\big]\\ 
&= \sum_{i=1}^K\sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)}P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})
\log P(\mathbf{S}=\mathbf{s}_i, \mathbf{V}=\mathbf{v};\boldsymbol{\phi})\nonumber\\ 
& \qquad\qquad\qquad\qquad-\sum_{i=1}^K\sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)}P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})\log P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})\\
&= \underbrace{\sum_{i=1}^K\underset{\mathbf{v}\sim \mathbf{V}\,\mid\, \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)}}{\mathop{\mathrm{\mathbb{E}}}}\big[\log P(\mathbf{S}=\mathbf{s}_i, \mathbf{V}=\mathbf{v};\boldsymbol{\phi})\big]}_{\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})} + \sum_{i=1}^K\underbrace{\underset{\mathbf{v}\sim \mathbf{V}\,\mid\, \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)}}{\mathop{\mathrm{\mathbb{E}}}}\big[\log P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})\big]}_{\mathrm{H}\big(\mathbf{V}\,\mid \,\mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)}\big)}\\ 
&\geq \mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})
\end{aligned}
\tag{9}
$$

Note that when going from the second to third lines in Eq. (9), we make use of the fact that for any $$\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)$$ we have $$P(\mathbf{S}=\mathbf{s}_i, \mathbf{V}=\mathbf{v};\boldsymbol{\phi}) = P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})$$ by definition. Then, we're simply using the equivalence of these values with the definitions of expected values and (Shannon) entropy, respectively.

Eq. (9) is typically referred to as the evidence lower bound (ELBO)---a proxy objective that is often used in machine learning. For example, it's used for training variational autoencoders, where it provides a tractable lower bound on the intractable log-likelihood of the data under a latent-variable model. In the case of EM, we go one step further and use one of the components of the ELBO as our proxy objective for observed data log-likelihood: the expected complete data log-likelihood $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$. And this is the basis the EM algorithm, which iteratively updates $$\boldsymbol{\phi}$$ by choosing the value of it that maximizes $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$ until convergence.

 After all those derivations, it's helpful to look at our ideal and actual objectives side-by-side, just to see what the difference is:

$$
\underbrace{\sum_{i=1}^K \log\sum_{\mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)} P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})}_{\text{objective we'd ideally be maximizing }} \qquad\qquad \underbrace{\sum_{i=1}^K\sum_{\mathbf{v}\in \mathcal{T}_{\mathcal{V}}(\mathbf{s}_i)}P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)})
\log P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})}_{\text{objective we maximize iteratively with EM}}
$$

And while the solution given by EM should converge to the solution we'd get from maximizing our ideal object, this isn't the case with the UnigramLM algorithm since it's not pure EM in isolation. I explain this next.



### The UnigramLM Algorithm

The UnigramLM algorithm is typically seen as a "simple" application of EM. This, however, is not exactly the case. Importantly, EM assumes that the support of the distribution whose parameters we're trying to estimate is known (and fixed), i.e., that we know $$\mathcal{V}$$. But, as discussed earlier, we don't know $$\mathcal{V}$$! The UnigramLM algorithm addresses this by beginning with an intentionally overcomplete initial vocabulary and progressively reducing it through a heuristic pruning step, which is done *after* an iteration of the standard E-step and M-step, throughout which $$\mathcal{V}$$ is held fixed. In short, as the algorithm iteratively re-estimates the model parameters, it gradually shrinks $$\mathcal{V}$$ toward the desired final size by removing pieces that are seemingly unimportant for achieving good corpus log-likelihood. You can think of this as putting your vocabulary on a strict likelihood-based diet: pieces that don't contribute enough to explaining the data get gently but firmly removed.

Where its necessary, to make this dependence explicit, we will use $$\mathcal{V}_n$$ to denote the current vocabulary. To reduce notational clutter, in defining the below algorithm, we'll use just $$\mathcal{V}$$; at step $$n$$ of the algorithm, you can assume $$\mathcal{V}=\mathcal{V}_n$$ (and that all random variables are defined over $$\mathcal{V}_n$$) unless otherwise stated.

If you'd like to look at a trimmed down version of the pseudocode, you can [skip to the end](#sec:pseudocode)

1. **Initialization:** Define an initial vocabulary $$\mathcal{V}_0$$. This could be something like all possible substrings of $$\mathcal{C}$$, subject to a maximum length constraint.[^5] Initialize $$\boldsymbol{\phi}^{(0)}$$ by some heuristic: the simplest would be uniform initialization, i.e., all pieces are assigned probability $$1/\lvert\mathcal{V}_0\rvert$$.

2. **Perform EM for $$n=1, \dots N$$ iterations or until piece probability estimates converge:**

    i. **E-step** (Expected data log-likelihood computation): The E-step in EM is for computing the expected complete data log-likelihood under our current parameter beliefs $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$. It turns out that expected token counts are a sufficient statistic for the M-step objective in this case, and so our problem boils down to computing expected token counts under $$\boldsymbol{\phi}^{(n)}$$. To see why this is the case... First, we define the count function on token sequences as
    
    $$
    c_v(\mathbf{v}) \mathrel{\stackrel{\textnormal{def}}{=}}\sum_{t=1}^{\lvert\mathbf{v}\rvert}\mathbb{1}\{v_t= v\}
    \tag{10}
    $$
    
    Then, note that for any valid $$\mathbf{s},\mathbf{v}$$ such that $$\mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s})$$, we can write
    
    $$
    \begin{aligned}
    \log P(\mathbf{S}=\mathbf{s}, \mathbf{V}=\mathbf{v};\boldsymbol{\phi})&=\log P(\mathbf{V}=\mathbf{v};\boldsymbol{\phi})\\ 
    &=\log P(M=\lvert\mathbf{v}\rvert)+\sum_{t=1}^{\lvert\mathbf{v}\rvert}\log \boldsymbol{\phi}_{v_t}\\ 
    &=\log P(M=\lvert\mathbf{v}\rvert)+ \sum_{v\in\mathcal{V}} c_v(\mathbf{v})\log \boldsymbol{\phi}_{v}
    \end{aligned}
    $$
    
    Substituting these relationships into our definition of $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$ and using the linearity of expectations rule, we get
    
    $$
    \begin{aligned}
    \mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})
    = \underbrace{\sum_{i=1}^K\underset{\mathbf{v}\sim \mathbf{V}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)}}{\mathop{\mathrm{\mathbb{E}}}}[\log P(M=\lvert\mathbf{v}\rvert)]}_{\text{constant in }\boldsymbol{\phi}}
    +\sum_{i=1}^K\sum_{v\in\mathcal{V}}
    \underbrace{\underset{\mathbf{v}\sim \mathbf{V}\mid \mathbf{S}=\mathbf{s}_i;\boldsymbol{\phi}^{(n)}}{\mathop{\mathrm{\mathbb{E}}}\left[ c_v(\mathbf{v})\right]}}_{\mathrel{\stackrel{\textnormal{def}}{=}}\widetilde{c}_v(\mathbf{s}_i;\boldsymbol{\phi}^{(n)})}\log \boldsymbol{\phi}_{v}
    \end{aligned}
    \tag{11}
    $$
    
    where $$\widetilde{c}_v(\mathbf{s};\boldsymbol{\phi}^{(n)})$$ are simply expected token counts under unigram model parameters $$\boldsymbol{\phi}^{(n)}$$, which can be computed as $$\widetilde{c}_v(\mathbf{s};\boldsymbol{\phi}^{(n)})= \sum_{\mathbf{v}\in\mathcal{T}_{\mathcal{V}}(\mathbf{s})} c_v(\mathbf{v}) P(\mathbf{V}=\mathbf{v}\mid \mathbf{S}=\mathbf{s};\boldsymbol{\phi}^{(n)})$$. Lastly, if we define the corpus-level expected counts as
    
    $$
    \widehat{c}_v(\mathcal{C};\boldsymbol{\phi}) \mathrel{\stackrel{\textnormal{def}}{=}} \sum_{\mathbf{s}\in\mathcal{C}} \widetilde{c}_v(\mathbf{s};\boldsymbol{\phi})
    \tag{12}
    $$
    
    and substitute them into our expansion in Eq. 11, then the equality reduces to
    
    $$
    \mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})
    = \text{const} + \underbrace{\sum_{v\in\mathcal{V}}\widehat{c}_v(\mathcal{C};\boldsymbol{\phi}^{(n)})\log \boldsymbol{\phi}_{v}}_{\mathrel{\stackrel{\textnormal{def}}{=}}\bar{\mathcal{Q}}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})}
    \tag{13}
    $$
    
    where we have added the definition of $$\bar{\mathcal{Q}}$$ (simply $$\mathcal{Q}$$ without the "$$\mathrm{const}$$" term) since it will be useful later. From the above, we can see that the posterior expected counts are sufficient statistics for the M-step objective $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$.
    
    In practice, the per-string expected counts $$\widetilde{c}_v(\mathbf{s};\boldsymbol{\phi})$$ can be computed efficiently using a forward--backward dynamic program defined over the segmentation lattice induced by $$\mathcal{T}_{\mathcal{V}}(\mathbf{s})$$. In words, this lattice forms a directed acyclic graph: nodes correspond to positions in the string and edges originating from the nodes correspond to tokens $$v\in \mathcal{V}$$ that can begin at that position and end at another (i.e., pieces whose symbol sequences match the substring). Each edge is weighted by the token's probability under the current parameters, $$\phi^{(n)}_v$$. Valid paths in this graph correspond to a valid segmentation of $$\mathbf{s}$$. The forward--backward algorithm then marginalizes over all valid paths in this graph to compute the posterior probability of each token's occurrence, from which the expected counts follow.
    
    A somewhat interesting observation is that this method of getting counts uses an inference procedure that is different from what is done when actually tokenizing text. In the latter case, only the maximum probability segmentation is ultimately used. Here, though, we consider all segmentations of a $$\mathbf{s}$$ that have non-zero probability, weighting the token counts from this segmentation (token sequence) by the probability of the segmentation under our current parameters $$\boldsymbol{\phi}^{(n)}$$.  **Another sidebar on your favorite topic: the length prior.** This is where a length prior *could* have an effect on the model parameters we learn, as it affects the probabilities of the segmentations.  But the term is often never actually used in the model definition. **Potential research direction: Exploring effect of difference between the segmentation strategy during training and inference.** This method of getting counts considers all valid segmentations of $$\mathbf{s}$$ that have non-zero probability under the current model parameters. In contrast, at inference, we only consider the maximum probability segmentation. The original UnigramLM paper actually proposes an inference strategy that's much more aligned with training: sampling tokenizations from the posterior. But nowadays, everyone uses the Viterbi version of inference. Looking at the effects of this could be interesting. For example, soft/expected representations computed over the distribution of segmentations could give benefits similar to dropout or data augmentation, especially for low-resource languages or noisy text.

    To make the forward–backward procedure concrete, we walk through the forward pass on our running example.

    <div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">
    **Example 5** *(Forward algorithm on the "hat" lattice)*. We illustrate how the forward algorithm computes the marginal $$P(\mathbf{S}=\text{hat};\boldsymbol{\phi}^{(0)})$$.
    
    ***Lattice***. The segmentation lattice for $$\text{hat}$$ is a directed acyclic graph with four nodes---one per character boundary (positions 0, 1, 2, 3)---and five edges, one per vocabulary piece that matches a substring

    <figure id="fig:lattice" style="text-align: center; margin: 2em auto;">
    <svg viewBox="0 0 520 180" xmlns="http://www.w3.org/2000/svg" 
        style="max-width: 520px; width: 100%; font-family: 'Computer Modern', Georgia, serif;">
    
    <!-- Nodes -->
    <circle cx="60"  cy="90" r="22" fill="white" stroke="#333" stroke-width="2"/>
    <text x="60"  y="96" text-anchor="middle" font-size="16" fill="#333">0</text>
    
    <circle cx="200" cy="90" r="22" fill="white" stroke="#333" stroke-width="2"/>
    <text x="200" y="96" text-anchor="middle" font-size="16" fill="#333">1</text>
    
    <circle cx="340" cy="90" r="22" fill="white" stroke="#333" stroke-width="2"/>
    <text x="340" y="96" text-anchor="middle" font-size="16" fill="#333">2</text>
    
    <circle cx="460" cy="90" r="22" fill="white" stroke="#333" stroke-width="2"/>
    <text x="460" y="96" text-anchor="middle" font-size="16" fill="#333">3</text>

    <!-- Character labels below nodes -->
    <text x="130" y="140" text-anchor="middle" font-size="16" fill="#888">h</text>
    <text x="270" y="140" text-anchor="middle" font-size="16" fill="#888">a</text>
    <text x="400" y="140" text-anchor="middle" font-size="16" fill="#888">t</text>
    
    <!-- Straight edges (single-character pieces) -->
    <!-- h: 0→1 -->
    <line x1="82" y1="90" x2="178" y2="90" stroke="#4A90D9" stroke-width="2" marker-end="url(#arrowBlue)"/>
    <text x="130" y="80" text-anchor="middle" font-size="14" font-weight="bold" fill="#4A90D9">h</text>
    
    <!-- a: 1→2 -->
    <line x1="222" y1="90" x2="318" y2="90" stroke="#4A90D9" stroke-width="2" marker-end="url(#arrowBlue)"/>
    <text x="270" y="80" text-anchor="middle" font-size="14" font-weight="bold" fill="#4A90D9">a</text>
    
    <!-- t: 2→3 -->
    <line x1="362" y1="90" x2="438" y2="90" stroke="#4A90D9" stroke-width="2" marker-end="url(#arrowBlue)"/>
    <text x="400" y="80" text-anchor="middle" font-size="14" font-weight="bold" fill="#4A90D9">t</text>
    
    <!-- Arc edges (multi-character pieces) -->
    <!-- ha: 0→2 -->
    <path d="M 80,78 Q 210,15 320,78" fill="none" stroke="#4A90D9" stroke-width="2" marker-end="url(#arrowRed)"/>
    <text x="200" y="38" text-anchor="middle" font-size="14" font-weight="bold" fill="#4A90D9">ha</text>
    
    <!-- at: 1→3 -->
    <path d="M 220,78 Q 340,15 442,78" fill="none" stroke="#4A90D9" stroke-width="2" marker-end="url(#arrowGreen)"/>
    <text x="335" y="38" text-anchor="middle" font-size="14" font-weight="bold" fill="#4A90D9">at</text>
    
    <!-- Arrowhead markers -->
    <defs>
        <marker id="arrowBlue" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#4A90D9"/>
        </marker>
        <marker id="arrowRed" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#4A90D9"/>
        </marker>
        <marker id="arrowGreen" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
        <polygon points="0 0, 8 3, 0 6" fill="#4A90D9"/>
        </marker>
    </defs>
    </svg>
    <figcaption style="margin-top: 0.5em; font-size: 0.9em; color: #555;">
    <b>Figure 1:</b> Segmentation lattice for the string "hat" with 
    vocabulary {h, a, t, ha, at}. Nodes represent character boundaries; 
    edges represent vocabulary pieces. Each source-to-sink path is a valid segmentation.
    </figcaption>
    </figure>
    Each source-to-sink path through this graph corresponds to exactly one valid segmentation. The path $$0 \to 1 \to 2 \to 3$$ corresponds to $$\langle \text{h}, \text{a}, \text{t} \rangle$$, the path $$0 \to 2 \to 3$$ to $$\langle \text{ha}, \text{t} \rangle$$, and $$0 \to 1 \to 3$$ to $$\langle \text{h}, \text{at} \rangle$$.
    
    
    Working with the lattice from Figure 1, we assume uniform parameters $$\phi^{(0)}_v = 0.2$$ for all $$v$$. 
    The forward variable $$\alpha(j)$$ accumulates the total probability of all paths from position 0 to position $$j$$. We initialize $$\alpha(0) = 1$$ and work left to right. At each position, we sum over all edges that arrive there:
    
    $$\begin{aligned}
    \alpha(1) &= \alpha(0)\cdot\phi_{\text{h}} = 1 \times 0.2 = 0.2\\
    \alpha(2) &= \alpha(0)\cdot\phi_{\text{ha}} + \alpha(1)\cdot\phi_{\text{a}} = 0.2 + 0.04 = 0.24\\
    \alpha(3) &= \alpha(1)\cdot\phi_{\text{at}} + \alpha(2)\cdot\phi_{\text{t}} = 0.04 + 0.048 = 0.088
    \end{aligned}
    $$

    The final value $$\alpha(3) = 0.088$$ is the marginal $$P(\mathbf{S}=\text{hat};\boldsymbol{\phi}^{(0)})$$---the sum of all three segmentation probabilities. The forward algorithm obtains this without enumerating segmentations explicitly; it works in time linear in the number of edges.
    A symmetric backward pass computes $$\beta(j)$$, the total probability of all paths from position $$j$$ to the end. Combining both, the posterior probability that a specific edge (piece $$v$$ spanning positions $$i\!\to\!j$$) is used in a random segmentation is:

    $$
    P(\text{edge } i\!\to\!j \text{ used}) = \frac{\alpha(i)\cdot\phi_v\cdot\beta(j)}{\alpha(3)}
    $$
    These edge posteriors are exactly the expected counts $$\widetilde{c}_v$$ that feed into the M-step. For instance, the expected count for $$\text{ha}$$ (the edge $$0\!\to\!2$$) is $$\alpha(0)\cdot\phi_{\text{ha}}\cdot\beta(j)\,/\,\alpha(3)$$. When computed for all edges, the resulting values match the expected counts in Example 3.
    </div>

    ii. **M-step** (maximize $$\boldsymbol{\phi}$$ and update $$\boldsymbol{\phi}^{(n)}$$): In the M-step, we want to maximize $$\mathcal{Q}(\boldsymbol{\phi};\boldsymbol{\phi}^{(n)})$$ with respect to $$\boldsymbol{\phi}$$ subject to these parameters giving us a valid probability distribution, i.e., $$\sum_{v\in\mathcal{V}}\phi_v=1$$ and $$\phi_v\ge 0$$. Subbing in the relationship established in Eq. 13, this actually boils down to a relatively simple problem: finding the $$\boldsymbol{\phi}$$ that maximizes the probability of having observed the expected counts that we got from the segmenting the corpus according to our prior model parameter beliefs:
    
    $$
    \begin{aligned}
    \max_{\boldsymbol{\phi}}&\sum_{v\in\mathcal{V}}\widehat{c}_v(\mathcal{C};\boldsymbol{\phi}^{(n)})\log \boldsymbol{\phi}_{v}\\ 
    &\text{s.t.}\quad
    \sum_{v\in\mathcal{V}}\phi_v=1,\phi_v\ge 0
    \end{aligned}
    $$
    
    The solution (normalized expected counts) is very recognizable, as it is essentially the same as the MLE for a standard multinomial distribution, albeit using expected counts rather than pure counts:
    
    $$
    \phi^{(n+1)}_v
    =
    \frac{\widehat{c}_v(\mathcal{C};\boldsymbol{\phi}^{(n)})}
    {\sum_{v'\in\mathcal{V}}\widehat{c}_{v'}(\mathcal{C};\boldsymbol{\phi}^{(n)})}.
    \tag{14}
    $$
    
    The length-prior term is constant in $$\boldsymbol{\phi}$$ and does not alter the update (for fixed $$\mathcal{V}$$).

    <div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">
    **Example 6** (One EM iteration on a single-string corpus). We return to the 'hat' vocabulary from our running example. We now run one full EM iteration starting from uniform initialization, i.e., $$\phi^{(0)}_v = 0.2$$ for all $$v \in \mathcal{V}$$; for exemplary purposes, let's assume our entire corpus $$\mathcal{C}$$ consists of just this one string. Recall the three valid segmentations of $$\mathbf{s}$$:

    $$
    \mathcal{T}_{\mathcal{V}}(\mathbf{s})
    =\{
        \langle \text{h}, \text{a}, \text{t} \rangle,
        \langle \text{ha}, \text{t} \rangle,
        \langle \text{h}, \text{at} \rangle\}.
    $$

    ***E-step.*** We first compute each segmentation's probability (dropping the length prior):

    $$
    \begin{aligned}
    P(\langle \text{h}, \text{a}, \text{t} \rangle;\boldsymbol{\phi}^{(0)}) &= 0.2^3 = 0.008\\
    P(\langle \text{ha}, \text{t} \rangle;\boldsymbol{\phi}^{(0)}) &= 0.2^2 = 0.04\\
    P(\langle \text{h}, \text{at} \rangle;\boldsymbol{\phi}^{(0)}) &= 0.2^2 = 0.04
    \end{aligned}
    $$

    The marginal is $$P(\mathbf{S}=\text{hat};\boldsymbol{\phi}^{(0)}) = 0.008 + 0.04 + 0.04 = 0.088$$. The posterior probabilities are thus:
    
    $$
    \begin{aligned}
    P(\langle \text{h}, \text{a}, \text{t} \rangle \mid \text{hat}) &\approx 0.091\\
    P(\langle \text{ha}, \text{t} \rangle \mid \text{hat}) &\approx 0.454\\
    P(\langle \text{h}, \text{at} \rangle \mid \text{hat}) &\approx 0.454
    \end{aligned}
    $$

    Note that the two shorter segmentations together account for over 90% of the total probability of $$\text{hat}$$, i.e., the posterior mass. Now we compute expected counts by weighting each segmentation's token counts by its posterior probability. For the token $$\text{h}$$, because it appears once in the segmentation $$\langle \text{h,a,t} \rangle$$ and once in $$\langle \text{h,at} \rangle$$, we get  $$\widehat{c}_{\text{h}}(\mathcal{C}; \boldsymbol{\phi}^{(0)}) = 1\times0.091+1\times0.454=0.5451$$ 


    ***M-step.*** Normalizing the expected counts (sum = 2.091) gives the updated parameters:

    $$\phi^{(1)}_{\text{h}} \approx 0.261, \quad
    \phi^{(1)}_{\text{a}} \approx 0.044, \quad
    \phi^{(1)}_{\text{t}} \approx 0.261, \quad
    \phi^{(1)}_{\text{ha}} \approx 0.217, \quad
    \phi^{(1)}_{\text{at}} \approx 0.217
    $$

    Because the piece $$\text{a}$$ only appears in the three-piece segmentation (which had low posterior probability), we see that its estimated probability decreases substantially. Meanwhile, $$\text{h}$$ and $$\text{t}$$ are boosted because they participate in multiple high-probability segmentations. 
    </div>

    iii. **Pruning:** EM alone (the above two steps) doesn't change the vocabulary. It only re-estimates piece probabilities over a fixed $$\mathcal{V}$$. But the initial vocabulary $$\mathcal{V}_0$$ is intentionally over-complete 
    (often $$\lvert\mathcal{V}_0\rvert \gg \lvert\mathcal{V}_{\text{target}}\rvert$$). The UnigramLM algorithm appends a **pruning step** to some EM iterations to trim it down: it scores every piece by how much 
    corpus log-likelihood would drop if that piece were removed, then 
    discards the $$k_n$$ lowest-scoring pieces. 
    Explicitly, after updating 
    $$\boldsymbol{\phi}^{(n+1)}$$, it removes $$k_n$$ of the least "important" pieces, leading to a new $$\mathcal{V}_{n+1}$$. In our above example $$\text{a}$$ would be a natural candidate: it has the lowest expected count and its removal only eliminates the already-improbable path $$\langle \text{h}, \text{a}, \text{t} \rangle$$. Following pruning, the remaining probabilities in $$\boldsymbol{\phi}^{(n+1)}$$ are renormalized to form a valid distribution over $$\mathcal{V}_{n+1}$$. This pruning is done until the vocabulary reaches the desired size.
    
    Formally, let $$\bar{\mathcal{Q}}(\boldsymbol{\phi}^{(n+1)};\boldsymbol{\phi}^{(n)})$$ be our expected complete data log-likelihood under updated model parameters (albeit still under the segmentations according to $$\boldsymbol{\phi}^{(n)}$$). The algorithm removes tokens whose absence leads to the smallest decrease in (our proxy for) observed data log-likelihood. Intuitively, we prune tokens that contribute least to explaining the data under the current model. We define the contribution (or "loss") associated with token $$v$$ as the change (typically a decrease) in the corpus log-likelihood when $$v$$ is removed from the model:
    
    $$
    L(v)
    \mathrel{\stackrel{\textnormal{def}}{=}}
    \bar{\mathcal{Q}}(\boldsymbol{\phi}^{(n+1)};\boldsymbol{\phi}^{(n)}) 
    -
    \bar{\mathcal{Q}}(\boldsymbol{\phi}^{(n+1)}_{-v};\boldsymbol{\phi}^{(n)}_{-v}),
    \tag{15}
    $$
    
    The notation $$\boldsymbol{\phi}^{(n)}_{-v}$$ in Eq. (15) refers to the unigram distribution obtained from $$\boldsymbol{\phi}^{(n)}$$ by removing $$v$$ from its support and renormalizing the remaining probabilities. The corresponding string-level distribution is thus identical to the one induced by $$\boldsymbol{\phi}^{(n)}$$, except that all segmentations containing $$v$$ are assigned zero probability and individual piece probabilities are renormalized over $$\mathcal{V}\setminus \{v\}$$ (this logic also applies to $$\boldsymbol{\phi}^{(n+1)}_{-v}$$). After computing $$L(v)$$ for all $$v\in \mathcal{V}_n$$, we remove the $$k_n$$ tokens with the smallest losses, where $$k_n$$ is a hyperparameter chosen such that after some number of iterations, we ultimately reach our desired vocabulary size.[^6] Intuitively, this can be seen as removing the tokens whose removal incurs the *least* penalty on the corpus log-likelihood. Notably, computing $$\bar{\mathcal{Q}}(\boldsymbol{\phi}^{(n+1)}_{-v};\boldsymbol{\phi}^{(n)}_{-v})$$ in Eq. (15) is very computationally expensive since it requires a separate forward--backward pass over the corpus. We discuss some approximations to $$L(v)$$ in the following section.

#### Approximations of $$L$$. {#sec:approx_loss}

Computing $$\bar{\mathcal{Q}}(\boldsymbol{\phi}^{(n+1)}_{-v};\boldsymbol{\phi}^{(n)}_{-v})$$ in Eq. (15) for a given $$v$$ generally requires a separate forward--backward pass over the corpus. This is because disallowing the use of $$v$$ in segmentations changes both the set of valid paths and the total probability of those paths.[^7] The new per-string marginal probabilities (and expected token counts) under $$\boldsymbol{\phi}^{(n+1)}_{-v}$$. cannot, in general, be recovered from forward/backward marginals computed under $$\boldsymbol{\phi}^{(n)}$$. Hence, we would need a fresh forward--backward evaluation on the pruned lattice to obtain the exact $$\bar{\mathcal{Q}}(\boldsymbol{\phi}^{(n+1)}_{-v};\boldsymbol{\phi}^{(n)}_{-v})$$.

Performing a separate forward--backward pass for each piece in the vocabulary whenever we want to prune is impractical for vocabularies of any reasonable size. For example, if our initial vocabulary is a mere 100$$k$$, then computing per-piece losses would require 100$$k$$ forward passes of the corpus on its own. In practice, approximations that reuse the statistics computed from the current EM iteration are done. We discuss those next. To avoid the need to resegment the corpus to compute each $$v$$'s loss, several approximations can be used to compute per-piece losses. A simple approximation would be to use as a token's loss its contribution to corpus log-likelihood, i.e., $$\widehat{L}(v) \approx \widehat{c}_v(\mathcal{C};\boldsymbol{\phi}^{(n+1)})\log \boldsymbol{\phi}^{(n)}_v$$. An arguably more sound approximation (and the one used by the original implementation of UnigramLM found in the SentencePiece library) is to look at the change in corpus log-likelihood when simply replacing $$v$$ by the best alternative segmentation of that piece, i.e., the best alternative segmentation of the string $$g(v)$$ when $$v$$ is not in the vocabulary.

Formally, let $$\mathbf{v}' = h_{\boldsymbol{\phi}^{(n)}_{-v}}(g(v))$$ be the best segmentation of the string $$\mathbf{s}= g(v)$$ under $$\boldsymbol{\phi}^{(n)}_{-v}$$.[^7] The approximate loss is then the change to corpus log-likelihood when replacing every use of $$\boldsymbol{\phi}^{(n)}_v$$ with $$\prod_{t=1}^{\lvert\mathbf{v}'\rvert}P(V=v'_t; \boldsymbol{\phi}^{(n)}_{-v})$$ under the new renormalized unigram probabilities $$\boldsymbol{\phi}^{(n)}_{-v}$$. This loss can be computed concisely as:

$$
\widehat{L}(v)\approx \widehat{c}_v(\mathcal{C};\boldsymbol{\phi}^{(n)})\left[\log \boldsymbol{\phi}^{(n)}_v - \log \prod_{t=1}^{\lvert\mathbf{v}'\rvert}P(V=v'_t; \boldsymbol{\phi}^{(n)}_{-v})\right]
$$

<div style="background: #f4f4f8; border-left: 4px solid #6c63ff; 
            padding: 1em 1.2em; margin: 1.5em 0; border-radius: 4px;" markdown="1">
**Example 7** (Toy pruning example). *Suppose our corpus contains the string $$\mathbf{s}= \text{"internationalization"}$$*

*and our vocabulary includes the tokens*

$$
\{
    \text{international},\quad
    \text{inter},\quad
    \text{national},\quad
    \text{ization},\quad
    \text{al},\ldots
\}
$$

*Assume that under the current parameters $$\boldsymbol{\phi}^{(n)}$$, the posterior expected corpus-level counts are*

$$
\widehat{c}_{\text{international}}(\mathcal{C};\boldsymbol{\phi}^{(n)})
    \ll
    \widehat{c}_{\text{inter}}(\mathcal{C};\boldsymbol{\phi}^{(n)}),
    \widehat{c}_{\text{national}}(\mathcal{C};\boldsymbol{\phi}^{(n)}),
    \widehat{c}_{\text{ization}}(\mathcal{C};\boldsymbol{\phi}^{(n)})
$$

*To approximate the contribution of $$v_{\text{international}}$$, we consider its best alternative segmentation when it is removed from the vocabulary. Let*

$$
\mathbf{v}' = \langle \text{inter}, \text{national} \rangle
$$

*be the Viterbi segmentation of the string $$g(\text{international})$$ under the renormalized distribution $$\boldsymbol{\phi}^{(n)}_{-v}$$. The approximate loss associated with pruning $$\text{international}$$ is then*

$$
\begin{aligned}
    \widehat{L}(&\text{international};\boldsymbol{\phi}^{(n)})
    \approx\\
    &\widehat{c}_{\text{international}}(\mathcal{C};\boldsymbol{\phi}^{(n)})
    \log \frac{
        P(V=\text{international};\boldsymbol{\phi}^{(n)})
    }{
        P(V=\text{inter}; \boldsymbol{\phi}^{(n)}_{-v})
        \cdot
        P(V=\text{national}; \boldsymbol{\phi}^{(n)}_{-v})
    }.
\end{aligned}
$$

*Intuitively, if $$v_{\text{international}}$$ is both rare (small $$\widehat{c}_{\text{international}}(\mathcal{C};\boldsymbol{\phi}^{(n)})$$) and easily replaced by a segmentation whose product of probabilities is similar to $$P(V=\text{international};\boldsymbol{\phi}^{(n)})$$, then its (approximate) loss will be small, making it a good candidate for pruning.*
</div>

While this approximation does not account for changes in other valid paths' probabilities that might happen as a result of removing $$v$$ from the vocabulary, it seems to work fairly well in practice as a pruning heuristic (although I don't believe that anyone has actually tried to run the algorithm with the exact, brute-force loss computation).

#### Concise Pseudocode 
```
Algorithm UnigramLM-Train(C, V_target_size, V0, phi0, k_n):
    V   <- V0
    phi <- phi0

    while |V| > V_target_size:

        # ---- E-step ----
        hat_c[v] <- 0 for all v in V
        for each string s in C:
            lattice <- build_lattice(s, V)
            tilde_c <- forward_backward_expected_counts(lattice, phi)
            for each v in V:
                hat_c[v] <- hat_c[v] + tilde_c[v]

        # ---- M-step ----
        Z <- sum_{v in V} hat_c[v]
        for each v in V:
            phi[v] <- hat_c[v] / Z

        # ---- Pruning (approx loss) ----
        for each v in V:
            v_alt <- viterbi_best_segmentation(g(v), V \ {v}, phi)
            alt_prob <- product_{t in v_alt} phi[t]
            Lhat[v] <- hat_c[v] * ( log(phi[v]) - log(alt_prob) )

        V <- V \ bottom_k_tokens_by(Lhat, k_n)
        phi <- renormalize(phi over V)

    return V, phi
```

## Implementation in the SentencePiece library

In practice, the UnigramLM algorithm as we know it is largely defined by the public [SentencePiece implementation](https://github.com/google/sentencepiece/blob/master/src/unigram_model_trainer.cc), since Kudo (2018)<d-cite key="kudo-2018-subword"></d-cite> give only a high-level description and leave many engineering choices under-specified. The library makes a number of concrete design decisions that go beyond the abstract EM + pruning picture above.

#### Text Preprocessing.

Arguably some of the more critical design choices to be aware of are those pertaining to normalization and pretokenization, as these change which segmentations are feasible. SentencePiece advertises that it does not apply any pretokenization, but that claim really depends on one's definition of pretokenization... By default the library, collapses whitespace, inserts a dummy-prefix marker, and treats whitespace (and script/number boundaries) as explicit segmentation cues, i.e., as markers that can be suffixes or prefixes of pieces, but that pieces cannot cross. Most of these behaviors can be disabled via training flags but the fact that they're used is not well advertised. It also applies NFKC normalization by default.

#### Initialization.

The seed vocabulary is not "all substrings up to length $$L$$": SentencePiece uses a version of the Enhanced (Extended) Suffix Array procedure to mine a large lexicon of frequent substrings from the corpus (on the order of $$10^6$$ pieces by default), subject to length and frequency thresholds.

#### EM Updates.

SentencePiece runs a fixed EM+prune schedule rather than iterating EM to convergence on a fixed vocabulary. Each outer iteration consists of a small fixed number of EM "sub-iterations" (typically two), after which the vocabulary is pruned by a fixed shrinking factor, and training stops once the target vocabulary size is reached. SentencePiece does not use the plain MLE M-step update from Eq. (14). Instead, it adopts a Variational Bayesian approach with a Dirichlet prior, replacing expected counts with their [digamma-transformed counterparts](https://github.com/google/sentencepiece/blob/336900241c4943ae1e5f844b18292f532b3a21c7/src/unigram_model_trainer.cc#L390): $$\phi^{(n+1)}_v\propto\exp(\Psi(\widehat{c}_v(\mathcal{C};\boldsymbol{\phi}^{(n)})+\alpha_v))$$. While it might not seem like a large change to the original update rule, this choice is implicitly adding a prior belief about the the number of counts we should observe for each token. Explicitly, we're now calculating the geometric mean of a posterior Dirichlet distribution, where we're added in the belief token $$v$$ will be observed $$\alpha_v$$ times. Notably, SentencePiece uses an improper Haldane prior ($$\alpha_v= 0$$ for all $$v\in\mathcal{V}$$). This choice essentially has the opposite effect of performing standard additive smoothing: it's always the case that $$\exp(\psi(x)) < x$$, however, for small $$x$$ (rare tokens), the relative "discount" is significantly larger. It thus acts as a regularizer that disproportionately penalizes tokens with low expected counts, sending their assigned probability mass closer to zero. This is done on top of the removal of tokens whose expected counts are below a certain threshold (discussed next).
**Potential Research Direction: Exploring the effects of a sparse prior.** The Bayesian version of the M step (with the Haldane prior) over-penalizes low-count tokens relative to plain MLE. This is a modeling choice, not just an implementation detail. It raises a concrete question: for which settings (e.g., low-resource languages, morphologically rich languages) does this rare-token downweighting help, and where does it actually harm coverage or fairness?

#### Pruning.

The pruning procedure in SentencePiece is conceptually aligned with the procedure described above, but the actual implementation diverges in several concrete ways. 

* **Loss computation uses Viterbi for frequencies, not forward–backward.**  The loss for a given token $$L(v)$$ requires its (estimated) frequency counts.  SentencePiece's pruning step [does *not* reuse the E-step expected counts](https://github.com/google/sentencepiece/blob/439fec3ade0a9f8a1e6bc3903597582587ec0f9b/src/unigram_model_trainer.cc#L434) $$\widehat{c}_v$$ ("soft" counts attained by marginalizing over all possible segmentations) for these frequency estimates. Instead, it runs a *separate* Viterbi pass over the full corpus, collecting "hard" counts, i.e., counts under the the single best segmentation of each 
string under the current unigram model parameters ($$h_{\boldsymbol{\phi^{(n)}}}(\mathbf{s})$$). This means the pruning loss is computed with respect to a 
different set of counts than those used to update 
$$\boldsymbol{\phi}$$.

* **Alternative segmentations.** Rather than removing a 
piece $$v$$ from the vocabulary, renormalizing 
$$\boldsymbol{\phi}_{-v}$$, and running a fresh Viterbi decode of 
$$g(v)$$, SentencePiece uses the [loss approximation](#sec:approx_loss) described above. 
Explicitly, it estimates loss as the change in corpus log-likelihood when replacing $$v$$ with the best alternative segmentation of that piece, i.e., the best alternative segmentation of the string $$g(v)$$ when $$v$$ is not in the vocabulary.

* **Pre-pruning by count threshold.** Before computing approximate 
losses, SentencePiece does a pre-pruning step,
[removing](https://github.com/google/sentencepiece/blob/336900241c4943ae1e5f844b18292f532b3a21c7/src/unigram_model_trainer.cc#L381) 
all pieces from the vocabulary whose Viterbi frequency falls below a fixed threshold 
(0.5 by default).

* **Final pruning outside EM.** Not all vocabulary reduction happens 
within the EM loop. SentencePiece initially trains to 110% of the 
target vocabulary size (`desired_vocab_size = vocab_size * 1.1`). 
After the final EM+prune loop, a final pruning pass removes the 
remaining excess pieces based on their estimated log-probabilities, 
bringing the vocabulary down to the exact target size.


Taken together, these implementation details instantiate one particular, very specific version of the abstract UnigramLM model described above, albeit the one that people are typically referring to (rather than an implementation-free mathematical ideal) when talking about "the UnigramLM tokenization algorithm."

## Conclusions

Tokenization shouldn't be just a monolithic preprocessing step you fix once and forget; it quietly defines what your model even sees as input, and can have a huge effect on the behavior and fairness of the systems trained on top of it. Taking that seriously, we should treat tokenization as a full-blown modeling choice and explore the whole design space: priors (e.g., over length), supports (which segmentations are even allowed by pretokenization choices), and inference rules (Viterbi vs sampling vs marginalization). UnigramLM occupies just one corner of that space, but understanding it clearly is a step toward thinking about tokenizers as models we can design and question, not just as default settings we inherit.

### Acknowledgements 

As with pretty much any technical work I've written, Tiago Pimentel provided critical commentary and recommendations for this blogpost. Sander Land helped with the explanations of the SentencePiece implentation of UnigramLM. François Yvon gave helpful pointers to foundational prior work. And the reviewers of this blogpost provided great constructive feedback that really improved it.

[^1]: Some tokenizers instead operate directly on raw bytes.

[^2]: $$v$$ are also sometimes called subwords; we avoid this naming because $$v$$ need not align with orthographic words, in their typical definition.

[^3]: $$\circ$$ denotes string concatenation and when applied to tokens, indicates the pieces' symbols are concatenated together (perhaps with some special formatting if symbols from $$\Gamma$$ are present in the piece).

[^4]: The more standard way to define a proper distribution over variable-length sequences is to include a distinguished end-of-sequence symbol $$\langle\text{eos}\rangle \in \mathcal{V}$$ with its own unigram probability $$\phi_{\langle\text{eos}\rangle}$$; the probability of a sequence of length $$m$$ then falls out of the model itself (it is the probability that $$\langle\text{eos}\rangle$$ is drawn at position $$m+1$$ and not before). The explicit length prior $$M$$ used here is a strict generalization: if $$M$$ follows the geometric distribution implied by an EOS probability, the two formulations coincide. I use $$M$$ throughout because it makes the role of the length assumption visible in the derivations---particularly where dropping it changes the algorithm's behavior. It also allows us to handle sequence length without adding a special token to our vocabulary.

[^5]: There are several ways that this seed vocabulary can be created. The Enhanced Suffix Array is one of the more common algorithms. Often, pretokenization is performed on the corpus and one of the more common pretokenization rules splits on whitespace, preventing pieces from crossing whitespace boundaries, although that's kind of an arbitrary rule...

[^6]: Explicitly, relative to the original model, two coupled changes occur when removing $$v$$ from $$\mathcal{V}_n$$: (i) the feasible set of paths shrinks from $$\mathcal{T}_{\mathcal{V}_n}(\mathbf{s})$$ to $$\mathcal{T}_{\mathcal{V}_{n+1}}(\mathbf{s})$$ (all segmentations using $$v$$ are removed); (ii) the per-edge weights change after the renormalization of $$\boldsymbol{\phi}^{(n)}$$ and the marginal probabilities of remaining paths must be recomputed.

[^7]: This segmentation may need to include an UNK token depending on the base vocabulary.