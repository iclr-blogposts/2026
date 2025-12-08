---
layout: distill
title: What Can You Do When You Have Zero Rewards During RL?
description: "If your base model has zero success rates, performing RL with outcome rewards won't do anything. What can you do then? 🤔<br>TL;DR: simply adding easy samples to your training dataset can unlock RL training!"
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
bibliography: 2026-04-27-zero-rewards.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "📌 Key Takeaways"
  - name: "What can you do when you have zero rewards? 🤔"
  - name: Naive RL fails
  - name: Contemporary baselines don’t really work in the zero-reward scenario
  - name: A simple intervention helps — add easy samples in your dataset
  - name: Not all easy samples work well
  - name: Just mix everything!
  - name: Why does this work?
  - name: Conclusions and Future Work
  - name: Limitations
  - name: Appendix
    subsections:
        - name: Why do the baselines fail? Case by case analysis

# Below is an example of injecting additional post-specific styles.
---

> We recommend reading this blog with the **white** theme/background ⚪️ for the best experience 🙂

{% include figure.liquid 
  path="assets/img/2026-04-27-zero-rewards/figure1.png" 
  class="img-fluid" 
  caption="<b>Figure 1:</b> Approaches such as naïve RL (<span style='color: #2196F3;'>Dr. GRPO</span>), reward densification (<span style='color: #FF9800;'>Progress-Reward</span>), credit assignment (<span style='color: #4CAF50;'>VinePPO</span>), and diversity incentives (<span style='color: #9C27B0;'>BoN Aware Finetuning</span>) fail to solve a hard task.<br><b>A simple intervention of <span style='color: #8BC34A;'>mixing easier samples</span> helps unlock RL training!</b>"
  width="80%"
%}

Before we dive into the blog, here are the key takeaways and a quick disclaimer.

## 📌 Key Takeaways

- When the base model can’t solve a task at all (i.e., outcome rewards are always zero during RL training), we show that a **simple data-centric intervention** of adding *easier* instances of the same task in the training set works surprisingly well!
- Choice of *easy* instances that you add matters! Adding only *very* easy examples doesn’t help. However, you don’t need to hunt for the *perfect difficulty.* Mixing all the easier instances you have works!
- We benchmark methods that incorporate desirable components to tackle zero outcome rewards such as dense rewards, diversity incentives, and improved credit assignment, and **find none of these to be effective in our zero reward settings**. Since there was no official code for these baselines, we’re releasing (single-file, hackable) code for our implementation: [rl-baselines](https://github.com/anon-zero-rewards/zero-rewards-rl). Hopefully you’ll find it useful in your own experiments 🙂
- We conclude with a simple and practical recipe for RL practitioners: add all available *easier* instances of the task that one can get their hands on! We also connect our findings to ideas in skill learning and related prior work.

### ⚠️ Disclaimer

We focus our experiments on the graph search problem introduced in <d-cite key="bachmann2024pitfalls"></d-cite>. While we haven’t yet explored other task types, **we believe the insights here are interesting and worth discussing!**

Now, without further ado, let’s get started!

## What can you do when you have zero rewards? 🤔

The community has spent enormous compute training LLMs with RL on tasks with verifiable rewards, like math problems and code generation. Surprisingly, this works quite well, even though the reward signals are extremely sparse (outcome based only). Much of this success stems from the strong capabilities that base LLMs already possess due to large-scale pre-training. 

However, if a model is unable to solve a task even after thousands of attempts, performing RL will have no effect, since the gradients will be zero (there is nothing to *reinforce*).

This highlights an inherent assumption when applying RL to language models: the model must have a reasonable probability of solving the task.

This begs the question:
> What can one do if there are zero rewards due to no successful rollouts being sampled by the base model?

To address the above problem, one could:
- **Option 1: Supervised Finetuning (SFT) on successful traces:** One approach is to perform SFT on successful trajectories from a stronger model, giving the base model non-zero success rates before applying RL. While this works well in practice (we tried it on the graph search problem, where it proved helpful), it can be restrictive, since the RL phase is then biased toward the SFT dataset distribution. Still, due to its simplicity and effectiveness, it is widely used! **For now, let’s assume we can’t do this.**
- **Option 2: Densifying Rewards:** Alternatively, one could apply reward shaping <d-cite key="setlur2024rewarding"></d-cite> to obtain dense rewards that provide a learning signal based on the quality of intermediate steps, even when outcome rewards are zero. One could also explore approaches that improve credit assignment <d-cite key="kazemnejad2024VinePPO"></d-cite> for intermediate steps in reasoning.
- **Option 3: Encouraging Diversity:** One could also modify the objective to incentivize the model to sample diverse responses during RL finetuning  <d-cite key="chow2024inference"></d-cite>, with the hope that at least one of these responses gets a non-zero reward and thereby kick-starts RL training.
> However, **both Options 2 and 3 fail** on this simple graph task in the zero-outcome-reward scenario which we will talk about below. ☹️
- **Option 4: Use a stronger base model:** Of course, you could just start with a bigger or better model that already generates some successful traces, which RL can then build on. But that’s boring 😅, **so we’ll assume we can’t do that either.**

Outside of these, there aren’t many obvious ways to address the problem. Let us know if you think otherwise **🙂**.

Before we move onto our solution, we describe the task and show that both naive RL and reward densification are ineffective.

Let’s start unpacking!

## What’s the task?

The task involves finding a path from the source node to the target in a star-shaped graph <d-cite key="bachmann2024pitfalls"></d-cite>. The source node is always the center of the star, and the target node is one of the leaf nodes on any branch. Note that this task is difficult for a transformer to solve by directly outputting the path (<d-cite key="bachmann2024pitfalls"></d-cite>, <d-cite key="saparov2025transformers"></d-cite>). However, performing reasoning first and then outputting the path helps the transformer search in context for the path from the source to the target. Refer Fig. 2 for an illustration of the task.

{% include figure.liquid 
  path="assets/img/2026-04-27-zero-rewards/figure2.png" 
  class="img-fluid"
  caption="Figure 2: A Degree-3-Path-3 graph in which the center node has degree 3, and each outgoing path contains 3 nodes. In this example, node 4 is the source and node 7 is the destination, and the model is expected to output the path ‘4, 2, 7.’ This figure is adapted from Bachmann et al. (2024)."
  width="80%"
%}

## Naive RL fails
For our experiments, we focus on a Degree-10-Path-10 graph — i.e., a graph where the center node has degree 10, and each outgoing path has 10 nodes.

Before diving into the main ideas, let us first examine naïve RL using GRPO on this task. As shown in Fig. 1, rewards remain at zero across 1,000 iterations of RL training, resulting in zero gradients.

All experiments were conducted on four NVIDIA H100 GPUs using `Qwen2.5-1.5B-Instruct` as the base model, with a maximum wall-clock time of 24 hours or 1,000 RL iterations, whichever occurred first.

## Contemporary baselines don’t really work in the zero-reward scenario

**1. Rewarding Progress**

Here, we briefly show dense rewards in the form of a recently proposed method Progress Rewards <d-cite key="setlur2024rewarding"></d-cite> fails to solve the task.

Briefly, Progress Rewards <d-cite key="setlur2024rewarding"></d-cite> aims to reward *partial* reasoning traces based on the *progress* they make toward the final answer (akin to intermediate rewards). This essentially means the model being finetuned could be potentially *rewarded* even when the terminal/outcome reward is zero.

Although the method performs well in the original paper, it is less effective on the Degree-10-Path-10 graph, as shown in Fig. 1. The main reason is the base model’s inability to sample successful trajectories: intermediate rewards are non-zero only when the state value changes before and after a step, but the difficulty of the task makes such changes rare or non-existent.

**2. Encouraging Exploration and Diversity**

Orthogonal to dense rewards, a recent approach modifies the RL objective to better align with inference-time requirements (Best-of-N aware finetuning) <d-cite key="chow2024inference"></d-cite>. Since the goal is often to obtain a single correct generation out of N attempts, this method encourages the model to produce at least one correct answer among the N generations, promoting sample diversity. The idea is that any one of these diverse samples can receive a non-zero reward, enabling RL training. However, as shown in Fig. 1, success rates remain flat throughout training. As noted earlier, a key reason is again the base model’s inability to sample successful trajectories across multiple attempts.

In trying these different baselines out, we implemented our version of these since there was no official code. To rule out any implementation issues, we also test on simpler variants of the task where we find all baselines to perform well.

> Check out the detailed analysis of why the above methods fail in [Appendix](#appendix).

## A simple intervention helps — add easy samples in your dataset

Starting with a Degree-10-Path-10 dataset, we added samples from an *easier* task, such as a Degree-5-Path-5 task, and ran RL training using only outcome rewards with Dr. GRPO. Note that we mixed both datasets in equal proportion. This helped the model to solve the original Degree-10-Path-10 task (refer Fig. 3b) without any other modifications to the algorithm.

Note that we did **not** provide the model with any instructions on how to solve the hard task. Simply adding easy samples to the dataset helped the model figure out how to solve the hard task.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure3a.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="Figure 3a: Rewards during RL training on a dataset containing an equal mixture of Degree-5-Path-5 and Degree-10-Path-10 examples. With training, the model successfully solves both types of problems." 
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure3b.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="Figure 3b: Success rates on a held-out test set of Degree-10-Path-10 problems show that Degree-5-Path-5 examples help the model solve harder instances from the Degree-10-Path-10 dataset." 
        %}
    </div>
</div>


## Not all easy samples work well

Here, we consider mixing two other types of easy datasets in equal proportion with the original Degree-10-Path-10 task. The first easy graph is a Degree-2-Path-5 graph, where the center node has a degree of 2, and each of the outgoing paths has 5 nodes. The second easy graph is a Degree-5-Path-2 graph, where the center node has a degree of 5, and each of the outgoing paths has only 2 nodes.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure4a.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 4a:</b> Rewards during RL training on datasets containing equal mixtures of (i) <span style='color: #FF9800;'>Degree-5-Path-2</span> with <span style='color: #FF9800;'>Degree-10-Path-10</span> and (ii) <span style='color: #2196F3;'>Degree-2-Path-5</span> with <span style='color: #2196F3;'>Degree-10-Path-10</span>. With RL training, the model learns to solve only the easier examples in the mixture, resulting in a training reward of ~0.5." 
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure4b.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 4b:</b> Success Rate on a held out test set of Degree-10-Path-10 examples. In this case both the easier datasets (<span style='color: #FF9800;'>Degree-5-Path-2</span> and <span style='color: #2196F3;'>Degree-2-Path-5</span>) fail to help transfer to Degree-10-Path-10 task." 
        %}
    </div>
</div>

On looking at the chain-of-thought traces closely, we observed a few things that could explain the results in Fig. 4.

- In the case where <span style="color: #FF9800;">Degree-5-Path-2 mixed with Degree-10-Path-10</span>, most Degree-5-Path-2 instances were solved using just a single lookup in the edge list, without any search. While this strategy worked for the simpler Degree-5-Path-2 instances, it was not helpful for Degree-10-Path-10 instances, where searching down multiple paths was necessary to arrive at the correct answer. The traces resembled those observed when training a model directly on a Degree-10-Path-10 dataset (see the naive RL fails section [above](#naive-rl-fails)).
- In the case where <span style="color: #2196F3;">Degree-2-Path-5 is mixed with Degree-10-Path-10</span>, there were instances where, given a Degree-10-Path-10 problem, the model explored only two branches and then 'forced' itself to output the answer path. We believe the model may have learned a strategy that produces an answer after a maximum of two backtracks, which works for Degree-2-Path-5 but is ineffective for Degree-10-Path-10 problems.

Thus, **not all easy samples are effective**, and it is apriori *unclear* what a model will learn from a particular dataset. One needs to collect samples of the *right* difficulty, meaning samples that the model can solve correctly and that encourage behavior that transfers to the original task. These requirements make this approach cumbersome 🙁.

Another point to note is that, although it might be easier to guess which tasks have solutions that will not generalize to larger instances in a simple graph problem like this (e.g., reading off the edge list works for Degree-5-Path-2 examples but does not generalize to Degree-10-Path-10), in general, for any arbitrary problem, it is difficult to predict what the model will learn and which solutions will generalize to harder instances.

But is all lost?

## Just mix everything!

If we add not only Degree-2-Path-5 (or Degree-5-Path-2) but also Degree-5-Path-5 to the Degree-10-Path-10 dataset, we observe from Fig. 5b that the model once again begins to solve the Degree-10-Path-10 task!

This means that instead of trying to choose the right version of the *easy* sample for the original task, one should add all possible versions of easy samples available. Hopefully, *at least one of those instances* will be of the *right* difficulty.

**This actually makes it simpler. The final approach could be just described as:**

> *add all available easy samples for your task in the original dataset*

This provides a very **practical recipe for an RL practitioner.**

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure5a.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 5a:</b> Rewards during RL training on an equal mixture of <span style='color: #2196F3;'>Degree-2-Path-5, Degree-5-Path-2, Degree-5-Path-5, and Degree-10-Path-10</span> samples. With training, the model begins to solve both the easy and hard examples from the mixture." 
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure5b.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 5b:</b> Success Rate on a held out test set of <span style='color: #2196F3;'>Degree-10-Path-10 examples</span>. With training, the model begins to solve harder examples." 
        %}
    </div>
</div>

## Why does this work?

One thing to note is that this is not a free lunch. There is an underlying assumption: we rely on the availability of easier samples for the task. While these may be difficult to obtain in general, they seem reasonable for most real-world tasks.

Now we talk about why this works.

We think there could be an interesting perspective of *skill learning* here. *Easier* samples help the model learn a *skill* (correlated action) from outcome rewards only. These *skills* (or correlated actions) help solve a difficult task that the base model originally could not have solved (i.e. these correlated actions transfer to downstream tasks).

Another similar perspective is: these correlated actions (or skills) help make the search problem during RL easier – since now the search happens in the space of *skills* (correlated actions) and not in the space of raw tokens. In other words: the action space is reduced, making search easier.

To be concrete, one correlated action or skill that is useful for this task is: **going down a branch to a leaf node without hallucinating and then back-tracking.** Another could be to: **explore all branches systematically one by one.** That being said, these are anthropomorphized skills; what a model considers a skill is up to the optimization process during RL 🙂.

For more details about skill learning, we highly recommend this insightful talk by Ben Eysenbach [here](https://ben-eysenbach.github.io/self-supervised-rl/).

## Conclusions and Future Work

We now leave you with a discussion of how this connects to recent results in the community and to the role of datasets in reasoning, where algorithmic innovations have become the trend.

- Stojanovski et al. <d-cite key="stojanovski2025reasoninggymreasoningenvironments"></d-cite> talked about faster convergence when using an easy-to-hard curriculum. Setlur et al. <d-cite key="setlur2025e3"></d-cite> also talks about using such a curriculum to help the model gradually extrapolate its in-context search abilities. However, none of these works discuss the scenario when the base model has zero success rates initially.
- Liu et al. <d-cite key="liu2025prorl"></d-cite> discusses making specific changes to the RL algorithm and training on a much larger dataset mixture. As highlighted in the paper, we believe the dataset mixture, particularly the presence of both easy and hard examples, is a crucial ingredient in enabling RL to scale well beyond its usual number of steps. For instance, in the context of solving Olympiad math problems, a helpful skill, or correlated action, could come from an easy or medium algebra task in Reasoning Gym.
- Before writing this blog, the dataset used for RL was not very important to us. However, after these experiments, it has become clear just how important data is in RL and in learning the right strategy to solve a task.

This gives us some hints for future work.

- If a model cannot solve a difficult task on its own, can it generate easier subtasks for itself, along with a verifier, so that it can first learn to solve the easier subtask and then ultimately tackle the original difficult task? This idea is similar to the recent work by Chen et al. <d-cite key="chen2025self"></d-cite>.
- Although all our experiments used a uniform mixture of easy and hard samples, varying the proportion of each could influence the convergence speed of different methods. We leave a systematic exploration of this factor for future work.
- We believe that when assessing the effectiveness of RL algorithms, evaluations should include harder tasks that the base model cannot solve. Demonstrating progress on such challenging tasks would provide concrete evidence of improvements in the models' exploration capabilities.

## Limitations

- In this study our experiments were only focussed on the graph search problem which is a toy task, Extending such an analysis to more realistic settings would be an interesting follow up.
- Even though we mix samples from all difficulty levels and found this approach to work in our setup, the interaction of the mixture with RL optimization could mean that, even if examples of the right difficulty are present in the dataset, the model may still fail to learn the correct solution. We leave further exploration and analysis of this dataset mixture with RL optimization for future work.
- There is again an inherent requirement for a good pre-trained base model, one that achieves non-zero success rates on easier tasks. In our setting, this translates to the assumption of have non-zero success rates on Degree-5-Path-5 instances, which allows us to bootstrap RL training on Degree-10-Path-10.


## Appendix

### Why do the baselines fail? Case by case analysis

As we discussed above, **a key reason for the failure of some baselines on the Degree-10-Path-10 task is the base model's inability to occasionally sample correct trajectories.**
Here we provide a more detailed analysis of why each baseline fails in this zero-reward scenario.

Before we begin, here are the quick takeaways:
- Instantiating Progress Rewards <d-cite key="setlur2024rewarding"></d-cite> is **practically challenging**.
- There is a requirement of a capable base model **to begin with** for VinePPO <d-cite key="kazemnejad2024VinePPO"></d-cite> and to possibly resolve unstable training of Best-of-N aware finetuning <d-cite key="chow2024inference"></d-cite>.

### 1. Dense rewards are not really dense in VinePPO and Progress Rewards
Methods like VinePPO <d-cite key="kazemnejad2024VinePPO"></d-cite> and Progress Rewards <d-cite key="setlur2024rewarding"></d-cite> go beyond Dr.GRPO by computing step-level advantages. **However, non-zero step-level advantages are obtained only when there is a change in the value of the state before and after taking the step.** This means that for VinePPO to produce a non-zero advantage for a step, some of the rollouts under the current policy must succeed. Similarly, for the Progress Rewards, some of the rollouts under the prover policy must succeed. In our setting, we observe that throughout training, neither the current policy nor the policy generates a successful rollout. Thus step-level advantages offer no learning signal.

### 2. Instantiating a helpful prover to get a meaningful Progress Rewards is hard

The Progress Rewards <d-cite key="setlur2024rewarding"></d-cite> work notes that a prover that is *too strong or too weak* is ineffective: a strong prover cannot distinguish between *good* and *bad* steps, while a weak prover fails from most intermediate states, resulting in zero step-level advantages and no learning. Consequently, they identify two desirable properties for provers: (i) the prover should neither be too strong nor too weak, and (ii) it should be reasonably aligned with the policy being optimized.

To satisfy these requirements for the Degree-10-Path-10 graph, we experiment with two provers. The first prover: $\pi_{5 \times 5}$, is a model trained using Dr.GRPO on the Degree-5-Path-5 task and achieves around 65% accuracy on Degree-10-Path-10, partially satisfying the first property. The second prover, $\pi_{5 \times 5 \text{ mixed with } 10 \times 10}$, is trained using Dr.GRPO on an equal mixture of Degree-5-Path-5 and Degree-10-Path-10 graphs and reaches around 85% accuracy on Degree-10-Path-10.

As shown in Fig. 6, both provers have a reasonable success rate on the Degree-10-Path-10 task from the start, indicated through non-zero step-level advantages early in training. The $\pi_{5 \times 5}$ prover provides non-zero step-level advantages about 50% of the time, while the $\pi_{5 \times 5 \text{ mixed with } 10 \times 10}$ prover does so about 60% of the time. However, as seen in Fig. 6, this signal does not lead to better task performance. In both cases, the model responses often become degenerate, repeating characters or words to fill the context window. **We believe this happens because the prover policy is not well aligned with the policy being optimized,** possibly violating the second desirable property mentioned above, even though the prover was obtained using the same base model as the policy (you'd hope this should be a *good* *alignment*).

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure6a.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 6a:</b> Fraction of non-zero step advantages for two provers: <span style='color: #FF9800;'>μ = Best-of-4(π<sub>5×5</sub>)</span> and <span style='color: #2196F3;'>μ = Best-of-4(π<sub>5×5-mixed-with-10×10</sub>)</span>, where the models were trained on Deg-5-Path-5 alone or mixed with Deg-10-Path-10, respectively. Both models provide non-zero step advantages for Progress Rewards due to their reasonable success rates on the harder task." 
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure6b.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 6b:</b> Success rate on a held-out test set of Degree-10-Path-10 examples. Despite using the same two provers, both models fail on the Degree-10-Path-10 task. We believe this is because the prover policy is not well aligned with the policy being optimized." 
        %}
    </div>
</div>

### 3. Unstable training in Best-of-N aware finetuning

We followed the practices suggested in <d-cite key="chow2024inference"></d-cite>, including (a) using a KL schedule and (b) clipping the sample-dependent weights multiplied by the log probability (Eq. 9, Lemma 3 in <d-cite key="chow2024inference"></d-cite>). Notably, we observed that the KL schedule in <d-cite key="chow2024inference"></d-cite> is quite aggressive, starting with a coefficient of 1 and decaying to 0.001, whereas the current standard is to keep it constant at 0.001 <d-cite key="kazemnejad2024VinePPO"></d-cite>. They also clip the failure probability ($$p_{\text{fail}}$$). During trainin, we observed large sample-dependent weights ($$g_N^+$$ and $$g_N^-$$ in Eq. 9, Lemma 3 in <d-cite key="chow2024inference"></d-cite>) -- we countered these large weights by directly clipping them to stabilize training. Unfortunately, interventions did not enable the model to solve the Degree-10-Path-10 dataset.

To investigate further, we applied the method to the Degree-5-Path-5 dataset which Dr.GRPO can effectively solve (see Fig. 7). The model still did not solve the task. We believe a major reason is the presence of very high negative gradients. When the failure probability ($$p_{\text{fail}}$$) is close to 1, the sample-dependent weights become large, and multiplying them by negative log probabilities produces high magnitude negative gradients. This drives the model responses toward degeneracy where it repeats the same set of characters. Fig. 7 shows this effect: with a <span style="color: #FF9800;">lower KL penalty</span> (0.001), the model's response lengths increase rapidly, and inspection of the outputs confirms degeneracy, while success rates remain zero. Using a <span style="color: #2196F3;">strong-to-weak KL penalty</span> (0.1 to 0.001) stabilizes training but does not help solve the hard task.

We believe that the issue of high negative gradients can be mitigated by starting with a capable base model. Such a model has a lower failure probability ($$p_{\text{fail}}$$ in Eq. 9, Lemma 3 in <d-cite key="chow2024inference"></d-cite>), which keeps the sample-dependent weights $$g_N^+$$ and $$g_N^-$$ (in Eq. 9, Lemma 3 in <d-cite key="chow2024inference"></d-cite>) within a reasonable range, thus ensuring stable training. As shown in Fig. 8, Best-of-N aware finetuning is able to solve the Degree-3-Path-3 task, likely due to its relatively lower initial failure rates compared to Degree-5-Path-5.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure7a.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 7a:</b> Response lengths with different KL schedules on Deg-5-Path-5. Using a <span style='color: #FF9800;'>lower KL coefficient</span> (0.001) in Best-of-N aware finetuning results in unstable training due to large-magnitude negative gradients, causing model responses to degenerate into repeating the same character." 
        %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid 
            path="assets/img/2026-04-27-zero-rewards/figure7b.jpg" 
            class="img-fluid rounded z-depth-1" 
            caption="<b>Figure 7b:</b> Success rate with different KL schedules on Deg-5-Path-5. Using a <span style='color: #2196F3;'>KL schedule</span> as recommended in <d-cite key='chow2024inference'></d-cite> (decaying from 0.1 to 0.001) remains stable but fails to learn, as success rates stay at zero." 
        %}
    </div>
</div>


### 4. Sanity check: Are we sure our implementations for baselines are correct?


To ensure our implementations of the baselines were correct, we ran sanity checks on easier variants of the task (Degree-3-Path-3) where the base model has non-zero success rates.

{% include figure.liquid 
  path="assets/img/2026-04-27-zero-rewards/figure8.png" 
  class="img-fluid" 
  caption="<b>Figure 8:</b> Success rates of different RL algorithms (<span style='color: #2196F3;'>Dr.GRPO</span>, <span style='color: #4CAF50;'>VinePPO</span>, <span style='color: #FF9800;'>Progress Rewards</span>, and <span style='color: #9C27B0;'>Best-of-N aware finetuning</span>) on a held-out test set of Degree-3-Path-3 graphs. These models were trained on Degree-3-Path-3 graphs. All algorithms are able to solve the task when the model starts with a reasonable success rate. Furthermore, <span style='color: #4CAF50;'>VinePPO</span> converges in fewer iterations compared to <span style='color: #2196F3;'>Dr.GRPO</span>, consistent with findings reported in the literature."
  width="50%"
%}


### 5. Sanity check: Easier versions of the task are solvable

We confirm that the base model can solve easier variants of the task, such as Degree-3-Path-3 and Degree-5-Path-5, as shown in Fig. 9.

{% include figure.liquid 
  path="assets/img/2026-04-27-zero-rewards/figure9.png" 
  class="img-fluid" 
  caption="<b>Figure 9:</b> Success Rate of Dr.GRPO on held out test sets of different levels of difficulty. The model manages to solve simpler variants (<span style='color: #2196F3;'>Degree-3-Path-3</span> and <span style='color: #FF9800;'>Degree-5-Path-5</span>), but is unable to solve the harder <span style='color: #4CAF50;'>Degree-10-Path-10</span> variant."
  width="50%"
%}