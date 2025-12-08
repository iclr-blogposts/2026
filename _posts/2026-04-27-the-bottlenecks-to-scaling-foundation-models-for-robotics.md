---
layout: distill
title: The Bottlenecks to Scaling Foundation Models for Robotics
description: "Current approaches to building Vision-Language-Action (VLA) models largely rely on combining pre-trained Vision-Language Models (VLMs) with imitation learning. While effective in narrow benchmarks, this paradigm faces fundamental limitations for developing general-purpose robots that operate in complex, dynamic environments. In this article, I first review the standard training recipe and identify key bottlenecks, drawing on both my observations and existing empirical evidence. I then outline a path forward: integrating online reinforcement learning with pre-trained VLMs to enable lightweight, computationally efficient methods that scale with available resources."

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
    affiliations:
      name: Anonymous

# must be the exact same name as your blogpost
bibliography: 2026-04-27-the-bottlenecks-to-scaling-foundation-models-for-robotics.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
toc:
  - name: Will Scaling Solve Robotics?
  - name: The Common Recipe Behind Robot Foundation Models
  - name: Can We Just Scale Our Way to Embodied Intelligence?
    subsections:
      - name: Limitations of Current VLA Approaches
  - name: "Learning Like a Squirrel: Online Adaptation in Action"
  - name: "From Passive Observation to Active Experience: The Role of Online Learning in VLAs"
    subsections:
    - name: "How Do We Leverage Expert Demonstrations in Online RL?"
    - name: "Re-defining What We Mean by a World Model"
  - name: Concluding Thoughts
---




**Will Scaling Solve Robotics?** This question has been circulating widely in the robotics community in the last couple of years <d-cite key="Kumar_2024,icra2025debate"></d-cite>. When I first encountered it, I found it oddly vague. Scaling what - data, compute, memory, neural network capacity? And for which algorithms and settings? As I probed deeper, the roots of the question became clearer: it is shaped by two ideas that loom large in modern AI discussions - (i) the empirical success of increasingly large transformer models <d-cite key="vaswani2017attention"></d-cite> trained on internet-scale data at solving language understanding and generation tasks (e.g., the so-called scaling laws in large language models) <d-cite key="kaplan2020scaling"></d-cite> and (ii) the philosophy espoused in Richard Sutton’s influential blog post, The Bitter Lesson <d-cite key="sutton2019bitter"></d-cite> <d-footnote>Rich is actually vociferouly against LLMs, stating that as powerful as they might turn out to be, they will not lead to AGI. He does mention that mimicking what people say is not really to build a model of the world at all. Source: <a href="https://www.dwarkesh.com/p/richard-sutton">Richard Sutton – Father of RL thinks LLMs are a dead end</a> </d-footnote>. Simply put, the oversimplified narrative is that large data solved language, large data solved vision, and therefore large data will also solve robotics.

There are genuine concerns about large language models: the illusion of “thinking” <d-cite key="shojaee2025illusion"></d-cite>, their substantial carbon footprint <d-cite key="faiz2023llmcarbon"></d-cite>, and the looming limitation that we are quickly exhausting the scrapeable text on the public internet <d-cite key="villalobos2022will"></d-cite>. At the same time, there is countervailing optimism. Many expect training and inference costs to fall through improved model distillation techniques, more efficient architectures, and advances in hardware. Others believe that access to higher-quality data, synthetic data generation, or improved scaling practices will mitigate many of the current shortcomings. More details around this can also be found in the excellent article by Nishanth. 

All this speculative optimism has set the field of robotics in a frenzy, with the community in the middle of a genuine hype cycle. Demo videos of humanoids performing carefully orchestrated household tasks flood social media. Announcements from well-funded startups arrive weekly, each promising that general-purpose robots are imminent. Venture capital has followed this optimism aggressively, backing companies with valuations that far outpace their actual deployments or revenue.  A major source of this confidence is the belief that the LLM playbook can be straightforwardly transplanted into robotics. In language modeling, the recipe is well established: gather internet-scale data, pretrain a large transformer, curate higher-quality datasets, and fine-tune on downstream tasks. At its core, this is next-token prediction using supervised learning. In robotics, the analogy maps neatly onto expert demonstrations and imitation learning—collect sequences of state-action pairs and train a model to imitate. 

What this argument conveniently misses is a simple fact: *imitation is only one facet of intelligence, but it is far from the whole story.* This is the point where we should all question our motivations. What is it that we are trying to build? Is it a general-purpose robot that can just live and function in the real world like we do? Or is it a more narrow, specialist robot that can handle a couple of tasks. 


# The Common Recipe Behind Robot Foundation Models

**Step 1: Use a vision-language model (VLM) pre-trained on large scale data**

The current recipe for building robot foundation models typically begins with the use of a vision-language model (VLM) that has been pre-trained on large-scale data. Researchers select a backbone for processing visual and language inputs, such as CLIP <d-cite key="radford2021learning"></d-cite>, Gemma <d-cite key="team2024gemma"></d-cite>, or SigLIP <d-cite key="zhai2023sigmoid"></d-cite>. These models provide general scene understanding and instruction grounding, allowing downstream robot policies to start from a rich multimodal representation rather than learning perception from scratch.

**Step 2: Collect expert demonstrations**

The next step is to collect expert demonstrations. These are gathered through human teleoperation, kinesthetic teaching, or VR-based imitation. The datasets usually span a range of objects, environments, and manipulation skills to ensure broad coverage. Synthetic data produced through procedural generation or large-scale simulation is often added to augment real demonstrations, increasing diversity and coverage at lower cost.

**Step 3: Train vision-language-action (VLA) models**

With these ingredients in place, researchers train vision-language-action (VLA) models. The objective is to learn mappings from visual observations, language or task goals, and proprioceptive information to continuous robot actions. The VLM is kept frozen, and new layers are trained to translate VLM embeddings and proprioceptive states into actions using behavior cloning or diffusion policy objectives, following a standard imitation learning setup.

**Step 4: Fine-tune and deploy to the real world**

Finally, the VLA is fine-tuned and deployed in the real world. Fine-tuning is performed on a small, task-specific dataset while keeping the VLM backbone frozen. When needed, policy distillation is applied to reduce model size and achieve faster inference for deployment.

**N.B:** While specific design choices vary across projects, this general recipe is shared by most state-of-the-art systems. $\pi_{0.5}$ <d-cite key="intelligence2025pi_"></d-cite>, RT-2 <d-cite key="zitkovich2023rt"></d-cite>, OpenVLA <d-cite key="kim2025openvla"></d-cite>, and Gr00t N1 <d-cite key="bjorck2025gr00t"></d-cite> all follow the same broad pattern: start from a powerful VLM backbone, collect a diverse demonstration dataset, train a VLA through imitation learning or diffusion policies, and perform light task-specific fine-tuning before deployment. Despite architectural differences, these works reflect the same underlying paradigm that now dominates robot foundation model development.

# Can We Just Scale Our Way to Embodied Intelligence?

<blockquote>
“… general methods that leverage computation are ultimately the most effective, and by a large margin”
<br>   - The Bitter Lesson (Sutton 2019) <d-cite key="sutton2019bitter"></d-cite>
</blockquote>


Sutton’s Bitter Lesson highlights a pattern that has repeatedly played out in AI: methods that can effectively leverage additional compute to improve their performance tend to win out over more specialized, handcrafted approaches. This view now shapes much of the thinking in robotics, where researchers are attempting to scale the standard VLA recipe described earlier—large scale demonstration collection, more computational resources—in hopes of achieving more general and reliable manipulation capabilities.

In this pipeline, imitation learning plays a central role. Because most current VLAs rely on behavior cloning <d-cite key="pomerleau1988alvinn"></d-cite> or diffusion-based action sampling <d-cite key="chi2025diffusion"></d-cite>, their performance is tightly coupled to the coverage and quality of expert demonstrations. These models excel when the test-time distribution closely matches the demonstrations, but often struggle with long-horizon tasks, unseen objects, or novel environment configurations. And since the VLM backbone is kept frozen, it cannot adapt its perceptual representations to the specific affordances or dynamics that matter for manipulation. As a result, VLA training tends to resemble supervised regression on expert trajectories rather than the acquisition of new skills, limiting generalization and robustness once deployed in the wild.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/2026-04-27-the-bottlenecks-to-scaling-foundation-models-for-robotics/bender_shovel_data.png" class="img-fluid rounded z-depth-1" max-width="50%" %}
    </div>
</div>
<!-- <div class="caption">
    A simple, elegant caption looks good between image rows, after each row, or doesn't have to be there at all.
</div> -->

Early scaling results show some promise. Larger VLAs trained on more diverse demonstrations exhibit better generalization, stronger zero-shot transfer within related task families, and improved robustness to modest distribution shifts. These gains have fueled real optimism that simply scaling the existing imitation-driven recipe might meaningfully increase the breadth of robotic competence.

But this optimism runs headlong into the realities of embodied learning. Training and inference remain expensive, requiring substantial compute and infrastructure. Robot data does not scale like internet data, and collecting demonstrations is slow, costly, and closely tied to specific hardware and environments. While the ceiling for robot foundation models is still unknown, it is already clear that only a handful of well-resourced organizations can realistically pursue this scaling recipe. And even if one could afford it, imitation learning on its own cannot drive genuine discovery or the acquisition of novel skills—it reproduces what experts have demonstrated, but does not inherently push beyond the boundaries of the data it is given.

## Limitations of Current VLA Approaches

**1. Computational Efficiency**

A major challenge in current robot foundation models is computational efficiency. Training and inference remain expensive and demand substantial compute and infrastructure. Large models also run slowly at inference time, which directly translates into sluggish robot motion during real-world execution. These limitations highlight the need for more efficient algorithms that can achieve strong performance with far less compute.


**2. Passive Data Collection**

Data collection presents an equally significant bottleneck. Robot data does not scale in the same way as internet-scale corpora; it is slow, costly, and often tied to specific environments. At present, most approaches rely on passive data collection, which depends heavily on humans in the loop—whether through scraping online resources, generating synthetic data, or providing expert demonstrations. This raises the question of whether data collection can be automated through active exploration.

**3. Loss of Plasticity and Adaptation**

Offline training seeks to achieve zero-shot generalization across tasks and environments, but performance often degrades once models are deployed. When this happens, practitioners typically collect additional data, either through synthetic augmentation with domain randomization or through new rounds of expert demonstrations. One option is to retrain from scratch, which requires expensive retraining cycles and substantial human supervision. 

Another option is to fine-tune a larger base model using methods such as Low-Rank Adaptation (LoRA) <d-cite key="hu2021lora"></d-cite>. However, this approach often forgets aspects of the pre-training distribution and tends to be less robust in continual learning settings compared to full fine-tuning, as shown in Shuttleworth et al. (2024). As a result, adaptation is generally treated as an afterthought rather than a core design choice.

The real challenges—handling uncertainty and adapting to novel situations—live outside the imitation paradigm.


# Learning Like a Squirrel: Online Adaptation in Action

<center>
<iframe width="560" height="315" src="https://www.youtube.com/embed/DkmeZwsi3HA?si=J29yCdy35SU-f4d5" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</center>

The above video shows a squirrel navigating an obstacle course to reach its target: hazelnuts. There are two types of platforms: blue ones, which are rigid and stable, and red ones, which are attached to a slinky and thus unstable.

Initially, the squirrel engages in trial-and-error interactions with the world, testing different sequences of steps. Over time, we observe a brilliant learned strategy: stepping once on the red platform and twice on the blue platforms to stabilize its trajectory. Finally, the squirrel exhibits fast, reactive movements, executing the sequence efficiently and reaching the goal with minimal hesitation.

This example illustrates key hallmarks of natural intelligence: the ability to react in milliseconds, learn from ongoing experience, dynamically update an internal model of the world, and plan actions based on predicted consequences.

An intelligent robot aiming to emulate this level of adaptability should similarly be able to:
- React in real time to changes in the environment
- Learn on the fly, continuously updating its policy from new experiences
- Maintain and update an internal model of the world to guide decisions
- Plan sequences of actions toward a goal, using its world model to anticipate outcomes

Recasting these ideas in reinforcement learning (RL) terms:
- **Online learning:** Acting and learning are intertwined—the agent continuously updates its policy as it interacts with the environment.
- **Planning with learned world models:** The agent maintains an internal predictive model to reason about consequences, simulate potential action sequences, and select actions that maximize expected reward.

The squirrel’s performance highlights the difference between reactive trial-and-error learning and slow, memorization-drive behavior in large VLAs. For robots, achieving this level of natural intelligence requires integrating fast perception, online learning, and model-based planning in a single, adaptive system.



# From Passive Observation to Active Experience: The Role of Online Learning in VLAs

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0 text-center">
        {% include figure.liquid path="assets/img/2026-04-27-the-bottlenecks-to-scaling-foundation-models-for-robotics/data_sources.png" class="img-fluid rounded z-depth-1" max-width="100%" %}
    </div>
</div>
<div class="caption">
    The Data We Can Use in Robot Learning.
</div>

When considering how to integrate online reinforcement learning with VLAs, it helps to compare different sources of training data. Internet-scale datasets, scraped from images, videos, and text, provide broad coverage for vision and language but are passively collected and exhibit low environment fidelity. Simulation allows researchers to choose sensors and modalities freely, also in a passive manner, though the usefulness of the resulting data depends heavily on the sim-to-real gap. Expert demonstrations, collected by human pilots, offer high-fidelity experiences with the chosen sensors, but are still passively gathered. Finally, real robot experience represents the highest-fidelity data, collected actively through reinforcement learning with human-in-the-loop safeguards, enabling the agent to learn from direct interaction with the environment. Each source represents a trade-off between data scale, fidelity, and the ability to adapt online.

## How Do We Leverage Expert Demonstrations in Online RL?
In VLAs, imitation typically involves directly predicting the next action or token in a sequence. By contrast, animals often imitate outcomes, inferring the underlying actions needed to achieve them. In online reinforcement learning, expert demonstrations (i.e., expert trajectories) can guide the agent toward desirable behavior using off-policy RL methods such as SAC <d-cite key="haarnoja2018soft"></d-cite> or TD3 <d-cite key="fujimoto2018addressing"></d-cite>. Specifically, demonstrations can be incorporated into the replay buffer alongside online experience, allowing learning updates to sample both demonstration data and newly collected experience. This approach enables the agent to adapt online while retaining knowledge from expert demonstrations. Existing batch methods already support this approach effectively (e.g., Ball et al., 2023 <d-cite key="ball2023efficient"></d-cite>).

## Re-defining What We Mean by a World Model

In control theory and reinforcement learning, a world model represents the system dynamics, capturing the probability of the next state given the current state and action: 
$P(S_{t+1}∣S_t,A_t)$. Essentially, it predicts how the environment will change in response to actions. Examples include: (i) RL algorithms such as Dyna <d-cite key="sutton1991dyna"></d-cite>, Dreamer <d-cite key="hafner2025mastering"></d-cite>, and TD-MPC <d-cite key="hansen2023td"></d-cite>; and (ii) physics-based models used in model-predictive control (MPC) algorithms, such as MPPI <d-cite key="williams2016aggressive"></d-cite>.

World models support learning and planning “in imagination” by simulating potential outcomes internally. When these models are lightweight enough for real-time operation, they enable decision-time planning that allows:
- Safer exploration in real-world environments
- Robust performance under uncertainty

In contrast, LLM- or VLM-style world models capture associations between vision and language via next-token prediction $P(S_{t+1}∣S_{\leq t})$. They model correlations rather than explicit system dynamics. Unlike RL world models, they are not explicitly conditioned on actions.

This raises an important question: would we benefit from action-conditioned world models for dexterous manipulation? Addressing this would require designing and training large-scale model-based RL methods—akin to Dyna—tailored for robotic control.

# Concluding Thoughts
In summary, while scaling Vision-Language-Action models using large pre-trained VLMs and imitation learning has driven impressive short-term results, fundamental bottlenecks remain. Passive data collection, limited online adaptation, and frozen perceptual representations constrain generalization and real-world robustness, highlighting that imitation alone cannot achieve true embodied intelligence. Drawing lessons from natural learners and reinforcement learning, the path forward lies in integrating online, interactive learning with action-conditioned world models. By combining high-fidelity expert demonstrations, real-world interaction, and lightweight predictive models capable of planning and adaptation, future VLAs can move beyond narrow task replication toward the flexible, goal-directed behavior needed for general-purpose robotics.
