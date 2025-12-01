---
layout: distill
title: From Remote Sensing Foundation Models to World Models: A pathway in the perspective of Generative Modeling
description: This blog post explores the pathway from remote sensing foundation models to world models through the lens of Generative Modeling, arguing that true understanding of the physical world comes from the ability to recreate it.
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

# must be the exact same name as your blogpost
bibliography: 2026-04-27-remote-sensing-foundation-models-to-world-models.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "Defining the Simulator: World Models"
  - name: "Capturing the State: Remote Sensing Foundation Models"
  - name: "The Synthesis Pathway: Bridging Observation and Simulation"
  - name: "\"What I Cannot Create, I Do Not Understand\""
  - name: "Beyond the Textual Shortcut: Why World Models Need More Than Language"
  - name: "The Ultimate Compression: From Neural Networks to Physical Laws"
---

Note: please use the table of contents as defined in the front matter rather than the traditional markdown styling.

## Defining the Simulator: World Models

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>Takeaway:</strong> World Models are not just predictors; they are internal simulators of the environment's dynamics. Unlike static recognition, they learn the causal structure of the world to predict future states from current observations and actions.
</div>

<!-- 
TODO: Intro to world model definition (Ha & Schmidhuber, JEPA).
- Discuss the shift from "what is this?" (classification) to "what happens next?" (prediction/simulation).
- The role of the "World Model" module in Model-Based RL.
-->
[Content to be added]

## Capturing the State: Remote Sensing Foundation Models

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>Takeaway:</strong> Current Remote Sensing Foundation Models (RSFMs) excel at static representation learning. They compress the massive scale of planetary observations into latent spaces, effectively capturing the "state" of the Earth at discrete moments.
</div>

<!-- 
TODO: Intro to RS Foundation Models (SatMAE, Prithvi, etc.).
- Scaling laws in Earth Observation.
- Success in segmentation, classification, and detection.
- Limitation: Mostly focuses on static snapshots rather than temporal dynamics.
-->
[Content to be added]

## The Synthesis Pathway: Bridging Observation and Simulation

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>Takeaway:</strong> Remote Sensing is the most direct observation of the physical world we wish to model. The pathway to true World Models lies in transitioning RSFMs from static observers to dynamic simulators that understand the spatial-temporal evolution of the Earth's surface.
</div>

<!-- 
TODO: Build connection via RS specific perspective.
- World Models need a "World" to model. RS provides the ground truth of the real world.
- Moving from "Labeling the World" to "Modeling the World".
-->
[Content to be added]

## "What I Cannot Create, I Do Not Understand"

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>Takeaway:</strong> Generative Modeling is the litmus test for understanding. By focusing on Satellite Image Generation (2D) and City Generation (3D/4D), we force models to internalize physical laws, geometry, and lighting, moving beyond pattern matching to structural understanding.
</div>

<!-- 
TODO: Focus on Satellite Image Generation and 3D/4D City Generation.
- Richard Feynman's quote as the core philosophy.
- Generative models (Diffusion, NeRFs/Gaussian Splatting) as the engine for this understanding.
- If a model can consistently generate a city evolving over time, it understands the city.
-->
[Content to be added]

## Beyond the Textual Shortcut: Why World Models Need More Than Language

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>Takeaway:</strong> Language is a low-bandwidth, human-defined abstraction that discards the rich, continuous physics of reality. A true World Model must be grounded in the high-bandwidth, Any2Any signals (visual, spatial, temporal) of the physical world, not just the symbolic shortcuts of text.
</div>

<!-- 
TODO: Critique of LLMs as World Models.
- Language = Information Loss. It is a "zip file" of human experience, not the experience itself.
- The real world is continuous, noisy, and dynamic (Newtons laws apply to matter, not words).
- Vision for Any2Any models (Input: RS/Sensor data -> Output: Future states/multimodal).
-->
[Content to be added]

## The Ultimate Compression: From Neural Networks to Physical Laws

<div style="border: 2px solid #ff9800; background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
  <strong>Takeaway:</strong> Intelligence is efficient compression. While Foundation Models achieve success through massive parameter counts, they are still inefficient approximations compared to the "infinite compression" of fundamental physical equations (e.g., Newton's Laws, Maxwell's Equations). The ultimate goal is to close this gap.
</div>

<!-- 
TODO: Conclusion on Compression is Intelligence.
- Neural Networks: Brute force compression (data hungry, approximate).
- Physics: Elegant compression (simple equations, universal generalization).
- The trajectory of AI: Moving from memorization to generalization, asymptotically approaching the efficiency of physical laws.
-->
[Content to be added]
