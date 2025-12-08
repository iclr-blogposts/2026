---
layout: distill
title: The Hidden Cost of Robustness: A Mechanistic Audit of Deep Video Watermarking.
description: In the race to label AI-generated content, deep watermarking tools like VideoSeal have become the industry standard for provenance. The promise is simple: insert an imperceptible signal into the video bits that is robust enough to survive compression, cropping, and flipping, yet "invisible" to the human eye. But "invisible to humans" does not mean "invisible to AI."

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
bibliography: 2026-04-27-inv-watermark-analysis.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: The Sample Space (Standard Metrics)
    # subsections:
    #   - name: Interactive Figures
  - name: The Mechanistic Scope (Methodology)
  - name: Internal Dynamics (Direct Logit Attribution)
  - name: Visualizing the Mechanism (Attention Maps)
  - name: The Functional Cost (Logits & Margins)
  - name: Conclusion

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

In the race to label AI-generated content, deep watermarking tools like VideoSeal have become the industry standard for provenance. The promise is simple: insert an imperceptible signal into the video bits that is robust enough to survive compression, cropping, and flipping, yet "invisible" to the human eye.
But "invisible to humans" does not mean "invisible to AI."
As downstream models—classifiers, detectors, and multimodal agents—increasingly consume watermarked content, we must ask a fundamental question: **Does the watermark interfere with the semantic understanding of the video?**

Standard metrics like PSNR (Peak Signal-to-Noise Ratio) and Bit Accuracy tell us if the watermark looks good and if it can be read. They do not tell us if the watermark acts as an adversarial perturbation to a neural network.

In this study, we propose a **Mechanistic Validation Framework** for video watermarking. Using a pre-trained TimeSformer as our microscope, we analyzed the internal activations of videos watermarked with VideoSeal under geometric attack (Horizontal Flip).

Our findings reveal a startling trade-off between robustness and invisibility:
1. The "Tax" of Survival: In cases where the watermark successfully survived the attack (high bit accuracy), the model's attention mechanism locked onto the watermark artifacts, causing a measurable drop in classification confidence. The watermark was robust, but it was visible to the AI.
2. The "Clean" Failure: In cases where the attack destroyed the watermark (low bit accuracy), the model's attention remained pristine, and classification confidence was perfect. The semantic utility was preserved precisely because the watermark failed.

This case study suggests that for deep learning models, **a robust watermark is effectively a distraction**.

---

## The Sample Space (Standard Metrics)
The sample set of videos was picked from the test set of Kinetics-400, which is a 
collection of human-action dataset of 10 second videos.

We leveraged the Videoseal watermark, an open-source video watermarking framework by Meta that produces state-of-the-art performances on human relevant and tradition video quality metrics like PSNR, SSIM.

To evaluate how the VideoSeal detector behaves under different watermark keys and attacks, three sets of classes (3 videos each) were watermarked; diving_cliff, dunking_basketball, eating_spaghetti. We picked the following two classes for further analysis:

### Dunking Basketball
Watermark keys used: GHAOSR and PARKER

Purpose: examine how changing the embedded key influences detection stability and bit-accuracy patterns

These videos showed a higher peak in *baseline bit accuracy* (≈0.75–0.87)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/metrics_bb.png' | relative_url }}" alt="baskeball metrics">
    </div>
</div>
<div class="caption">
    Figure: Metrics for dunking basketball class.
</div>


### Eating Spaghetti
*Watermark key used:* PARKER for both videos

*Purpose:* maintain key consistency and isolate video-specific factors

*Baseline bit accuracy* stabilized under 0.76

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/metrics_spa.png' | relative_url }}" alt="Two flips">
    </div>
</div>
<div class="caption">
    Figure: Metrics for spaghetti class.
</div>


At the embed time watermarks used for dunking basketball videos were GHAOSR, `PARKER` to understand the shift in bit accuracy and model behaviour whereas for eating spaghetti videos `PARKER` was used for both the videos at the time of model
For testing purposes a wrong key was used as well to check model behaviour for eating spaghetti videos that used `PARKER` with a false key `HFSODF`, the detector rightfully evaluated with the wrong key, bit accuracy remains between 0.6 and 0.7 due to structured model output.

Across all videos, watermark embedding was performed using the publicly released y_256b_img.jit model from VideoSeal.



---

## The Mechanistic Scope (Methodology)

To understand why watermarks survive or fail, we cannot treat the semantic model as a black box. We need to look inside.

We employed TimeSformer (trained on Kinetics-400) not just as a classifier, but as a diagnostic tool. By exploiting its specific architecture—which separates spatial attention from temporal attention—we were able to trace the watermark's footprint through the network's layers.

We focused on three internal metrics to validate semantic integrity:
**Attentional Invisibility:** Validating Attention Map_Watermarked ≈ Attention Map_Original. If the map changes, the model is "looking" at the watermark.
**Feature Orthogonality:** Measuring Semantic Drift (Cosine Distance) in the residual stream. A high drift indicates the model's internal representation is being dragged away from the "true" video concept.
**Functional Stability:** Analyzing Logit Margins. Does the watermark reduce the gap between the correct class and incorrect classes?
By applying this framework to a "Best Case vs. Worst Case" dataset (High Survival vs. Low Survival), we uncovered the divergent mechanisms shown in Figure 1.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/two_flips.png' | relative_url }}" alt="Two flips">
    </div>
</div>
<div class="caption">
    Figure: The Tale of Two Flips. Top (Case A): The watermark survives the flip, creating a "Red Spot" in the attention map (distraction) and lowering confidence. Bottom (Case B): The watermark is destroyed by the flip, leaving the attention map clean and confidence perfect.
</div>

---

## Internal Dynamics (Direct Logit Attribution)

Our first mechanistic stop is the Direct Logit Attribution (DLA). This technique allows us to decompose the final classification score into the contributions made by each layer of the network. We specifically analyzed the Dunking (Survivor) case to see if the surviving watermark corrupted the decision-making process.
The Observation:
We compared the layer-by-layer contribution of the Original, Watermarked, and Attacked videos


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/dla_bb.png' | relative_url }}" alt="Mechanistic Validation of Dunking basketball">
    </div>
</div>
<div class="caption">
    Figure: Direct Logit Attribution of dunking basketball video.
</div>

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/dla_spa.png' | relative_url }}" alt="Mechanistic Validation of Dunking basketball">
    </div>
</div>
<div class="caption">
    Figure: Direct Logit Attribution of dunking basketball video.
</div>

Figure 2 (DLA): However, when projecting these states to the final class logit, the trajectories were nearly identical for both videos.

---

## Visualizing the Mechanism (Attention Maps)
If the model was reacting at Layer 10 (as shown by the drift), what exactly was it looking at?
We visualized the Spatial Attention of Layer 10 to resolve this question. This comparison (Figure 1) provides the visual proof of our "Trade-off" hypothesis.

Case A: The Survivor (Dunking)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/mechanistic_validation_spa.png' | relative_url }}" alt="Mechanistic Validation of Dunking basketball">
    </div>
</div>
<div class="caption">
    Figure 1: Attention map of dunking basketball video.
</div>

Visualization: The difference map (Attacked - Original) reveals a distinct "Red Hotspot."
Interpretation: This confirms that the Layer 10 drift was not random. The model's attention head snagged on a specific watermark artifact revealed by the flip. The watermark was salient—it acted as a localized distractor.

Case B: The Casualty (Spaghetti)

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/mechanistic_validation_spa.png' | relative_url }}" alt="Mechanistic Validation of Eating Spaghetti">
    </div>
</div>
<div class="caption">
    Figure 1: Attention map of eating spaghetti video.
</div>

Visualization: The difference map shows Diffuse Noise (Blue/Green).
Interpretation: Because the watermark structure was destroyed by the attack, there was no cohesive artifact for the model to track. The "distractor" was successfully scrubbed from the frame.

---

## The Functional Cost (Logits & Margins)

Finally, we look at the output Logits to quantify the "cost" of these internal dynamics. Does the "Red Spot" distraction matter in the real world?

` Dunking Original `
| Class | Confidence | CLS Logit |
| -------- | --------- | ---------- |
| Rank 1: playing basketball        | Prob: 0.6127 | Logit: 13.9037
| Rank 2: dunking basketball        | Prob: 0.2632 | Logit: 13.0587
| Rank 3: shooting basketball       | Prob: 0.1211 | Logit: 12.2822
| Rank 4: passing American football (not in game) | Prob: 0.0011 | Logit: 7.6069
| Rank 5: dribbling basketball      | Prob: 0.0004 | Logit: 6.4802


` Dunking Watermarked `
| Class | Confidence | CLS Logit |
| -------- | --------- | ---------- |
| Rank 1: playing basketball        | Prob: 0.5887 | Logit: 13.8093
| Rank 2: dunking basketball        | Prob: 0.2645 | Logit: 13.0092
| Rank 3: shooting basketball       | Prob: 0.1420 | Logit: 12.3873
| Rank 4: passing American football (not in game) | Prob: 0.0022 | Logit: 8.1996
| Rank 5: dribbling basketball      | Prob: 0.0007 | Logit: 7.0379


` Dunking Attacked (WM Survived) `
| Class | Confidence | CLS Logit |
| -------- | --------- | ---------- |
| Rank 1: playing basketball        | Prob: 0.5637 | Logit: 13.6449
| Rank 2: dunking basketball        | Prob: 0.2827 | Logit: 12.9546
| Rank 3: shooting basketball       | Prob: 0.1482 | Logit: 12.3086
| Rank 4: passing American football (not in game) | Prob: 0.0028 | Logit: 8.3365
| Rank 5: dribbling basketball      | Prob: 0.0005 | Logit: 6.6408


` Spaghetti Original `
| Class | Confidence | CLS Logit |
| -------- | --------- | ---------- |
| Rank 1: eating spaghetti          | Prob: 0.9971 | Logit: 18.0037
| Rank 2: dining                    | Prob: 0.0028 | Logit: 12.1250
| Rank 3: tasting food              | Prob: 0.0000 | Logit: 7.6617
| Rank 4: eating burger             | Prob: 0.0000 | Logit: 7.4389
| Rank 5: making pizza              | Prob: 0.0000 | Logit: 6.2776


` Spaghetti Watermarked `
| Class | Confidence | CLS Logit |
| -------- | --------- | ---------- |
| Rank 1: eating spaghetti          | Prob: 0.9976 | Logit: 18.0596
| Rank 2: dining                    | Prob: 0.0023 | Logit: 11.9913
| Rank 3: tasting food              | Prob: 0.0000 | Logit: 7.7378
| Rank 4: eating burger             | Prob: 0.0000 | Logit: 7.0424
| Rank 5: making pizza              | Prob: 0.0000 | Logit: 6.1444

` Spaghetti Attacked (WM Destroyed) `
| Class | Confidence | CLS Logit |
| -------- | --------- | ---------- |
| Rank 1: eating spaghetti          | Prob: 0.9981 | Logit: 18.3036
| Rank 2: dining                    | Prob: 0.0018 | Logit: 11.9933
| Rank 3: tasting food              | Prob: 0.0000 | Logit: 7.5005
| Rank 4: eating burger             | Prob: 0.0000 | Logit: 7.2384
| Rank 5: making pizza              | Prob: 0.0000 | Logit: 6.1130



*The Tax on Robustness (Dunking)*:

Original Confidence: 61%
Attacked Confidence: 56%
Result: A 5% drop in confidence (and a ~20% reduction in the margin between Top-1 and Top-2).
Conclusion: While the classification didn't flip, the model became less certain. The "energy" spent attending to the Red Spot (watermark) was energy taken away from the semantic features. This is the hidden tax of robustness.

*The Clean Getaway (Spaghetti)*:
Original Confidence: 99.7%
Attacked Confidence: 99.8%
Result: Zero Impact.
Conclusion: In the case where the watermark failed (Bit Acc ~50%), the semantic utility was perfectly preserved. By failing to survive, the watermark failed to distract.

SUMMARY COMPARISON :

| Video | Top1 Class | Top1 Prob | Margin (P1-P2) |
| -------- | --------- | ---------- | ---------- |
| Dunking Original playing basketball | 0.612679 | 0.349501 |
| Dunking Watermarked playing basketball  | 0.588731 | 0.324213 |
| Dunking Attacked (WM Survived) playing basketball  | 0.563726 | 0.281071 |
| Spaghetti Original eating spaghetti  | 0.997102 | 0.994312 |
| Spaghetti Watermarked eating spaghetti  | 0.997592 | 0.995283 |
| Spaghetti Attacked (WM Destroyed) eating spaghetti  | 0.998113 | 0.996299 |


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        <img class="img-fluid rounded z-depth-1" src="{{ '2026/assets/img/2026-04-27-inv-watermark-analysis/mechanistic_divergence.png' | relative_url }}" alt="Bit Accuracy Analysis">
    </div>
</div>
<div class="caption">
    Figure 1: Visualizing the attention divergence between original and attacked models.
</div>

---

## Conclusion

Our mechanistic audit reveals a fundamental tension in deep video watermarking. We often treat Robustness (survival of bits) and Utility (semantic quality) as separate optimization targets. However, our results with TimeSformer suggest they are mechanistically linked.

Invisibility is Fragile: A watermark that is invisible at rest can become salient under attack, acting as an adversarial patch that distracts the model.
The Zero-Sum Game: In our case study, we only achieved perfect semantic integrity (Spaghetti) when the watermark was reduced to a random chance (i.e; 50% bit accuracy). When the watermark survived (Dunking), it exacted a "compute tax" on the downstream model.

Future watermarking research must look beyond PSNR. To ensure watermarks are safe for an AI-driven web, we must validate them against the very models that will consume them, ensuring that provenance does not come at the cost of perception.

---
