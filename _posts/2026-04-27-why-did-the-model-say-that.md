---
layout: distill
title: "Why Did the Model Say That? A Methodological Practical Guide for Explaining Multimodal Medical Decisions"
description: A methodological guide that summarizes key explainability techniques for understanding multimodal medical AI systems, addressing the critical challenge of understanding what these systems have learned and how they'll behave in real-world clinical settings.
tags: explainable-ai xai multimodal-ai medical-ai healthcare-ai interpretability
giscus_comments: true
date: 2026-04-27
featured: true

authors:
  - name: Anonymous
    url: "#"
    affiliations:
      name: ""

bibliography: 2026-04-27-why-did-the-model-say-that.bib

# Table of contents
toc:
  - name: Introduction
  - name: Part 1 - How Multimodal Models Build Representations
    subsections:
      - name: The Representation Space
      - name: Multimodal Representation
  - name: Part 2 - Explainability Techniques for Probing Multimodal Systems
    subsections:
      - name: Visual Attribution Methods
      - name: Probing Learned Representations
      - name: Cross-Modal Attention Visualization
      - name: Model-Agnostic Methods
      - name: Modality Ablation Studies
  - name: Part 3 - Current Challenges and Gaps
    subsections:
      - name: Dataset Scarcity
      - name: Missing Data Handling
      - name: General-Purpose Encoders
      - name: Shallow Fusion Strategies
  - name: Conclusion
---

## Introduction

Medical AI has emerged as a potential solution to some of healthcare's most intractable problems, including specialist shortages, treatment backlogs, and limited access to care affecting millions worldwide. These systems promise to automate diagnosis, predict disease progression, and detect abnormalities at scale. The technology has advanced rapidly, with algorithms now achieving impressive accuracy rates on diagnostic tasks. But as these systems scale from research labs into hospitals and clinics, a critical gap has become apparent: we do not fully understand what these models have actually learned or how they will behave in the unpredictable conditions of real-world medical practice. Accuracy, as it turns out, is only part of the equation.

Consider what happened when researchers trained AI models to detect COVID-19 from chest X-rays {% cite degrave2021ai %}. The models achieved impressive accuracy on paper. But when the researchers dug deeper, they discovered something troubling. The algorithms weren't learning to recognize actual disease pathology. Instead, they were exploiting shortcuts. To understand what the models were actually seeing, the researchers used a technique called Saliency Mapping. Saliency maps work by measuring how much each pixel in an image contributes to the final prediction. The result is a visualization that highlights the regions the model relied on most. The researchers noticed that the models sometimes focused on the lungs, suggesting they had potentially learned something about COVID-19 pathology. But the models also highlighted regions that had nothing to do with disease, such as stamped dataset markers or the edges of images. The researchers discovered that instead of learning to diagnose COVID-19, the model had learned to distinguish datasets.

The COVID-19 detection models operated in a relatively simple world. They looked at chest X-rays and made a call. But the cutting edge of medical AI has moved well beyond single data sources. Modern systems integrate chest X-rays with radiology reports, patient histories, lab results, and vital signs to render diagnoses. This multimodal approach has become the dominant paradigm in medical AI research. In 2018, just three published papers explored multimodal medical diagnostics. By 2024, that number exceeded 150 {% cite schouten2025navigating %}. The fifty-fold increase reflects a fundamental insight about how medicine actually works. Doctors do not diagnose from images alone. They synthesize visual findings with clinical context, patient symptoms, and prior medical history. AI systems that mimic this approach consistently outperform single-modality models.

But the same complexity that makes multimodal AI powerful also makes it harder to understand. Multimodal models require investigating several interacting systems simultaneously. The transition from research prototype to clinical tool will be happening soon, not in some distant future. Understanding what these systems have learned before they become embedded in medical practice has become critical. These systems operate as what researchers call "black boxes." A black box model produces accurate outputs without revealing its internal reasoning process. You can observe what goes in and what comes out, but the transformation happening inside remains opaque. This creates a fundamental tension. The architectural sophistication that drives performance gains also deepens this opacity. The ideal would be systems that are both highly accurate and genuinely interpretable. Current multimodal AI achieves the former while struggling with the latter.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/spectrum.png" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 1: The Encoding Process - Transforming Heterogeneous Medical Data into Unified Embeddings
</div>

This is where Explainable AI (XAI) becomes critical. XAI encompasses techniques designed to make neural network decisions interpretable. Attention visualization reveals which image regions influence predictions. Attribution methods quantify feature importance. Embedding analysis exposes how models internally organize medical knowledge. These tools provide a window into the representations models learn, revealing not just what they predict, but how they structure information about disease.

This tutorial explores explainability as a lens for studying multimodal representation learning in medical AI. It's organized around three interconnected themes. First, we will examine how multimodal models represent medical knowledge internally. When a model processes both a chest X-ray and a clinical report, how does it encode that information? What does the learned representation space look like, and where do different medical concepts sit within it? Understanding these representations is fundamental to interpreting what models have actually learned. Second, we will explore explainability techniques specifically designed for probing multimodal systems. These range from attention visualization methods that reveal cross-modal alignment to attribution techniques that measure how models weight different data sources. We will see how these tools can be used not just to explain individual predictions, but to compare learned representations across different architectures. Finally, we will examine the real-world challenges and research gaps that remain. Missing data, deployment failures, and systematic disparities in how multimodal medical AI is developed all point to fundamental questions about representation learning that the field has yet to answer: Do different architectures converge on similar representations? Are there universal medical concepts that all models encode similarly? Can we predict which models will transfer knowledge effectively?

The goal is not just to make multimodal models more transparent. It is to use explainability as a research methodology for understanding what multimodal models learn, and whether that learning reflects genuine medical insight or sophisticated pattern matching.

---

## Part 1: How Multimodal Models Build Representations

### The Representation Space

When a doctor learns to diagnose a disease, the process works like traditional education. Read the textbooks. Memorize the diagnostic criteria. Learn which symptoms point to which conditions, what findings appear on imaging, and when to order which tests. Ask a doctor why they made a diagnosis, and they'll walk you through it. This is how human doctors learn. Multimodal medical AI systems learn differently. Every input, whether it be a chest X-ray, a clinical note, or lab results, first passes through an encoder, a component that transforms the raw data into a mathematical representation called an embedding. These embeddings are vectors of numbers that capture the essential features of the input.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/learned_representation.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 2: Visualizing the Learned Representation Space of Multimodal Medical AI
</div>

All these embeddings exist in a high-dimensional mathematical space called the representation space (also called latent space or embedding space). In this space, every medical concept occupies its own location based on the patterns the model discovered during training. The model doesn't store knowledge as rules or criteria. Instead, it organizes information geometrically. Concepts that appear in similar contexts or share statistical patterns in the training data tend to be positioned closer to each other. This often, but not always, corresponds to semantic or clinical similarity.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/shared_representation.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 3: Example Shared Representation Space
</div>

The representation space reveals what the system has learned. If the model has grasped genuine medical insight, its representation space can reflect clinical reality. Bacterial and viral pneumonias should cluster together. Conditions affecting the same organ system should be neighbors. Different severities of the same disease should sit along a spectrum. But if the model has learned superficial shortcuts, like those COVID-19 detectors that memorize dataset markers, the space might look different. The representation space becomes a diagnostic tool in itself. Not for diagnosing patients, but for diagnosing what the AI has learned. We will explore specific techniques for probing and visualizing these representation spaces in Part 2 of this blog.

### Multimodal Representation

But how does a model create a unified representation space when working with fundamentally different data types? A chest X-ray arrives as a 512×512 grid of pixel intensities. A radiology report is a sequence of words. Lab values are structured numbers. These modalities don't naturally speak the same language. Modern medical AI systems solve this through a two-step process.

#### Step 1: Encoding Data

First, each modality gets its own specialized encoder. An encoder is a computational structure that transforms raw data into a standardized numerical format called an embedding. For medical images like chest X-rays or pathology slides, the dominant encoder type is a convolutional neural network. CNNs process images through stacked filters that detect patterns at increasing levels of complexity. Early filters might recognize edges or textures. Deeper filters identify anatomical structures like blood vessels or tissue boundaries. By the final stage, the CNN has compressed a high-resolution image into a compact vector of numbers that captures its essential visual features.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/image_encoder.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 4: Image Encoder: CNN
</div>

Text data requires different machinery. Radiology reports and clinical notes get processed by transformers or recurrent neural networks. These encoders understand sequential information and can capture relationships between words. A transformer can learn that "bilateral infiltrates" and "diffuse opacities" describe similar findings even though they use different terminology. The encoder converts the report into a numerical vector that encodes its medical meaning.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/test_encoder.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 5: Text Encoder: Transformer Language Model
</div>

Structured data like lab values or vital signs typically uses simpler encoders. Multi-layer perceptrons or even handcrafted feature extractors can transform these already-numerical inputs into representations compatible with the other modalities. Some researchers design custom encoders for specialized data types like genetic sequences or protein structures.

#### Step 2: Fusion

Fusion is a model architecture decision. Architecture describes the arrangement of computational components within a model and the paths data takes as it moves from input to output. Different architectural choices create different learning capacities. While some structures allow deep interaction between modalities, others keep them largely separate until the final step.

Fusion is the process of combining or integrating representations from different modalities so the model can learn cross-modal patterns and relationships. The timing of fusion varies across architectures. Intermediate fusion dominates current research, appearing in 79% of studies {% cite schouten2025navigating %}. In this architectural pattern, each modality first passes through its own dedicated encoder, and fusion occurs after encoding. After fusion, the output (whether a single combined vector or interacting representations) then flows through additional neural network layers before reaching the final output, whether that's a diagnosis classification or risk prediction.

Intermediate fusion allows each encoder to specialize in its own data type while still enabling the model to learn cross-modal patterns. Intermediate fusion sits between two extremes. It's not early fusion, where raw data combines before any encoding. It's not late fusion, where each modality maintains complete independence until final predictions merge. Common intermediate fusion strategies include concatenation, attention mechanisms, projection into shared embedding spaces, and methods that capture feature interactions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/fusion.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 6: Adapted from Schouten et al. (2025), Fig. 6. Navigating the landscape of multimodal AI in medicine: A scoping review on technical challenges and clinical applications. ScienceDirect. https://doi.org/10.1016/j.artmed.2025.102368
</div>

The fusion approaches described in this blog post apply specifically to neural networks, the deep learning systems that dominate modern medical AI. Traditional machine learning approaches like random forests or support vector machines handle multiple data types differently, typically requiring manual feature engineering to combine modalities before training. Neural networks can learn to fuse information automatically through their layered structure.

##### Concatenation

Concatenation stacks embeddings into a single unified representation. For example, an X-ray might be encoded into a 512-dimensional vector while a clinical report gets encoded into a 768-dimensional vector. Concatenation places them side by side into a single 1280-dimensional vector that downstream layers learn to interpret. This method dominates intermediate fusion research and appears in roughly 69% of these models {% cite schouten2025navigating %}. The encoders don't explicitly learn to align the modalities in a shared space. Instead, subsequent layers learn to interpret the combined vector and extract cross-modal relationships useful for prediction.

##### Attention-based Mechanisms

Attention-based fusion offers another approach used in about 12% of intermediate fusion models {% cite schouten2025navigating %}. Here, the model learns to dynamically weight different modalities based on their relevance to each specific prediction. For one patient, the imaging might be most informative. For another, the clinical history carries more weight. Attention mechanisms let the model decide which information source deserves emphasis. Unlike concatenation, attention-based fusion lets representations remain distinct while querying each other.

##### Projection into Shared Embedding Space

More sophisticated architectures create a shared embedding space where different modalities can be directly compared. Each encoder still produces its initial embedding in its own modality-specific space. But then a projection step maps all embeddings into a common geometric space where the model learns alignments. For instance, a chest X-ray showing pneumonia and its radiology report describing fluid in both lungs would be positioned close together in this space. Meanwhile, that same X-ray would sit far from a report describing a normal chest. In the shared space, distance encodes semantic similarity, meaning that concepts with related medical meanings occupy nearby positions. Related conditions cluster together. Unrelated findings stay far apart.

##### Other Fusion Strategies

Other approaches, like outer products or bilinear pooling, create joint representations that capture feature combinations across modalities. Rather than simply placing features side by side, these methods encode pairwise interactions. For example, if one modality detects a hazy pattern on chest imaging and another notes elevated inflammatory markers, the joint representation can encode not just both features independently but their co-occurrence as a meaningful pattern. This captures relationships like "feature A from imaging combined with feature B from labs" that simple concatenation might miss.

##### Beyond Intermediate Fusion

The intermediate fusion approaches described above dominate in the multimodal medical AI space. But other timing strategies exist. In late fusion, each modality gets its own complete model that makes independent predictions, and the final predictions of each model are combined, often through simple averaging or by training a separate model on top of the individual predictions. This prevents modalities from teaching each other during learning, but makes handling missing data easier since each model trains independently.

In early fusion, modalities combine before any encoding happens. This approach faces a fundamental obstacle in that raw data from different sources exists in incompatible formats. A chest X-ray arrives as a grid of pixels, while lab values are structured numbers, and clinical notes are text sequences. Some researchers solve this by embedding clinical variables directly into images or building graph networks that connect different data types from the start.

---

## Part 2: Explainability Techniques for Probing Multimodal Systems

Encoders and fusion strategies determine how models can integrate information across modalities. But architecture is only half the story. What matters is what the trained model actually learned. Understanding what a model has actually learned requires probing different aspects of how the model processes and integrates information. Researchers have adapted explainability techniques for unimodal models to probe multimodal systems. These approaches fall into several categories, which we explore below.

### Visual Attribution Methods

Visual attribution methods answer a basic question: which parts of an image actually drove the model's prediction? The technique produces heatmaps where bright regions show areas that influenced the output and dark regions show areas the model ignored.

Gradient-weighted Class Activation Mapping (Grad-CAM) is a common visual attribution method used in medical AI. Grad-CAM builds on earlier Class Activation Mapping (CAM) methods introduced by Zhou et al. in 2016 {% cite selvaraju2017gradcam %}, which highlighted discriminative image regions by weighting feature maps from the final convolutional layer. CAM saw early adoption for tasks like weakly-supervised nodule segmentation and lesion localization, but it required specific architectural constraints, including global pooling followed by a fully connected layer. Grad-CAM, introduced by Selvaraju et al. in 2017 {% cite selvaraju2017gradcam %}, removed these restrictions by using gradients of the class score with respect to feature maps instead of requiring weights from a fully connected layer. Most medical image encoders use convolutional neural networks (CNNs), which process images through stacked layers that detect increasingly complex patterns. Grad-CAM exploits this layered structure by flowing gradients backward through these convolutional layers to identify which image regions increased or decreased confidence in the prediction.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/2025-10-16-why-did-the-ai-say-that/grad_cam.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Figure 7: Gradient-weighted Class Activation Mapping (Grad-CAM)
</div>

Because Grad-CAM works with any CNN architecture, it has been widely adopted across medical imaging research. In 2021, researchers studying diabetic retinopathy {% cite huff2021interpretation %} used Grad-CAM to verify that models correctly identified hemorrhages and microaneurysms rather than camera artifacts. Some researchers have also applied Grad-CAM analysis to attention modules within their models. One study used this approach to predict microvascular invasion, a condition where cancer cells spread into small blood vessels surrounding a liver tumor. By generating heatmaps from the model's attention layers, researchers revealed that the model focused on the edges of tumors and the tissue immediately surrounding them.

Other gradient-based methods, like integrated gradients, offer complementary approaches. Integrated gradients compute how much each input pixel would need to change to alter the prediction by considering the path from a baseline image to the actual input.

### Probing Learned Representations

Visual attribution reveals what image encoders learned to recognize. A complementary approach probes the representation space directly by analyzing the embeddings the model generates and the geometric relationships between them. These techniques reveal how embeddings cluster together, separate into distinct groups, and relate to one another in the space.

**Dimensionality reduction:** Dimensionality reduction provides the most direct window into representation space. Techniques like t-SNE, UMAP, and PCA compress high-dimensional embeddings into two or three dimensions that humans can visualize, preserving relative distances to show structure. They can be used to examine whether diseases form distinct clusters or continuous spectrums, how different conditions relate, and where individual cases sit within this landscape. One study of COVID-19 chest X-ray models {% cite covid_cxr_umap %} used UMAP to visualize the learned representations within the neural network. The researchers extracted the 256-dimensional output from the penultimate fully connected layer, which was the layer by which the network had already encoded diagnostically relevant patterns before final classification. Using UMAP for dimensionality reduction, they projected this high-dimensional latent space onto two dimensions, creating a visualization where proximity indicated similarity in the model's internal representation. The visualization exposed the continuous structure the model had learned, revealing how the entire patient population was distributed across disease stages and how individual trajectories moved through this space during disease progression and recovery.

**Nearest neighbor analysis:** Nearest neighbor analysis is a crucial technique for understanding what features a model has actually learned by examining its representation space. This technique involves identifying which examples in the training data are closest to a given input when measured by distance metrics (such as Euclidean distance) in this learned representation space. Given a specific medical image, researchers can identify which other images sit closest in the embedding space by computing distances between their learned representations. These neighbors reveal what the model considers similar and provide insight into the features driving the model's decisions.

This approach has been applied as a post hoc explainability method in clinical settings, where similar patient retrieval uses k-nearest neighbor techniques to identify comparable cases from training data. For example, in a recent study on ICU mortality prediction {% cite icu_mortality %}, researchers fine-tuned a medical language model and applied k-nearest neighbor retrieval using cosine similarity to find the top-3 most similar patients based on their admission notes. When evaluated by 32 practicing clinicians, this exemplar-based approach demonstrated a greater ability to build trust compared to feature-based methods, as it mirrors how practitioners naturally rely on past experiences in clinical decision-making. Similarly, in medical imaging, another classification model called the Retrieval Augmented Medical Diagnosis System {% cite retrieval_augmented %} combines standard predictions with nearest neighbor-based retrieval of similar historical cases for breast ultrasound classification, achieving a 21% improvement in sensitivity while providing clinicians with interpretable reference points for understanding the model's reasoning.

**Probing classifiers:** Probing classifiers work like diagnostic tests for models. They reveal what information is hiding inside a model's internal representations. Researchers freeze a trained model and extract embeddings from one or more layers, then train a simple classifier on top of these embeddings to predict specific attributes or categories. If this simple classifier succeeds, the model's representation space at that layer must already contain that information. Researchers typically use linear classifiers because they provide the clearest test of whether information is directly accessible in the embeddings, though some studies use slightly more complex probes. Researchers often probe multiple layers systematically to understand how different types of information emerge at different depths in the network. Earlier layers typically encode low-level features, while later layers encode higher-level concepts.

Researchers have used probing classifiers to detect when models encode sensitive information. In one study, researchers trained simple classifiers to predict patient race and sex {% cite race_sex_cxr %} from chest X-ray embeddings in disease detection models. These classifiers succeeded with high accuracy, confirming that demographic information had been learned and encoded in the model's representations.

However, probing classifiers have an important limitation. While they confirm the presence of information in the embeddings, they cannot determine whether the model actually uses this information for its predictions.

### Cross-Modal Attention Visualization

Attention mechanisms have become central to modern multimodal architectures. These components learn to weigh different parts of the input based on relevance. In multimodal medical AI, attention determines how imaging features and text tokens interact. Visualizing these attention patterns reveals how models align information across modalities.

Attention visualization works by extracting attention weights from the model's fusion layers. These weights indicate how strongly the model connected each image region to each text token when making a prediction. The weights can be rendered as matrices or heatmaps. High attention between a lung opacity in an image and the word "consolidation" in a report suggests the model learned a meaningful clinical relationship.

Research on breast cancer diagnosis demonstrates attention visualization's power. One influential study {% cite breast_cancer_attention %} examined models that processed mammograms alongside BI-RADS structured reports. BI-RADS is a standardized radiology vocabulary that describes breast findings using specific descriptors like mass shape and margin characteristics. The researchers visualized attention weights between image regions and BI-RADS descriptors in the reports. The results showed clinically coherent alignments. When the text mentioned "irregular mass," attention weights concentrated on the corresponding mass location in the mammogram. When reports described architectural distortion, attention was highlighted in the distorted tissue regions. The model had learned to ground abstract clinical language in concrete visual features. This alignment suggests the representation space encodes meaningful medical relationships rather than spurious correlations.

### Model-Agnostic Methods

Some explainability techniques work regardless of model architecture. These model-agnostic methods treat the AI system as a black box. They only require the ability to query the model with inputs and observe outputs. Shapley Additive explanations provide a unified framework grounded in game theory. SHAP values represent each feature's contribution to the prediction as the average marginal contribution across all possible feature coalitions. The math comes from Shapley values originally developed for cooperative games. Computing exact SHAP values requires exponentially many model evaluations, so practical implementations use approximations. DeepSHAP adapts the approach for neural networks by leveraging backpropagation. KernelSHAP provides a model-agnostic sampling method. These variants trade computational efficiency for exactness. Medical AI research has adopted SHAP extensively. A study examining a multimodal sepsis prediction model called DeepSEPS {% cite deepseps %} used SHAP to measure how lab values and vital signs contributed to predictions. The analysis revealed that for septic shock prediction, the model appropriately prioritized lactate as the most important feature, despite it being a sparse laboratory measurement. However, for general sepsis prediction (without shock), the model relied more heavily on readily available features like oxygen delivery type and heart rate, with lactate playing a secondary role. This demonstrates how SHAP can quantify feature importance and reveal how models weight different data sources depending on the specific prediction task, even within the same clinical domain.

It is important to note that SHAP faces a fundamental limitation in multimodal settings. It explains individual features independently but struggles to capture cross-modal interactions. If a prediction emerges from the combination of an imaging finding with contradictory clinical text, feature-level attributions may not reveal this relationship. The method works best for identifying which modality dominates rather than explaining how modalities interact. This limitation means SHAP should complement rather than replace techniques specifically designed for multimodal analysis.

### Modality Ablation Studies

Ablation studies offer the most direct way to measure what each modality contributes. The approach follows a simple protocol. Train the model with all modalities. Then systematically remove modalities and measure how performance changes. A large performance drop when removing imaging suggests the model relied heavily on visual information. Minimal change when removing clinical text suggests that modality contributed little.

Medical researchers use ablation studies to validate that multimodal models actually benefit from multiple data sources. For example, the PneumoFusion-Net study {% cite pneumofusion %} used ablation experiments to quantify each modality's contribution. When classifying pneumonia from CT images, clinical text, numerical lab data, and imaging reports, researchers found that CT image features contributed approximately 45% to the decision-making process, clinical text 12%, numerical data 33%, and imaging reports contributed to the remaining percentage. Removing numerical data resulted in the largest drop in accuracy, indicating that biomarkers were highly critical for disease classification.

Ablation studies also reveal redundancy across modalities. Some research on cardiac risk prediction found that models achieved similar performance with either echocardiogram videos or structured clinical measurements. The modalities contained overlapping information about cardiac function. This suggested that in clinical deployment, either modality alone might suffice when the other is unavailable.

The limitation is that ablation studies only measure the overall contribution. They don't reveal what specific features within each modality the model uses or how modalities interact during fusion. A modality might be essential overall, while the model still ignores important information within that modality. Ablation studies work best when combined with other techniques that provide finer-grained analysis. Together, these methods build a comprehensive picture of what multimodal medical AI systems actually learned and whether that learning reflects genuine medical understanding.

---

## Part 3: Current Challenges and Gaps

Multimodal medical AI faces several major challenges that limit its progress and real-world usability.

### Dataset Scarcity

A fundamental obstacle is the lack of large, paired datasets. Collecting multimodal medical data is difficult because each case must include both imaging and text components, such as an X-ray and its radiology report. Hospitals rarely release complete datasets, and many patient records are missing one modality. Because of this scarcity, most research teams rely on pretrained image and text encoders that were originally built for general data rather than medical data. This shortcut saves time but introduces gaps in domain-specific understanding.

### General-Purpose Encoders

That reliance on general-purpose encoders creates another issue. Models trained on everyday images and common text sources struggle to interpret the subtle patterns and precise language of medicine. The image encoder may recognize shapes and textures but not specific clinical abnormalities, while the text encoder may misinterpret medical abbreviations or context. As a result, these models can combine data streams effectively but do not truly understand either modality in the way a clinician does.

### Shallow Fusion Strategies

Fusion presents its own difficulties. Early systems used late fusion, merging outputs only after each modality was processed independently. This design is simple but limits cross-modal learning. Intermediate fusion through concatenation is more common today but still often too shallow to capture complex relationships. Attention-based fusion offers a promising alternative by allowing the model to prioritize the most informative data source for each case, yet these methods are still inconsistent and not fully optimized for clinical reliability.

### Missing Data Handling

Finally, the missing data problem remains one of the toughest barriers. Real-world medical records are messy and incomplete, but most multimodal models can only handle fully paired inputs. In practice, this forces researchers to exclude incomplete cases, reducing the amount of usable data and making deployment unrealistic. Some recent models attempt to adapt dynamically when one modality is missing, but the majority still fail in these situations.

Multimodal medical AI remains a promising field with clear potential to enhance clinical reasoning and decision support, but it is not yet ready for standard clinical deployment. Until issues like data scarcity, missing modalities, and fusion reliability are addressed, these systems will remain closer to experimental research tools than dependable components of routine medical care.

---

## Conclusion: From Explanation to Understanding

Multimodal AI in medicine has proven its performance value. Recent research shows substantial gains from combining radiology images with clinical text. But performance alone isn't enough for deployment.

Through this tutorial, we've seen how explainability techniques from attention visualization to gradient-based methods serve dual purposes: they make individual models trustworthy AND provide tools for studying representation learning across models.

The question of whether different multimodal medical AI systems converge on similar representations and what that convergence means remains largely unexplored. But the methods we've discussed offer a path forward: using attention visualization, attribution methods, and embedding analysis to empirically study representation similarity across models, architectures, and training paradigms.

The fusion layer gives us a natural window into multimodal reasoning. Early fusion architectures make this window larger and clearer. Now it's time to look through that window not just at individual models, but to compare what different models see and understand what representation convergence tells us about building reliable, trustworthy medical AI systems.

---

## References

<!-- Your citations will appear here automatically if you add them to the bibliography file -->

