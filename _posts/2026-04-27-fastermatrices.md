---
layout: distill
title: "Approximating Faster Transformers"
description: This post offers a comprehensive overview of sketching and sampling algorithms for DistilBERT
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
    url: "[https://en.wikipedia.org/wiki/Albert_Einstein](https://en.wikipedia.org/wiki/Albert_Einstein)"
    affiliations:
      name: Anonymous
  - name: Anonymous
    url: "[https://en.wikipedia.org/wiki/Boris_Podolsky](https://en.wikipedia.org/wiki/Boris_Podolsky)"
    affiliations: Anonymous
# must be the exact same name as your blogpost
bibliography: 2026-04-27-fastermatrices.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Equations
  - name: Images and Figures
    subsections:
      - name: Interactive Figures
  - name: Citations
  - name: Footnotes
  - name: Code Blocks
  - name: Diagrams
  - name: Tweets
  - name: Layouts
  - name: Other Typography?

# Below is an example of injecting additional post-specific styles.
# This is used in the 'Layouts' section of this post.
# If you use this post as a template, delete this _styles block.
_styles: |
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
  
  /* --- Shared Settings for Grids --- */
  .layout-seq, .layout-3-2 {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 15px;
    margin-bottom: 20px;
  }
  .layout-seq > div, .layout-3-2 > div {
    text-align: center;
    min-width: 150px;
  }
  .layout-seq img, .layout-3-2 img {
    width: 100%;
    height: auto;
    object-fit: cover;
    border-radius: 4px;
  }
  .layout-seq p, .layout-3-2 p {
    font-family: monospace;
    font-size: 0.9em;
    color: #666;
    margin-top: 5px;
    text-align: center;
  }
  /* Grid 1: 5 items in a row */
  .layout-seq > div {
    flex: 1 1 18%;
  }
  /* Grid 2: 3 top, 2 bottom */
  .layout-3-2 > div:nth-child(-n+3) {
    flex: 1 1 30%;
  }
  .layout-3-2 > div:nth-child(n+4) {
    flex: 1 1 45%;
  }

  /* --- NEW: Vertical Sequential Layout --- */
  .layout-vertical {
    display: flex;
    flex-direction: column; /* Stacks items vertically */
    gap: 40px;              /* Space between plots */
    margin: 30px 0;
  }
  .layout-vertical > div {
    width: 100%;            /* Forces full width */
    text-align: center;
  }
  .layout-vertical img {
    width: 100%;            /* Image fills the container */
    height: auto;
    border-radius: 6px;
    border: 1px solid rgba(0,0,0,0.1); /* Optional frame */
  }
  .layout-vertical p {
    font-family: monospace;
    font-size: 1.1em;       /* Slightly larger text for headers */
    color: #888;
    margin-top: 10px;
  }
---

Optimizing matrix multiplication is a problem as old as time. The product of two matrices $\mathbf{A}$ and $\mathbf{B}$ can be looked at as the inner product of rows of $\mathbf{A}$ with columns of $\mathbf{B}$ or the outer product of columns of $\mathbf{A}$ with rows of $\mathbf{B}$.

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/innerproduct.png' | relative_url }}" alt="Inner Product">

With traditional algorithms having a time complexity of $O(N^3)$, and researchers like Strassen <d-cite key="strassen1969gaussian"></d-cite> coming up with better recursive approaches, the time complexity is still only down to about $O(N^{2.81})$, which is still a pretty heavy task.

Ever since they were introduced, Transformers have gained tremendous popularity. However, a major bottleneck in the attention mechanism is matrix multiplication, which powers the core of the transformers. Optimizing matrix multiplication would mean an improvement in the speed of these models. This blog aims at utilizing several techniques from RandNLA to improve DistilBERT.

## Introduction
RandNLA is a field of linear algebra that uses randomization to improve very large-scale algorithms in linear algebra (see [RandNLA](https://arxiv.org/abs/2302.11474)). <d-cite key="murray2023randomized"></d-cite> The "Sketch and Solve" paradigm refers to using a sketching matrix to bring down the size of the problem and then solving the problem for a compressed size.

### Sketching for Matrix Multiplication

When applying the sketch and solve paradigm to the multiplication of two massive matrices $\mathbf{A}$ ($m \times n$) and $\mathbf{B}$ ($n \times p$), the sketching matrix $\mathbf{S}$ is used.

Instead of computing the exact product $\mathbf{C} = \mathbf{A}\mathbf{B}$, which requires $O(mnp)$ operations, we compute an approximate product using a sketching matrix $\mathbf{S}$ of size $k \times n$ (where $k \ll n$). The approximation is constructed by compressing the columns of $\mathbf{A}$ and the rows of $\mathbf{B}$:

$$\tilde{\mathbf{C}} = (\mathbf{A} \mathbf{S}^\top) (\mathbf{S} \mathbf{B})$$

Where:
* $\mathbf{A} \mathbf{S}^\top$ is an $m \times k$ matrix (a compressed version of $\mathbf{A}$).
* $\mathbf{S} \mathbf{B}$ is a $k \times p$ matrix (a compressed version of $\mathbf{B}$).

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sketch_and_solve.png' | relative_url }}" alt="Sketching Matrix">

For the approximation $\tilde{\mathbf{C}}$ to be accurate, the sketching matrix $\mathbf{S}$ must satisfy specific properties:

* The sketching matrix, when squared, acts as the identity matrix in expectation. If $\mathbf{S}$ is drawn from an appropriate distribution, then $\mathbb{E}[\mathbf{S}^\top \mathbf{S}] = \mathbf{I}_n$. This ensures that the approximation is an unbiased estimator of the true product: $\mathbb{E}[\tilde{\mathbf{C}}] = \mathbb{E}[\mathbf{A} (\mathbf{S}^\top \mathbf{S}) \mathbf{B}] = \mathbf{A}\mathbf{B}$.

* A matrix $\mathbf{S}$ has the property that if, for any vector $\mathbf{x}$, the norm satisfies:
  $(1 - \epsilon)\|\mathbf{x}\|_2^2 \le \|\mathbf{S}\mathbf{x}\|_2^2 \le (1 + \epsilon)\|\mathbf{x}\|_2^2$.

* A sketching matrix $\mathbf{S}$ should also satisfy:
  $\| \mathbf{A}\mathbf{B} - (\mathbf{A}\mathbf{S}^\top)(\mathbf{S}\mathbf{B}) \|_F \leq \epsilon \| \mathbf{A} \|_F \| \mathbf{B} \|_F$.
  Here we are bounding the error ensuring accurate results.

#### Sketching and Sampling

The sketches can be of various types, e.g., Gaussian, Count Sketch, Hadamard, and Learned sketches.
* **Gaussian sketches** involve projecting data using a matrix where entries are sampled independently from a normal distribution $\mathcal{N}(0, \frac{1}{k})$ to preserve distances [Sobczyk and Luisier, 2022].<d-cite key="sobczyk2022approximate"></d-cite>
* **Count Sketch** offers a sparser alternative by using hash functions to map rows to buckets and randomly flipping their signs [Clarkson and Woodruff, 2017].<d-cite key="clarkson2017low"></d-cite>
* **Hadamard-based sketches** (like the PHD matrix) combine randomized Hadamard transforms with uniform subsampling for structured efficiency [Clarkson and Woodruff, 2017].<d-cite key="clarkson2017low"></d-cite>

For the broader problem of matrix multiplication, not just sketching techniques but sampling algorithms have also been studied which have shown good promise.

The core idea behind the sampling algorithms is to select a subset of the original data. Several algorithms like uniform sampling, random sampling, priority sampling, threshold sampling, and leverage score sampling are popular. They basically "look" and find the "most important" parts of the entire matrix and then perform the multiplication on the smaller subset of the matrices.

* **Uniform Sampling** samples each row with the probability $p=k/n$ of selection, effectively treating every single row as equally important [Cohen et al., 2015].<d-cite key="cohen2015uniform"></d-cite>
* **Leverage Score Sampling** selects rows based on their statistical influence called leverage scores which are either calculated using SVD or QR-decomposition [Drineas et al., 2012].<d-cite key="drineas2012fast"></d-cite>
* **Priority and Threshold Sampling** is an innovative method which selects the rows based on their "importance," i.e., rows beyond a certain threshold are selected [Daliri et al., 2025].<d-cite key="daliri2025matrix"></d-cite>

### Sketch and Solve for Transformers

Transformers, as an architecture,<d-cite key="vaswani2017attention"></d-cite> is the industry standard because it has improved context as compared to previous architectures such as LSTM and RNNs, and provides the ability to parallelize the block which means faster training and inference.

Its architecture is split into two distinct blocks, both composed of a stack of $N$ identical layers:

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/transformers.png' | relative_url }}" alt="Transformers architecture">

**Encoder**

The Encoder’s primary function is to map an input sequence of symbols $(x_1, ..., x_n)$ into a rich, continuous representation $\mathbf{Z} = (z_1, ..., z_n)$.

Where each token is allowed to look at every other token available and hence it essentially understands the semantics.

**Decoder**
The Decoder generates the output sequence $(y_1, ..., y_m)$ one element at a time, being **auto-regressive** (consuming previously generated symbols as input for the next) instead of looking at the entire sentence like the encoder does.

The core logic which powers both encoder and decoder-based architectures is the attention mechanism; it basically creates a better, richer representation of our input context.

### Attention Mechanism

The input to the attention layer is our input embedding matrix $\mathbf{X}$. We project it using three learned weight matrices:
* **Query ($\mathbf{Q}$):** What the token is looking for.
* **Key ($\mathbf{K}$):** What the token identifies as.
* **Value ($\mathbf{V}$):** The actual content the token holds.

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/mha.png' | relative_url }}" alt="Multihead Attention">

The attention score is calculated using:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}
$$

* First, we calculate the dot product between the Query matrix and the transpose of the Key matrix; in a way, for a token, it calculates the similarity between every single token and every other token in the sequence.
* These are then normalized by a factor of $\sqrt{d_k}$ and then we apply **Softmax** over it, which converts the raw scores into probabilities.
* The normalized weights are then multiplied with the matrix $\mathbf{V}$.

Calculating this basically tells us the relationship of one token in a sequence to all the other tokens. However, the operation $\mathbf{Q}\mathbf{K}^\top$ is a very heavy operation.
To compute this, we have to perform operations proportional to the square of the sequence length ($N^2$) (assuming the length of our sequence to be $N$).

More specifically,
* If $N=512$, the attention matrix has roughly 260,000 entries.
* If $N=4096$, it jumps to over 16 million entries.

It is quite evident that improvement with $\mathbf{Q}\mathbf{K}^\top$ can lead to improvement in the overall speed of the transformers.

### Better Transformers

In regards to optimizing the attention matrix, a lot of work has already been done.
Instead of computing the full matrix, architectures like the **Sparse Transformer** enforce a fixed sparsity pattern, such as a sliding window where tokens only attend to their immediate neighbors [Child et al., 2019].<d-cite key="child2019generating"></d-cite> This reduces the complexity to approximately $O(N\sqrt{N})$.
A more dynamic approach was introduced by the **Reformer**<d-cite key="kitaev2020reformer"></d-cite>, which replaces exact dot-product attention with Locality-Sensitive Hashing (LSH) [Kitaev et al., 2020]. By grouping similar query and key vectors into the same hash buckets, the model computes attention only within these buckets, effectively dropping the complexity to $O(N \log N)$.

The **Linformer**<d-cite key="wang2020linformer"></d-cite> relies on the low-rank property of the attention matrix [Wang et al., 2020]. Linformer projects the Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices into a lower-dimensional space using learned linear projections ($\mathbf{E}$ and $\mathbf{F}$). By compressing the original $(N \times d)$ matrices into much smaller $(k \times d)$ matrices, the attention operation becomes linear $O(N)$ in time and space.

**LevAttention**<d-cite key="kannan2025levattention"></d-cite> utilizes the concept of leverage scores (generalized $f$-sensitivities) to identify a small "universal set" of keys that dominate the attention scores for *any* query [Kannan et al., 2024]. This method proves that a subset of high-leverage keys, independent of the sequence length, captures the vast majority of the attention mass, allowing for efficient $O(N \cdot \text{poly}(d/\epsilon))$ computation.

**Matrix Product Sketching via Coordinated Sampling** <d-cite key="daliri2025matrix"></d-cite> proposes estimating the product $\mathbf{Q}\mathbf{K}^\top$ directly using coordinated random sampling [Daliri et al., 2025]. Unlike traditional linear sketching (such as Johnson-Lindenstrauss projections) which can be inefficient for sparse data, coordinated sampling (specifically Priority Sampling) selects rows from $\mathbf{Q}$ and $\mathbf{K}$ based on their norms using a shared random seed.

In one way or another, all of these methods are trying to make the models faster by leveraging properties of the matrix itself or the matrix multiplication.

Our framework addresses three fundamental questions:

1. **Performance Comparison**: How do different efficient attention mechanisms perform across diverse Hindi NLP tasks?
2. **Cross-Attention Compatibility**: Are attention mechanisms interchangeable? Can a model trained with one attention mechanism perform well with another during inference?
3. **Task-Specific Patterns**: Do certain attention mechanisms work better for specific task types (classification, sequence labeling, multiple choice)?

To answer these questions, we developed a comprehensive pipeline that:
- Trains models with different attention mechanisms on multiple tasks
- Evaluates performance with both matching and mismatched attention mechanisms
- Provides detailed visualization and interpretability tools
- Tracks computational efficiency and environmental impact

### Experiment Setup

For the course of our experiments, we have picked the **DistilBERT**<d-cite key="sanh2019distilbert"></d-cite> (6 encoder blocks, 12 heads each) architecture which is a distilled version of **BERT** that retains 97% of its performance while being 40% lighter.

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/distbert.drawio (1).png' | relative_url }}" alt="DistilBert">

We chose DistilBERT-multilingual-cased as our foundation model due to its balance between performance and efficiency. With 6 transformer layers, 768 hidden dimensions, and 12 attention heads, it provides sufficient capacity while remaining computationally manageable for our large-scale experiments. Its multilingual training includes Hindi, making it suitable for our target language.

We then fine-tuned this model for various different tasks using our datasets and different attention mechanisms such as:

* priority sampling
* Leverage score based sampling
* Lewis score based sampling
* Learned sketches
* vanilla fine tuning
  
Each attention mechanism was implemented as a custom MultiHeadSelfAttention module, maintaining the same interface as the original while implementing the specific approximation strategy.

#### Training The Base Models
For each attention mechanism, we followed an identical five-step procedure:

**Step 1: Model Initialization**
We loaded the base DistilBERT model and configured it for each specific task type (sequence classification, token classification, or multiple choice).

**Step 2: Pre-Training Attention Injection**
Crucially, we injected the custom attention mechanism before fine-tuning. This ensured the model learned task-specific representations through the lens of that particular attention mechanism from the beginning of training.

The process involved iterating through all transformer layers, creating a custom attention instance, copying weights from the original attention module to preserve pre-trained knowledge, and then replacing the attention module. This surgical replacement maintained the model's architecture while changing the attention computation mechanism.

**Step 3: Task-Specific Fine-Tuning**
The datasets were taken from the  **IndicGLUE Benchmark** (Hindi Language subsets), courtesy of [AI4Bharat](https://indicnlp.ai4bharat.org/pages/indic-glue/).<d-cite key="kakwani2020indicnlpsuite"></d-cite>
We fine-tuned each model on five carefully selected Hindi NLP tasks:

| Task & Dataset | Description | Unique Labels | Hindi Example |
| :--- | :--- | :--- | :--- |
| **Sentiment Analysis**<br>`iitp-mr.hi` | Classifies movie reviews as positive, negative, or neutral. | `0` (Negative)<br>`1` (Neutral)<br>`2` (Positive) | *“यह फिल्म देखने लायक है, कहानी बहुत अच्छी है।”* |
| **News Classification**<br>`bbca.hi` | Classifies news articles into 14 distinct topics (e.g., Sports, India). | 14 Topics<br>(`india`, `sport`, `entertainment`, etc.) | *“भारतीय क्रिकेट टीम ने आज ऐतिहासिक जीत दर्ज की।”* |
| **Discourse Mode**<br>`md.hi` | Identifies the rhetorical role of a sentence in a narrative. | `Argumentative`<br>`Descriptive`<br>`Dialogic`<br>`Informative`<br>`Narrative` | *“एक बार की बात है, एक घना जंगल था।”*<br>(Narrative) |
| **Causal Reasoning (COPA)**<br>`copa.hi` | Selects the most plausible cause or effect for a given premise. | `0` (Choice 1)<br>`1` (Choice 2) | **Premise:** *“लड़के का पैर फिसल गया।”*<br>**Correct:** *“वह गिर गया।”* |
| **Named Entity Recognition**<br>`wiki-ner.hi` | Tags entities like Persons, Locations, and Organizations in text. | `O`, `B-PER`, `I-PER`<br>`B-ORG`, `I-ORG`<br>`B-LOC`, `I-LOC` | *“**राहुल** (B-PER) **गांधी** (I-PER) **दिल्ली** (B-LOC) में हैं।”* |
| **Section Title Prediction**<br>`wstp.hi` | Predicts the correct section title for a Wikipedia paragraph. | `0` (Title A)<br>`1` (Title B)<br>`2` (Title C)<br>`3` (Title D) | **Text:** (Paragraph about Cricket rules)<br>**Correct:** *“नियम” (Rules)* |


**Step 4: Model Deployment to HuggingFace Hub**
We uploaded all models with a consistent naming convention for easy programmatic access: shreshthamodi02/bert-{attention_type}-hindi-{task_name}

This created 25 publicly available models (5 attention types × 5 tasks).


All of the datasets were split into training and testing set and had a sequence lenght of 512

Naturally, we wanted our matrices to be a good representation of our entire corpus and hence we kept the values of k to be in the ranges: [64,128,256]

So for each of the dataset mentioned below, we have finetuned about 5 different models 

All the models have been fine-tuned on Kaggle using the GPU P100 and are available for inference freely on Hugging Face.

### Modified attention 

 The goal here is not to come up with a one-size-fits-all solution but rather to find out how each of these perform in practice and the pitfalls (if any) for several of these algorithms.



####  Learned Sketch

Adapting the idea from linformer, the assumption here is that the attention matrix is low rank. Hence, instead of computing the entire $N \times N$ matrix, we take smaller versions of the projected Key ($\mathbf{K}$) and Value ($\mathbf{V}$) matrices. [Wang et al., 2020].<d-cite key="wang2020linformer"></d-cite>

Those smaller projection matrices are defined as $\mathbf{E} \in \mathbb{R}^{k \times N}$ and $\mathbf{F} \in \mathbb{R}^{k \times N}$, where $k \ll N$ such that

$$\mathbf{K}_{proj} = \mathbf{E}\mathbf{K}, \quad \mathbf{V}_{proj} = \mathbf{F}\mathbf{V}$$

```latex
Input: Q (N x d), K (N x d), V (N x d)

Compressed matrices
   K_proj = E * K    (k x d)
   V_proj = F * V    (k x d)

Attention calculation
   Scores = (Q * K_proj^T) / sqrt(d)    (N x k)

weights
   Weights = Softmax(Scores)

Final product
   Output = Weights * V_proj   (N x d)
```



**Leverage Scores based attention**

This architecture is adopted from LevAttention, where Leverage scores are used to identify the most influential keys (rows of K), and we restrict attention computation to only these selected keys.
Instead of applying this to vision-based tasks as in the original paper, we adapt this approach for natural language tasks.

**Statistical Leverage Scores** are used here to identify the most "influential" keys in the sequence. High leverage indicates a key that is unique or critical to the subspace [Kannan et al., 2024].<d-cite key="kannan2025levattention"></d-cite>

For a matrix $\mathbf{K} \in \mathbb{R}^{n \times d}$ with $n > d$ and full column rank, the statistical leverage score of the $i$-th row $\mathbf{k}_i$ is formally defined as:

$$\tau_i(\mathbf{K}) = \mathbf{k}_i^\top (\mathbf{K}^\top \mathbf{K})^{-1} \mathbf{k}_i$$

This corresponds to the $i$-th diagonal element of the projection matrix $\mathbf{P} = \mathbf{K}(\mathbf{K}^\top \mathbf{K})^{-1}\mathbf{K}^\top$, which projects onto the column space of $\mathbf{K}$ [Drineas et al., 2012; Mahoney et al., 2011]. <d-cite key="drineas2012fast, mahoney2011randomized"></d-cite> The leverage scores satisfy $0 \leq \tau_i \leq 1$ and $\sum_{i=1}^n \tau_i = d$, providing a natural probability distribution over the rows of $\mathbf{K}$.

For a matrix $\mathbf{K} \in \mathbb{R}^{N \times d}$, the leverage score $\tau_i$ of the $i$-th row $\mathbf{k}_i$ is defined as:
$$\tau_i(\mathbf{K}) = \mathbf{k}_i (\mathbf{K}^\top \mathbf{K})^{-1} \mathbf{k}_i^\top$$



```latex
K (N x d), Q (N x d), V (N x d), budget k, damping (for stability) lambda

Leverage Scores for K:
   G = K^T * K + lambda * I
   G_inv = inverse(G)
   For each row i in K:
       tau_i = K_i * G_inv * K_i^T

Universal Set U calculation:
   U = {indices of the top-k largest tau_i}

Attention Mask M:
   M = matrix of -infinity (N x N)
   For all query positions i and key positions j:
       if j in U: M[i,j] = 0
       else: M[i,j] = -infinity

Attention:
   attention_scores = (Q * K^T) / sqrt(d) + M
   attention_weights = softmax(attention_scores)
   output = attention_weights * V
```

#### L1 weights based attention

We have also tried to use Lewis weights for the key selection instead of Leverage scores for the attention mechanism.

Lewis Weights are a generalization of leverage scores for $\ell_1$ norms. They provide a sensitivity measure for the $\ell_1$ norm, ensuring that the selected keys capture the geometry of the data distribution more robustly [Cohen & Peng, 2015].<d-cite key="cohen2015lp"></d-cite>

For a matrix $\mathbf{K} \in \mathbb{R}^{n \times d}$, the $\ell_1$ Lewis weights $w_i$ are defined as the unique values satisfying:
$$w_i = \tau_i(\mathbf{W}^{-1/2}\mathbf{K})$$
where $\mathbf{W} = \text{diag}(w_1, \ldots, w_n)$ is the diagonal weight matrix, and $\tau_i(\cdot)$ denotes the standard $\ell_2$ leverage score of the re-weighted matrix. 

We are using
``` latex
Input: K (N x d), Q (N x d), V (N x d)

Compute L1 Lewis Weights for K:
   Initialize: w_i = 1 for all i
   Iterate T times:
      K_tilde = K / sqrt(w)
      tau = LeverageScores(K_tilde)  
      w = sqrt(w * tau)
   S = {indices of largest w_i}

 Create Mask for Attention:
   M[i,j] = 0 if j in S else -inf

Compute Attention with Mask:
   attention_scores = (Q * K^T) / sqrt(d) + M
   attention_weights = softmax(attention_scores)
   output = attention_weights * V

```


#### Priority Sampling 

Here we are using adapting the priority sampling algorithm for distilbert.

We select the key vectors probabilistically, where the probability of selecting a key is proportional to its squared norm, using a threshold derived from the squared Frobenius norm of the key matrix [Daliri et al., 2025].<d-cite key="daliri2025matrix"></d-cite>

More formally,

Let $\mathbf{K} \in \mathbb{R}^{k\_len \times d}$ be the key matrix for a given head.
lendi  
The squared Frobenius norm of the key matrix is given by:
$$
\mathbf{A}_{\text{norm}}^2 = \|\mathbf{K}\|_F^2 = \sum_{i=1}^{k\_len} \|\mathbf{k}_i\|_2^2
$$

For a target sample size $k$, the threshold $\tau$ is:
$$
\tau = \frac{k}{\mathbf{A}_{\text{norm}}^2}
$$

For each key vector $\mathbf{k}_i$, generate a random hash $h_i \sim \text{Uniform}(0, 1)$. The key is selected if:
$$
h_i \leq \tau \cdot \|\mathbf{k}_i\|_2^2
$$
```latex
Input: K (N x d), k, random seed s

Generate Shared Randomness:
   h_i ~ Uniform(0, 1) for i in 1..N 

Compute Ranks :
   r_i = h_i / ||K_i||^2   // Lower rank = Higher priority

Determine Threshold:
   tau = (k+1)-th smallest value in r

Select Keys:
   S = {i | r_i <= tau}

Attention:
   Output = Softmax((Q * K_S^T) / sqrt(d)) * V_S
```

 This is what the structure for our training and inference models look like. For each task, we have fine-tuned 5 different attention models as given below and have used different types of attention methods for inference as well. The following table highlights the pair of attentions used for fine-tuneing and the corresponding pairs used for inference
 

| Training Model (Fine-tuned) ↓ \ Inference Method → | Vanilla Attention | Priority Sampling | Learned Sketch | LevAttention |
| :--- | :---: | :---: | :---: | :---: |
| **Vanilla Attention** | ✅ | ✅ | ✅ | ✅ |
| **Priority Sampling** | ✅ | ✅ | ✅ | ✅ |
| **Learned Sketch** | ✅ | ✅ | ✅ | ✅ |
| **L1 Attention** | ✅ | ✅ | ✅ | ✅ |
| **LevAttention** | ✅ | ✅ | ✅ | ✅ |

### Inference Methodology
**Grid Dimensions**:
- **5 Tasks**: sentiment analysis, NLI, NER, discourse classification, WSTP
- **5 Trained attention types**: What the model was originally trained with
- **5 Inference attention types**: What we use during evaluation

**Total Experimental Conditions**: 5 × 5 × 5 = 125

**Core Component 1: Argument Parser (`setup.py`)**

The `setup.py` file functions as the central configuration hub for the experimental framework. It orchestrates the following critical operations:
* **Argument Parsing:** Manages command-line arguments to distinguish between `trained_attn_type` and `inference_attn_type`.
* **Directory Management:** Automates the creation of directory structures to ensure results are stored systematically.
* **Dynamic Module Loading:** Handles the runtime initialization of specific attention modules based on the active experiment configuration.
* **Convention Mapping:** Resolves naming inconsistencies by mapping between training and inference identifiers.

**Core Component 2: Task-Specific Inference Scripts**

Each NLP task utilizes a dedicated inference script. While these scripts adhere to a unified architectural template, they are specialized to handle task-specific nuances:
1.  **`sentiment_analysis.py`:** Handles binary and multi-class classification, including the extraction of confidence scores.
2.  **`nli.py`:** Manages Natural Language Inference tasks, focusing on multiple-choice inference via pair-wise comparison.
3.  **`ner.py`:** Executes token-level classification and integrates `seqeval` metrics for rigorous entity-level evaluation.
4.  **`discourse.py`:** Performs multi-class classification tailored for discourse analysis, incorporating mode analysis.
5.  **`wstp.py`:** Processes multiple-choice questions with extended context windows (four options).

**Core Component 3: Metrics Collection Framework**

To ensure a holistic evaluation of the attention mechanisms, every inference run captures a comprehensive suite of metrics:
* **Performance Metrics:** Standard, task-appropriate evaluators (Accuracy, F1-Score, Precision, Recall), alongside confidence distribution analysis and calibration error checking.
* **Efficiency Metrics:** Computational profiling, including total inference time, per-batch latency, system throughput (samples/second), and GPU memory footprint (peak vs. average).
* **Environmental Metrics:** Ecological impact estimation, tracking carbon emissions and energy consumption via **CodeCarbon**.
* **Attention-Specific Metrics:** Intrinsic analysis of the mechanism, quantifying attention pattern entropy, sparsity ratios (for sparse attention variants), and head diversity.

### Results

Looking at the experiments and the attention mechanism, it is clear that there is some speedup while maintaining accuracy. However, interesting thing to note is that the benifits offered by the modifications are not uniform across all the tasks suggesting either the need for optimizing the code further or looking for alternative ways to make the methods work for more complex tasks


###### Sentiment Analysis

The task here was to classify the sentences into one of the three given labels.
<div style="font-family: 'Segoe UI', sans-serif; border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; max-width: 600px; margin-bottom: 25px;">
  
  <div style="background-color: #f6f8fa; padding: 12px 16px; border-bottom: 1px solid #e1e4e8; font-size: 12px; font-weight: 600; color: #57606a; text-transform: uppercase; display: flex; justify-content: space-between;">
    <span>Input Text</span>
    <span>Predicted Label</span>
  </div>

  <div style="padding: 12px 16px; border-bottom: 1px solid #eaecef; display: flex; justify-content: space-between; align-items: center; background-color: #fff;">
    <div style="font-size: 15px; color: #24292e; padding-right: 15px;">और खुश भी है।</div>
    <div style="background-color: #dafbe1; color: #1a7f37; border: 1px solid rgba(26, 127, 55, 0.2); padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; white-space: nowrap;">
      <span style="margin-right: 6px;">Positive</span><span>▲</span>
    </div>
  </div>

  <div style="padding: 12px 16px; border-bottom: 1px solid #eaecef; display: flex; justify-content: space-between; align-items: center; background-color: #fff;">
    <div style="font-size: 15px; color: #24292e; padding-right: 15px;">दानिश बड़ा होता है।</div>
    <div style="background-color: #f6f8fa; color: #57606a; border: 1px solid rgba(87, 96, 106, 0.2); padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; white-space: nowrap;">
      <span style="margin-right: 6px;">Neutral</span><span>•</span>
    </div>
  </div>

  <div style="padding: 12px 16px; display: flex; justify-content: space-between; align-items: center; background-color: #fff;">
    <div style="font-size: 15px; color: #24292e; padding-right: 15px;">लंबे समय तक अटकी रही।</div>
    <div style="background-color: #ffebe9; color: #cf222e; border: 1px solid rgba(207, 34, 46, 0.2); padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; display: inline-flex; align-items: center; white-space: nowrap;">
      <span style="margin-right: 6px;">Negative</span><span>▼</span>
    </div>
  </div>

</div>



A good idea to see how is our model predicting is using saliency plots. This is a relatively simple task which requires the model to understand the data

<div class="layout-vertical">
  
  <div>
    <img src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_vanilla/inference_vanilla/prediction_dashboard_Instance_Analysis.png' | relative_url }}">
    <p>1. Vanilla Attention</p>
  </div>

  <div>
    <img src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_priority/inference_priority_sampling/prediction_dashboard_Instance_Analysis.png' | relative_url }}">
    <p>2. Priority Sampling</p>
  </div>

  <div>
    <img src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_levattention/inference_levattention/prediction_dashboard_Instance_Analysis.png' | relative_url }}">
    <p>3. LevAttention</p>
  </div>

  <div>
    <img src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_l1_attention/inference_l1/prediction_dashboard_Instance_Analysis.png' | relative_url }}">
    <p>4. L1 attention</p>
  </div>
  <div>
    <img src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_learnedsketch/inference_learned_sketch/prediction_dashboard_Instance_Analysis.png' | relative_url }}">
    <p>5. Learned Sketch</p>
  </div>
  
  </div>

Let us also check the attentiion head's distribution for the attention mechanism
<div class="layout-vertical">
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_vanilla/inference_vanilla/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Vanilla Attention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_priority/inference_priority_sampling/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_levattention/inference_levattention/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_learnedsketch/inference_learned_sketch/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/sentiment_analysis/trained_l1_attention/inference_l1/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>

Looking at these, it is pretty clear that L1 and Priority based attention mechanism follow the vanilla most closely. Learned is the farthest from the original distribution. It could be due to the way we initalize the projection matrices.



| Trained_Attn   | Inference_Attn   |   Accuracy |
|:-------------------|:---------------|:-----------------|
| l1_attention   | levattention     |      0.565 |
 | l1_attention   | l1               |      0.565 |
| l1_attention   | vanilla          |      0.565 |
 | levattention   | vanilla          |      0.539 |
| levattention   | l1               |      0.539 |

Thus, the best attention mechanism for this task is likely L1 or Leverage score based sampling

Another interesting thing to note is that they have taken more time in training and inference as compared to the others

In terms of time, priority is the best

Given below is the table which contains the best performing training and inference attention method for sentiment analysis

###### NER

For NER, the task here was to categorize the parts of sentences after identifying them. Given below is what the example for NER sentence in our dataset looks like

<div style="font-family: sans-serif; line-height: 1.8; overflow-x: auto; white-space: nowrap; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background: #f9f9f9;"><span style="margin: 0 2px; color: #333;">जिंदगी</span><span style="margin: 0 2px; color: #333;">एक</span><span style="margin: 0 2px; color: #333;">रिहर्सल</span><span style="background-color: #FFD1DC; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>विष्णु</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">B-PER</span></span><span style="background-color: #FFD1DC; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>प्रभाकर</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-PER</span></span><span style="margin: 0 2px; color: #333;">द्वारा</span><span style="margin: 0 2px; color: #333;">रचित</span><span style="margin: 0 2px; color: #333;">कहानी</span><span style="margin: 0 2px; color: #333;">संग्रह</span><span style="margin: 0 2px; color: #333;">है।</span></div>
<br>


<div style="font-family: sans-serif; line-height: 1.8; overflow-x: auto; white-space: nowrap; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background: #f9f9f9;"><span style="background-color: #AEC6CF; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>न्यूटन</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">B-ORG</span></span><span style="background-color: #AEC6CF; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>का</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-ORG</span></span><span style="background-color: #AEC6CF; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>शीतलन</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-ORG</span></span><span style="background-color: #AEC6CF; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>का</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-ORG</span></span><span style="background-color: #AEC6CF; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>नियम</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-ORG</span></span></div>
<br>


<div style="font-family: sans-serif; line-height: 1.8; overflow-x: auto; white-space: nowrap; padding: 10px; border: 1px solid #e0e0e0; border-radius: 5px; background: #f9f9f9;"><span style="background-color: #C1E1C1; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>अण्टीगुआ</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">B-LOC</span></span><span style="background-color: #C1E1C1; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>और</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-LOC</span></span><span style="background-color: #C1E1C1; color: #333; padding: 4px 6px; border-radius: 6px; margin: 0 3px; display: inline-block;"><strong>बारबूडा</strong> <span style="font-size: 0.75em; opacity: 0.7; font-family: monospace;">I-LOC</span></span></div>
<br>




  *Attention heads Distribution Plots*
<div class="layout-vertical">
   <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_vanilla/inference_vanilla/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Vanilla</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_priority/inference_priority_sampling/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_levattention/inference_levattention/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_learnedsketch/inference_learned_sketch/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_l1_attention/inference_l1/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>


*NER Tags*
<div class="layout-vertical">
   <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_vanilla/inference_vanilla/ner_confidence_wrapped_.png' | relative_url }}">
    <p>Vanilla</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_priority/inference_priority_sampling/ner_confidence_wrapped_.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_levattention/inference_levattention/ner_confidence_wrapped_.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_learnedsketch/inference_learned_sketch/ner_confidence_wrapped_.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/ner/trained_l1_attention/inference_l1/ner_confidence_wrapped_.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>



| Trained_Attn   | Inference_Attn   |   Accuracy |
|:-------------------|:---------------|:-----------------|
| priority       | vanilla          |      0.944 |
| priority       | l1               |      0.944 |
| priority       | levattention     |      0.944 |
| levattention   | vanilla          |      0.941 |
| l1_attention   | vanilla          |      0.941 |

Unsurprisingly, learned sketch performed worse as compared to the other methods here as well. Priority sampling performed the best in terms of accuracy and time and likely could be a better choice for tasks like these as compared to the other methods

Interestingly enough, this task benifits a lot from the optimized attention mechanism

Given below is table which shows the best performing training and inference attention pair for NER

###### WSTP
Taken from the wikipedia dataset, we used our models to predict the title of the given text. Since this is multiple choice, it is one of the more complex tasks

<div style="font-family: sans-serif; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; margin-bottom: 30px; background-color: white; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
  
  <div style="color: #888; font-size: 12px; font-weight: bold; text-transform: uppercase; margin-bottom: 10px;">
    Sample 1
  </div>

  <div style="font-size: 16px; line-height: 1.6; color: #333; margin-bottom: 20px; padding-bottom: 20px; border-bottom: 1px solid #eee;">
    उन्होंने सबसे पहले ब्रिटिश राज के दौरान पूर्ण स्वराज की मांग उठाई। लोकमान्य तिलक ने जनजागृति का कार्यक्रम पूरा करने के लिए महाराष्ट्र में गणेश उत्सव तथा शिवाजी उत्सव सप्ताह भर मनाना प्रारंभ किया। इन त्योहारों के माध्यम से जनता में देशप्रेम और अंग्रेजों के अन्यायों के विरुद्ध संघर्ष का साहस भरा गया।
  </div>

  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
    
  <div style="background-color: #f7fafc; border: 1px solid #cbd5e0; color: #4a5568; padding: 10px 15px; border-radius: 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
      <span><span style="opacity: 0.6; margin-right: 8px;">A</span> भारतीय राष्ट्रीय कांग्रेस</span>
    </div>

  <div style="background-color: #f7fafc; border: 1px solid #cbd5e0; color: #4a5568; padding: 10px 15px; border-radius: 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
      <span><span style="opacity: 0.6; margin-right: 8px;">B</span> राजनीतिक यात्रा</span>
    </div>

  <div style="background-color: #f7fafc; border: 1px solid #cbd5e0; color: #4a5568; padding: 10px 15px; border-radius: 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
      <span><span style="opacity: 0.6; margin-right: 8px;">C</span> माण्डले में कारावास</span>
    </div>

  <div style="background-color: #e6fffa; border: 1px solid #38b2ac; color: #234e52; font-weight: bold; padding: 10px 15px; border-radius: 6px; font-size: 14px; display: flex; justify-content: space-between; align-items: center;">
    <span><span style="opacity: 0.6; margin-right: 8px;">D</span> सामाजिक योगदान और विरासत</span>
      <span>&#10003;</span>
    </div>

  </div>
</div>

*Attention Heads Distribution Plots*
<div class="layout-vertical">
   <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/wstp/trained_vanilla/inference_vanilla/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Vanilla</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/wstp/trained_priority/inference_priority_sampling/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/wstp/trained_levattention/inference_levattention/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/wstp/trained_learnedsketch/inference_learned_sketch/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/wstp/trained_l1_attention/inference_l1/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>

This is the task where our modifed attention took a lot more time as compared to the vanilla mechanism which likely means that similar tasks will not be benifitted from optimizing attention here

Given below is the table which talks about the top inference and training attention pair in terms of accuracy

| Trained_Attn   | Inference_Attn   |   Accuracy |
|:---------------|:-----------------|-----------:|
| levattention   | vanilla          |      0.714 |
| levattention   | l1               |      0.714 |
| levattention   | levattention     |      0.714 |
| vanilla        | vanilla          |      0.71  |
| vanilla        | levattention     |      0.71  |

However, out of all of the attention mechanism, the lev attention took the least amount of time and hence in terms of time and accuracy, lev attention is the better choice for more complex tasks. 

###### NLI

NLI is also taken up from the AI4Bharat series of datasets. Here, we look at the given text called premise and try to identify the cause and the effect relationship

<div style="font-family: sans-serif; max-width: 700px;">

  <div style="border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); background: white;">
    <div style="background-color: #fffbeb; border-bottom: 1px solid #fcd34d; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center;">
      <span style="color: #d97706; font-weight: 800; font-size: 12px; letter-spacing: 1px; text-transform: uppercase;">Find the CAUSE</span>
      <span style="font-size: 16px;"></span>
    </div>
    <div style="padding: 25px; text-align: center; border-bottom: 1px solid #f3f4f6;">
      <div style="font-size: 11px; color: #9ca3af; margin-bottom: 8px; text-transform: uppercase; font-weight: 600;">Premise</div>
      <div style="font-size: 18px; font-weight: 500; color: #1f2937; line-height: 1.5;">मेरे शरीर ने घास पर छाया डाली।</div>
    </div>
    <div style="display: flex; flex-direction: row;">
      <div style="flex: 1; padding: 20px; background-color: #ecfdf5; border-right: 1px solid #e5e7eb; opacity: 1.0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: #064e3b;">
          <span style="font-size: 11px; font-weight: bold;">OPTION 1</span>
          <span style="font-weight: bold; font-size: 16px;">&#10003;</span>
        </div>
        <div style="font-size: 14px; color: #064e3b; font-weight: bold; line-height: 1.5;">सूरज उग रहा था।</div>
      </div>
      <div style="flex: 1; padding: 20px; background-color: #f9fafb;  opacity: 0.6;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: #9ca3af;">
          <span style="font-size: 11px; font-weight: bold;">OPTION 2</span>
          <span style="font-weight: bold; font-size: 16px;"></span>
        </div>
        <div style="font-size: 14px; color: #9ca3af; font-weight: normal; line-height: 1.5;">घास काटी गई।</div>
      </div>
    </div>
  </div>

  <div style="border: 1px solid #e5e7eb; border-radius: 8px; overflow: hidden; margin-bottom: 30px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); background: white;">
    <div style="background-color: #eff6ff; border-bottom: 1px solid #bfdbfe; padding: 12px 20px; display: flex; justify-content: space-between; align-items: center;">
      <span style="color: #2563eb; font-weight: 800; font-size: 12px; letter-spacing: 1px; text-transform: uppercase;">Find the EFFECT</span>
      <span style="font-size: 16px;"></span>
    </div>
    <div style="padding: 25px; text-align: center; border-bottom: 1px solid #f3f4f6;">
      <div style="font-size: 11px; color: #9ca3af; margin-bottom: 8px; text-transform: uppercase; font-weight: 600;">Premise</div>
      <div style="font-size: 18px; font-weight: 500; color: #1f2937; line-height: 1.5;">चिकित्सक ने मरीज को गलत बताया।</div>
    </div>
    <div style="display: flex; flex-direction: row;">
      <div style="flex: 1; padding: 20px; background-color: #ecfdf5; border-right: 1px solid #e5e7eb; opacity: 1.0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: #064e3b;">
          <span style="font-size: 11px; font-weight: bold;">OPTION 1</span>
          <span style="font-weight: bold; font-size: 16px;">&#10003;</span>
        </div>
        <div style="font-size: 14px; color: #064e3b; font-weight: bold; line-height: 1.5;">मरीज ने चिकित्सक के खिलाफ कदाचार का मुकदमा दायर किया।</div>
      </div>
      <div style="flex: 1; padding: 20px; background-color: #f9fafb;  opacity: 0.6;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px; color: #9ca3af;">
          <span style="font-size: 11px; font-weight: bold;">OPTION 2</span>
          <span style="font-weight: bold; font-size: 16px;"></span>
        </div>
        <div style="font-size: 14px; color: #9ca3af; font-weight: normal; line-height: 1.5;">मरीज ने चिकित्सक को गोपनीय जानकारी दी।</div>
      </div>
    </div>
  </div>

</div>

*Attention Head Distribution Plots*
<div class="layout-vertical">
   <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/copa_hindi/trained_vanilla/inference_vanilla/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Vanilla</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/copa_hindi/trained_priority/inference_priority_sampling/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/copa_hindi/trained_levattention/inference_levattention/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/copa_hindi/trained_learnedsketch/inference_learned_sketch/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/copa_hindi/trained_l1_attention/inference_l1/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>

Given below is  the table for accuracy and the type of the attention model used for training and inference

| Trained_Attn   | Inference_Attn    |   Accuracy |
|:---------------|:------------------|-----------:|
| vanilla        | priority_sampling |      0.563 |
| levattention   | priority_sampling |      0.563 |
| l1_attention   | learned_sketch    |      0.528 |
| priority       | vanilla           |      0.526 |
| priority       | l1                |      0.526 |

Priority comes out as a better choice in terms of time and inference for tasks similar to NLI

###### Discourse

With discourse we are trying to understand the sementics of a sentence and the the type of the sentence

<div style="font-family: sans-serif; max-width: 700px; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; margin-bottom: 30px;">
  
  <div style="background: #fcfcfc; padding: 10px 15px; border-bottom: 1px solid #eee; color: #888; font-size: 11px; font-weight: bold; letter-spacing: 1px; text-transform: uppercase;">
   
  </div>

  <div style="padding: 15px; border-bottom: 1px solid #f0f0f0; display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; background-color: #fff;">
    <div style="font-size: 15px; line-height: 1.6; color: #333; width: 75%;">एक कबूतर पंख फडफ़ड़ाता हुआ कहवाख़ाने के अन्दर आया और कुछ लोग मिलकर उसे बाहर निकालने की कोशिश करने लगे।</div>
    <div style="background-color: #e3f2fd; color: #0d47a1; border: 1px solid #90caf9; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; white-space: nowrap; display: flex; align-items: center;">
      <span style="margin-right: 6px;">Narrative</span><span>📖</span>
    </div>
  </div>

  <div style="padding: 15px; border-bottom: 1px solid #f0f0f0; display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; background-color: #fff;">
    <div style="font-size: 15px; line-height: 1.6; color: #333; width: 75%;">हर महीने दस दस के पाँच नोट वो अपने ख़फ़ीफ़ तौर पर काँपते हुए हाथों से पकड़ता और अपने पुराने वज़ा के लंबे कोट की अंदरूनी जेब में रख लेता।</div>
    <div style="background-color: #f3e5f5; color: #4a148c; border: 1px solid #ce93d8; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; white-space: nowrap; display: flex; align-items: center;">
      <span style="margin-right: 6px;">Descriptive</span><span>👁️</span>
    </div>
  </div>

  <div style="padding: 15px; border-bottom: 1px solid #f0f0f0; display: flex; align-items: flex-start; justify-content: space-between; gap: 15px; background-color: #fff;">
    <div style="font-size: 15px; line-height: 1.6; color: #333; width: 75%;">आख़िर शरीफ़ ख़ान-दान से तअल्लुक़ है ”</div>
    <div style="background-color: #fce4ec; color: #880e4f; border: 1px solid #f48fb1; padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; white-space: nowrap; display: flex; align-items: center;">
      <span style="margin-right: 6px;">Dialogue</span><span>💬</span>
    </div>
  </div>

</div>


*Attention Heads Distribution Plots*
<div class="layout-vertical">
   <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_vanilla/inference_vanilla/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Vanilla</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_priority/inference_priority_sampling/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_levattention/inference_levattention/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_learnedsketch/inference_learned_sketch/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_l1_attention/inference_l1/attention_histograms_Global_Dist.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>

*Saliency Plots*
<div class="layout-vertical">
   <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_vanilla/inference_vanilla/saliency_class4_Discourse_Saliency.png' | relative_url }}">
    <p>Vanilla</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_priority/inference_priority_sampling/saliency_class4_Discourse_Saliency.png' | relative_url }}">
    <p>Priority Sampling</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_levattention/inference_levattention/saliency_class4_Discourse_Saliency.png' | relative_url }}">
    <p>Levattention</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_learnedsketch/inference_learned_sketch/saliency_class1_Discourse_Saliency.png' | relative_url }}">
    <p>Learned Sketch</p>
  </div>
  <div>
    <img class="rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_l1_attention/inference_l1/saliency_class4_Discourse_Saliency.png' | relative_url }}">
    <p>L1 attention</p>
  </div>
</div>

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/discourse/trained_vanilla/inference_vanilla/discourse_dashboard_Mode_Probabilities.png' | relative_url }}" alt="dashboard marker">


Given below are the inference and training attention pairs which are best performing for the given task


| Trained_Attn   | Inference_Attn   |   Accuracy |
|:---------------|:-----------------|-----------:|
| levattention   | vanilla          |      0.794 |
| levattention   | l1               |      0.794 |
| levattention   | levattention     |      0.794 |
| l1_attention   | levattention     |      0.794 |
| l1_attention   | l1               |      0.794 |

Learned sketch performed poorly here as well. For tasks similar to discourse, using a vanilla fine-tuned model and using other models for inference is a better idea. Using priority sampling based attention is also a good idea for these

There is no one-sized fits all method really as we can see. However, there are some methods which on average are better than the others let us look at them and compare the accuracies and the time taken for each of these

##### Comparing accuracies across tasks 

<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/attention_accuracy.png' | relative_url }}" alt="dashboard marker">

As you can see, in terms of accuracy, for simpler tasks all of the methods except learned sketch achieve good accuracy, one of the reasons for that could be the fact that the dimensions that we are sketching for could be too little to capture the nuances effectively. However, if we were to increase the dimensions then it would take significantly more time and would not achieve the promised speedup. An interesting experiment would be to reproduce this on a larger dataset with a heavier model to see if there are more speedups as compared to DistilBert architecture

Levattention is a method which consistently matches the accuracy given by vanilla attention. For simpler tasks, all of the attention mechanisms seem to give satisfactory results but for tasks which are complex, L1 and Leverage score based attention mechanisms outperform everyone else

##### Comparing the inference across tasks
<img class="img-fluid rounded" src="{{ 'assets/img/2026-04-27-fastermatrices/corrected_time_diff_heatmap.png' | relative_url }}" alt="Time Spent">

In terms of the time taken here, this graph shows the change in the percentage time per attention as compared to the vanilla.

As accurate as leverage score based sampling is, it seems to take even more time than vanilla based attention mechanism. One of the reasons for that could be the leverage score calculation; We are yet to explore more efficient ways to calculate leverage scores. Since we are applying the leverage scores based selection only for keys, it might be the case that we need to apply it either for values or the query matrix. To utilize the full potential of this method, another interesting thing would be to also look at sharing the leverage scores across the layers and the head

Priority sampling is a promising method as it provides speedup for simpler tasks yet it retains considerable accuracy as compared to the other methods. 

In terms of time, L1 sampling based attention is also not far behind Leverage score based sampling and hence we need to look more into the optimization of these mechanisms to better suit the architectures.

Learned sketch offers very promising results in terms of time but the accuracy stagnates across the epoch suggesting some errors with the implementation. Regardless, the sketching based mechanisms offer greater time speedup

In order to reap the full benifits of these methods, inputs with larger context lenght should be considered which is where they will give considerable speedup
