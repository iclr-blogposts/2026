// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-home",
    title: "home",
    section: "Navigation",
    handler: () => {
      window.location.href = "/2026/";
    },
  },{id: "nav-about",
          title: "about",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/about/";
          },
        },{id: "nav-blog",
          title: "blog",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/blog/";
          },
        },{id: "nav-call-for-blogposts",
          title: "call for blogposts",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/call/";
          },
        },{id: "nav-submitting",
          title: "submitting",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/submitting/";
          },
        },{id: "nav-reviewing",
          title: "reviewing",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/2026/reviewing/";
          },
        },{id: "dropdown-lt-strong-gt-2027-lt-strong-gt",
              title: "&lt;strong&gt;2027&lt;/strong&gt;",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2027/";
              },
            },{id: "dropdown-lt-strong-gt-2026-lt-strong-gt",
              title: "&lt;strong&gt;2026&lt;/strong&gt;",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2026/";
              },
            },{id: "dropdown-2025",
              title: "2025",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2025/";
              },
            },{id: "dropdown-2024",
              title: "2024",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2024/";
              },
            },{id: "dropdown-2023",
              title: "2023",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blogposts.github.io/2023/";
              },
            },{id: "dropdown-2022",
              title: "2022",
              description: "",
              section: "Dropdown",
              handler: () => {
                window.location.href = "https://iclr-blog-track.github.io/home/";
              },
            },{id: "post-why-ai-evaluations-need-statistical-rigor",
        
          title: "Why AI Evaluations Need Statistical Rigor",
        
        description: "AI evaluations often rely on single-run scores even though models, agents, and judges are inherently stochastic, making many reported differences unstable. This post surveys statistical tools—error bars, reliability measures, Bayesian models—that show and help manage this variance. Overall, it highlights how incorporating established statistical practices can make evaluations more trustworthy and informative.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/why-ai-evaluations-need-error-bars/";
          
        },
      },{id: "post-computer-use-survey-a-visual-survey-of-computer-use-agents",
        
          title: "Computer Use Survey - A Visual Survey of Computer Use Agents",
        
        description: "In recent years, AI systems operating on the web and in computer environments have become a major topic of interest for both academia and industry. The goal of this blog is to provide an interesting and interactive survey of historical and recent works on computer use agents. We define key terms used in the literature, catalogue the expansive list of environments and datasets, discuss the evolution of the methodologies, and assess both today’s landscape and possible paths forward.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/web-agent/";
          
        },
      },{id: "post-wait-do-we-need-to-wait-revisiting-budget-forcing-for-sequential-test-time-scaling",
        
          title: "Wait, Do We Need to Wait? Revisiting Budget Forcing for Sequential Test-Time Scaling...",
        
        description: "This blog revisits budget forcing, a sequential test-time scaling technique for reasoning models by controlling when it continues thinking versus when it must answer. We evaluate how well the method transfers across model types, including non-reasoning models, and whether alternative keywords work. We provide practical guidelines for using the technique.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/wait-do-we-need-to-wait/";
          
        },
      },{id: "post-visualizing-llm-latent-space-geometry-through-dimensionality-reduction",
        
          title: "Visualizing LLM Latent Space Geometry Through Dimensionality Reduction",
        
        description: "In this blog post, we extract, process, and visualize latent state geometries in Transformer-based language models through dimensionality reduction to build a better intuition of their internal dynamics. We demonstrate experiments with GPT-2 and LLaMa models, uncovering interesting geometric patterns in their latent spaces. Notably, we identify a clear separation between attention and MLP component outputs across intermediate layers, a pattern not documented in prior work to our knowledge.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/vis-llm-latent-geometry/";
          
        },
      },{id: "post-what-and-what-not-are-calibrated-probabilities-actually-useful-for",
        
          title: "What (and What Not) are Calibrated Probabilities Actually Useful for?",
        
        description: "This blogpost clarifies the practical usefulness of having a model with calibrated probabilities, something that is not often clearly stated in the calibration literature. We show that a calibrated model can be relied on to estimate average loss/reward, however, good calibration does not mean that a model is useful for per-sample decision making.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/useful-calibrated-uncertainties/";
          
        },
      },{id: "post-unigramlm-an-attempt-at-writing-the-missing-manual",
        
          title: "UnigramLM - An Attempt at Writing the Missing Manual",
        
        description: "This post is my attempt to write down the UnigramLM tokenization algorithm cleanly and explicitly because no such derivation appears to exist and I think understanding the theory behind the method could help us improve it. I&#39;ll formalize the generative model around which the algorithm is based, derive the EM updates, explain why pruning is needed (and how it&#39;s done), and point out the spots where the practical implementation defined by the SentencePiece library diverges from the pretty mathematical models. I hope this post provides a new lens through which to look at the UnigramLM tokenization algorithm while pointing out some interesting potential extensions/revisions to the current implementation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/unigramlm-manual/";
          
        },
      },{id: "post-trade-offs-in-llm-compute-for-reasoning-intensive-information-retrieval",
        
          title: "Trade-offs in LLM Compute for Reasoning-Intensive Information Retrieval",
        
        description: "The BRIGHT benchmark (ICLR 2025 Spotlight) revealed that reasoning-intensive information retrieval requires LLM-augmented pipelines, but this raises a critical resource allocation question: where should computational budget be invested for maximum effectiveness? We conduct a systematic study on BRIGHT using the Gemini 2.5 model family, evaluating trade-offs across model strength, inference-time thinking depth, and reranking depth. Our controlled experiments quantify the marginal gains of allocating compute to query expansion versus reranking, providing practical guidance for optimizing LLM-based retrieval systems on reasoning-intensive tasks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/trade-offs-in-llm-compute-for-reasoning-intensive-information-retrieval/";
          
        },
      },{id: "post-tracing-the-principles-behind-modern-diffusion-models",
        
          title: "Tracing the Principles Behind Modern Diffusion Models",
        
        description: "Diffusion models can feel like a jungle of acronyms, but the core idea is simple: start from noise and gradually move a cloud of samples until it looks like real data. This post gives an intuition-first tour showing that DDPMs, score-based models, and flow matching are the same recipe with different prediction targets, all rooted in the change-of-variable rule from calculus and powered by one shared “conditional trick” that turns learning into supervised regression. Finally, we zoom out to the speed problem and show how flow map models aim to replace many tiny denoising steps with a few big, accurate jumps toward real-time generation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/tracing-principles-behind-modern-diffusion-models/";
          
        },
      },{id: "post-the-evolution-of-flashattention",
        
          title: "The Evolution of FlashAttention",
        
        description: "We present a mathematical &amp; technical overview of FlashAttention and its evolution across versions 1 to 4. We explain why IO-aware design became central to scalable transformers and how these kernels shape modern long-context LLMs as memory patterns and hardware limits shift. We then describe the changes across versions with Triton examples and place these kernels in the context of recent work on efficient attention. We close by outlining principles that can guide the next generation of attention algorithms.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/the-evolution-of-flashattention/";
          
        },
      },{id: "post-artistic-style-and-the-play-of-neural-style-representations",
        
          title: "Artistic Style and the Play of Neural Style Representations",
        
        description: "How do neural networks percieve the complex human construct of artistic style? We explore the dynamic interplay between diverse machine representations of style and style definitions. We reveal a profound divergence where models often reject established historical narratives in favour of their own perceptual truths.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/style-representations/";
          
        },
      },{id: "post-where-39-s-the-chicken-unpacking-spatial-awareness-in-vision-language-models",
        
          title: "Where&#39;s the Chicken? Unpacking Spatial Awareness in Vision-Language Models",
        
        description: "Modern vision-language models (VLMs) have achieved impressive success in recognizing and describing visual content, yet they continue to struggle with understanding spatial relationships. The limitation persists even with massive data and model scaling, suggesting that the root of the problem lies in the architecture and training objective rather than data alone. This post examines the underlying causes and discusses why recent proposed fixes, while promising, remain insufficient to achieve robust spatial reasoning.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/spatial-awareness/";
          
        },
      },{id: "post-don-39-t-look-up-every-token-escaping-quadratic-complexity-via-geometric-patterns-and-algorithms",
        
          title: "Don&#39;t Look Up (Every Token): Escaping Quadratic Complexity via Geometric Patterns and Algorithms",
        
        description: "Large Language Models (LLMs) have brought about a significant change in the field of artificial intelligence, where they have transitioned in scope from being specialized research tools to common resources that drive the next generation of software. With increasing model parameters and training data, LLMs demonstrate new abilities in reasoning, code generation, and solving complex problems that were once considered unattainable. However, scaling these models effectively for long-context applications uniquely poses a challenge. This is primarily due to the inherent limitations of the self-attention mechanism, which has quadratic time complexity. This quadratic bottleneck hinders applications for long documents, high-resolution images, and large codebases, among others. However, what is interesting to observe is that effectively only a few parameters are used in token computation, and most calculations are sparse. Hence, sparsity emerges as an effective solution to this problem. Rather than relying on the entire attention matrix, one can utilize an approximate or sparse version of attention to achieve almost the same results much faster. The backbone of this approach is the idea that tokens do not require the entire context; they only need local context, and thus, most of the computation carried out is wasteful. In this blog, we analyze the types of attention patterns that emerge and how to use them to our advantage for faster and efficient LLMs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/sparsity/";
          
        },
      },{id: "post-using-graph-neural-networks-in-reinforcement-learning-a-practical-guide",
        
          title: "Using Graph Neural Networks in Reinforcement Learning: A Practical Guide",
        
        description: "Graph Neural Networks (GNNs) have achieved excellent results for modelling relational data in many supervised learning domains. However, much fewer works have explored their potential in Reinforcement Learning (RL) despite the ubiquity of practical problems defined over graphs. In this blog post, we discuss how GNNs can be effectively integrated in Deep RL frameworks, covering crucial design decisions and practical implementation concerns. In doing so, we hope to facilitate unlocking new capabilities for RL agents to reason in graph-structured environments with dynamic action spaces and varying input sizes.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/rl-with-gnns/";
          
        },
      },{id: "post-revisiting-the-nethack-learning-environment",
        
          title: "Revisiting The NetHack Learning Environment",
        
        description: "The NetHack Learning Environment (NLE) was proposed as a challenging benchmark to test an agents abilities to perform complex reasoning over long time horizons in a stochastic, partially-observed, procedurally generated setting. To date, no approach, including those based on reinforcement learning, using large pretrained models, using handcoded symbolic agents, imitating expert trajectories or any hybrid method has achieved significant progress towards completing the game. We take a deeper look into the mechanics and interface of the NLE and show that much of the complexity of NetHack is inaccessible due to constraints on the observation and action spaces. We propose a series of modifications and show that they meaningfully improve performance on the NLE.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/revisiting-the-nle/";
          
        },
      },{id: "post-rethinking-the-diffusion-model-from-a-langevin-perspective",
        
          title: "Rethinking the Diffusion Model from a Langevin Perspective",
        
        description: "Diffusion models are often introduced from multiple perspectives—such as VAEs, score matching, or flow matching—accompanied by dense and technically demanding mathematics that can be difficult for beginners to grasp. One classic question is How does the reverse process invert the forward process to generate data from pure noise? This article systematically organizes the diffusion model from a fresh Langevin perspective, offering a simpler, clearer, and more intuitive answer. We also address the following questions 1. How can ODE-based and SDE-based diffusion models be unified under a single framework? 2. Why are diffusion models theoretically superior to ordinary VAEs? 3. Why is flow matching not fundamentally simpler than denoising or score matching, but equivalent under maximum-likelihood? We demonstrate that the Langevin Perspective offers clear and straightforward answers to these questions, providing pedagogical value for both learners and experienced researchers seeking deeper intuition.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/rethinking-diffusion-langevin/";
          
        },
      },{id: "post-dynamic-parameter-reuse-augments-reasoning-via-latent-chain-of-thought",
        
          title: "Dynamic Parameter Reuse Augments Reasoning via Latent Chain of Thought",
        
        description: "Standard language models often rely on massive parameter counts for their performance, utilizing each parameter only once per inference pass. This prompts consideration of recurrent structures, where models reuse parameters across sequential time, depth, or training progression to achieve improved performance and reduced training cost. We draw connections in the landscape of parameter reuse, from growing models via stacking to recurrent looping, and postulate that these architectural priors act as a form of Latent Chain of Thought (LCoT), allowing models to reason in a continuous state space. By shifting towards deeper and dynamic computation, grown and recurrent architectures offer a path toward improved reasoning in compact networks, ascending beyond scaling laws of standard architectures.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/recur-refine-reason/";
          
        },
      },{id: "post-probabilistic-circuits-for-uncertainty-quantification",
        
          title: "Probabilistic Circuits for Uncertainty Quantification",
        
        description: "Deep learning models struggle with epistemic uncertainty quantification, often exhibiting blind confidence on out-of-distribution data. This work reviews Probabilistic Circuits (PCs) as a versatile framework for rigorous, tractable reasoning. PCs model the joint probability distribution and by enforcing structural constraints, specifically smoothness, decomposability, and determinism, they allow for the exact computation of marginals, conditionals, and moments in polynomial time without retraining. We discuss the suitability of PCs for Uncertainty Quantification, describing their advantages and highlighting their potential for tractable UQ in high-dimensional problems.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/probabilistic-circuits-for-uncertainty-quantification/";
          
        },
      },{id: "post-extracting-model-precision-from-20-logprobs",
        
          title: "Extracting Model Precision from 20 Logprobs",
        
        description: "We demonstrate that the internal floating-point precision of language models can be inferred from API-exposed logprobs.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/precision-extraction/";
          
        },
      },{id: "post-effect-of-parallel-environments-and-rollout-steps-in-ppo",
        
          title: "Effect of Parallel Environments and Rollout Steps in PPO",
        
        description: "This blog post explores batch size in PPO-what happens when we increase the number of parallel environments versus the number of rollout steps, while keeping the total samples per update fixed. We discuss how this affects bias and variance in gradient estimation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/ppo-batch-size/";
          
        },
      },{id: "post-faster-svd-via-accelerated-newton-schulz-iteration",
        
          title: "Faster SVD via Accelerated Newton-Schulz Iteration",
        
        description: "Traditional SVD algorithms rely heavily on QR factorizations, which scale poorly on GPUs. We show how the recently proposed Chebyshev-Accelerated Newton-Schulz (CANS) iteration can replace them and produce an SVD routine that is faster across a range of matrix types and precisions.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/polar-svd/";
          
        },
      },{id: "post-performative-prediction-made-practical",
        
          title: "Performative Prediction made practical",
        
        description: "Performative Prediction studies settings where deploying a model induces a distribution shift in the data with the aim of building robust and good-performing models under these post-deployment effects. Most existing work in this area is theoretical and relies on strict assumptions to converge to those models, making the resulting techniques difficult to apply in practice and limiting their accessibility to the broader Machine Learning (ML) community. In this blog post, we use visualization techniques 1) to provide an intuitive explanation of Performative Prediction and 2) to extract practical insights for studying convergence when theoretical assumptions do not hold.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/performative-prediction/";
          
        },
      },{id: "post-language-as-a-window-into-the-mind-how-nlp-and-llms-advance-human-sciences",
        
          title: "Language as a Window Into the Mind: How NLP and LLMs Advance Human...",
        
        description: "Can NLP predict heroin-addiction outcomes, uncover suicide risk, or simulate (and even influence) brain activity? Could LLMs one day contribute to research worthy of a Nobel Prize for advancing our understanding of human behavior? And what role do NLP scientists play in shaping that possibility? This post explores these questions, arguing that language technologies are not just tools that support scientific work (like literature search agents, writing tools, or coding assistants), but that by treating language as a window into the human mind, NLP and LLMs can actively help researchers uncover mechanisms of human behavior, cognition, and brain function.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/nlp-for-human-sciences/";
          
        },
      },{id: "post-model-misspecification-in-simulation-based-inference-recent-advances-and-open-challenges",
        
          title: "Model Misspecification in Simulation-Based Inference - Recent Advances and Open Challenges",
        
        description: "Model misspecification is a critical challenge in simulation-based inference (SBI), particularly in neural SBI methods that use simulated data to train flexible neural density estimators. These methods typically assume that simulators faithfully represent the true data-generating process, an assumption that is often violated in practice. Resulting discrepancies can make observed data effectively out-of-distribution relative to the simulations, leading to biased posterior distributions and misleading uncertainty quantification. This post reviews recent work on model misspecification in neural SBI, covering formal definitions, methods for detection and mitigation, and their underlying assumptions. It also discusses practical implications for SBI workflows and outlines open challenges for developing robust SBI methods that remain reliable in realistic, imperfectly specified applications.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/model-misspecification-in-sbi/";
          
        },
      },{id: "post-evaluating-machine-learned-inter-atomic-potentials-for-a-practical-simulation-workflow",
        
          title: "Evaluating Machine-Learned Inter-Atomic Potentials for a Practical Simulation Workflow",
        
        description: "MLIPs are a promising paradigm in atomistic simulation, potentially offering the accuracy of ab-initio methods at the speed of empirical potentials. In this blog post, we give an overview of recent MLIP architectures, followed by an evaluation on a practical CO2 adsorption simulation. We find that as of today these models, though promising, are far from plug-and-play, requiring significant engineering effort to operate within established simulation frameworks, while also failing to produce physically consistent results.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/mlip-practical/";
          
        },
      },{id: "post-is-the-evidence-in-39-language-models-learn-to-mislead-humans-via-rlhf-39-valid",
        
          title: "Is the evidence in &#39;Language Models Learn to Mislead Humans via RLHF&#39; valid?...",
        
        description: "Language Models Learn to Mislead Humans via RLHF (published at ICLR 2025) argues that RLHF can unintentionally train models to mislead humans – a phenomenon termed Unintentional-SOPHISTRY. However, our review of the paper&#39;s code and experiments suggests that a significant portion of their empirical findings may be due largely to major bugs that make the RLHF setup both unrealistic and highly prone to reward hacking. In addition to high-level claims, we  correct these issues for one of their experiments, and fail to find evidence that supports the original paper&#39;s claims.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/mislead-lm/";
          
        },
      },{id: "post-misalignment-patterns-and-rl-failure-modes-in-frontier-llms",
        
          title: "Misalignment Patterns and RL Failure Modes in Frontier LLMs",
        
        description: "With the rapid ability grokking of frontier Large Models (LMs), there is growing attention and research focus on aligning them with human values and intent via large scale reinforcement learning and other techniques. However, as LMs are getting stronger and more agentic, their misalignment and deceptive behaviors are also emerging and becoming increasingly difficult for humans to pre-detect and keep track of. This blog post discusses current misalignment patterns, deceptive behaviors, RL failure modes, and emergent traits in modern large models to further AI safety discussions and advance the development of mitigation strategies for LM misbehaviors.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/misalign-failure-mode/";
          
        },
      },{id: "post-learning-to-maximize-rewards-via-reaching-goals",
        
          title: "Learning to Maximize Rewards via Reaching Goals",
        
        description: "Goal-conditioned reinforcement learning learns to reach goals instead of optimizing hand-crafted rewards. Despite its popularity, the community often categorizes goal-conditioned reinforcement learning as a special case of reinforcement learning. In this post, we aim to build a direct conversion from any reward-maximization reinforcement learning problem to a goal-conditioned reinforcement learning problem, and to draw connections with the stochastic shortest path framework. Our conversion provides a new perspective on the reinforcement learning problem: maximizing rewards is equivalent to reaching some goals.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/mdp-to-gcmdp/";
          
        },
      },{id: "post-loneliness-as-a-case-study-for-social-reward-misalignment",
        
          title: "Loneliness as a Case Study for Social Reward Misalignment",
        
        description: "The goal of this blogpost is to use loneliness as a clean case study of social proxy-reward misalignment in RL. We introduce a minimal homeostatic environment with loneliness drift and accumulated harm, and show that engagement-optimized agents learn short-term “social snack” policies that reduce the error signal without improving the underlying social state. This simple testbed highlights why reward inference or well-being objectives may be a better foundation than engagement proxies for socially aligned AI.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/loneliness-social-misalignment/";
          
        },
      },{id: "post-from-reinforce-to-dr-grpo-a-unified-perspective-on-llm-post-training",
        
          title: "From REINFORCE to Dr. GRPO: A Unified Perspective on LLM Post-Training",
        
        description: "Recently, many reinforcement learning (RL) algorithms have been applied to improve the post-training of large language models (LLMs). In this article, we aim to provide a unified perspective on the objectives of these RL algorithms, exploring how they relate to each other through the Policy Gradient Theorem — the fundamental theorem of policy gradient methods.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/llm-post-training/";
          
        },
      },{id: "post-the-human-knowledge-loophole-in-the-39-bitter-lesson-39-for-llms",
        
          title: "The human knowledge loophole in the &#39;bitter lesson&#39; for LLMs",
        
        description: "Are LLMs a proof that the &#39;bitter lesson&#39; holds for NLP? Perhaps the opposite is true: they work due to the scale of human data, and not just computation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/llm-bitter-lesson/";
          
        },
      },{id: "post-the-layered-ontology-of-models-resolving-the-epistemological-crisis-of-ai",
        
          title: "The Layered Ontology of Models, Resolving the Epistemological Crisis of AI",
        
        description: "I propose a five-layer model framework and discuss the concepts of Meaning and Truth in the era of large models through two thought experiments.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/layered-ontology-model/";
          
        },
      },{id: "post-justrl-scaling-a-1-5b-llm-with-a-simple-rl-recipe",
        
          title: "JustRL: Scaling a 1.5B LLM with a Simple RL Recipe",
        
        description: "Training small reasoning models with RL has become a race toward complexity, using multi-stage pipelines, dynamic schedules, and curriculum learning. We ask whether this complexity necessary? We show that JustRL, a simple recipe with fixed hyperparameters, achieves state-of-the-art performance on two different 1.5B base models (54.5% and 64.3% across 9 math benchmarks) while using 2× less compute than sophisticated approaches. The same hyperparameters transfer across both models without tuning, and training remains stable over thousands of steps without intervention. This suggests the field may be adding complexity to solve problems that disappear with a stable, scaled-up baseline.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/justrl/";
          
        },
      },{id: "post-how-to-open-the-black-box-amp-58-modern-models-for-mechanistic-interpretability",
        
          title: "How To Open the Black Box&amp;#58 Modern Models for Mechanistic Interpretability",
        
        description: "Understanding how transformers represent and transform internal features is a core challenge in mechanistic interpretability. Traditional tools like attention maps and probing reveal only partial structure, often blurred by polysemanticity and superposition. More principled alternatives work by recovering interpretable structure directly from activations&amp;#58 Sparse Autoencoders extract sparse, disentangled features from the residual stream; Semi-Nonnegative Matrix Factorization decomposes MLP activations into neuron-grounded building blocks; Cross-Layer Transcoders trace how these features propagate and transform across depth. Together, they form a coherent, feature-centric framework for understanding what transformers learn and how they compute.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/interpret-model/";
          
        },
      },{id: "post-heuristic-based-ideation-for-guiding-llms-toward-structured-creativity",
        
          title: "Heuristic-Based Ideation for Guiding LLMs Toward Structured Creativity",
        
        description: "Large Language Models (LLMs) hold immense promise for accelerating scientific discovery, yet current LLM-based ideation methods often rely on ad-hoc strategies rather than systematic frameworks. This blog introduces Ideation Heuristics, a systematic approach that formalizes 20 heuristics that structure how researchers generate new ideas. We show that researchers across disciplines find these heuristics highly useful, and we demonstrate how they can be operationalized through skills.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/ideation-heuristics/";
          
        },
      },{id: "post-in-context-learning-of-representations-can-be-explained-by-induction-circuits",
        
          title: "In-context learning of representations can be explained by induction circuits",
        
        description: "Park et al., 2025 find that when large language models process random walks on a graph in-context, the geometry of their token representations comes to mirror the graph&#39;s connectivity structure. This suggests a two-step mechanism: the model first reshapes its representations to reflect the graph structure, then operates over them to predict valid next steps. We offer a simpler mechanistic explanation. The in-context graph tracing task can be solved by induction circuits, which implement a simple rule: if token B followed token A earlier in the sequence, predict B the next time A appears. Since consecutive tokens in a random walk are always graph neighbors, this rule naturally produces valid steps. Ablating the attention heads that comprise induction circuits substantially degrades task performance, compared to ablating random heads. As for the geometric structure of representations, we propose that induction circuits can account for this too: their previous-token heads mix each token&#39;s representation with that of its predecessor, always a graph neighbor in a random walk. We show that a single round of such &#39;neighbor mixing&#39; applied to random embeddings is sufficient to recreate the graph-like PCA structure. These results suggest that the observed representation geometry is a byproduct of how the model solves the task, not the means by which it does so.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/iclr-induction/";
          
        },
      },{id: "post-diffusion-as-infinite-hierarchical-vaes-do-diffusion-models-generalize-better-than-deep-vaes",
        
          title: "Diffusion as Infinite Hierarchical VAEs - Do Diffusion Models Generalize Better than Deep...",
        
        description: "This blogpost unifies Diffusion Models and Variational Autoencoders. We demonstrate that DPMs are mathematically equivalent to Hierarchical VAEs (HVAEs) in the limit of infinite depth. By analyzing this architectural link, we explain why diffusion models avoid the posterior collapse that plagues deep VAEs and identify the sweet spot for generalization where these models perform best.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/generalization-in-diffusion-as-infinite-hvae/";
          
        },
      },{id: "post-ready-for-general-agents-let-39-s-test-it",
        
          title: "Ready For General Agents? Let&#39;s Test It.",
        
        description: "General-purpose agents are emerging, promising seamless deployment across domains. However, we currently do not measure their adaptability to diverse, unseen settings—a core requirement for true generality. We outline the key challenges and chart a path toward a unified evaluation framework designed to guide the development of general agents.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/general-agent-evaluation/";
          
        },
      },{id: "post-generative-ai-archaeology",
        
          title: "Generative AI Archaeology",
        
        description: "We document the rise of the Generative AI Archaeologist, whose tools include linear algebra and probability theory, jailbreaking, and debuggers, compared to the metal detectors, pickaxes, and radar surveys of traditional archaeology. GenAI Archaeologists have reported findings both through luck by observing unexpected behaviour in publicly accessible models, and by exploiting the mathematical properties of models. In this blog, we survey four types of discoveries unearthed by GenAI Archaeologists and discuss the status of those findings.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/genai-archaeology/";
          
        },
      },{id: "post-divide-conquer-and-standardize-a-recursive-architecture-for-multi-agent-systems-mas",
        
          title: "Divide, Conquer, and Standardize - A Recursive Architecture for Multi-Agent Systems (MAS)",
        
        description: "The scalability and robustness of current Multi-Agent Systems (MAS) are severely constrained by the heterogeneity of communication interfaces and a reliance on fragile ad-hoc integrations. We introduce FRACTAL-MAS, a recursive architecture that standardizes orchestration through the convergence of MCP and A2A protocols, integrating a unified control loop with procedural memory grounded in Case-Based Reasoning (CBR). This design allows for continuous adaptation without fine-tuning and enables a seamless transition from rigid hierarchical structures to decentralized networks, providing a reference architecture for the robust and scalable construction of MAS.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/fractal-mas/";
          
        },
      },{id: "post-flow-where-you-want",
        
          title: "Flow Where You Want",
        
        description: "This tutorial demonstrates how to add inference-time controls to pretrained flow-based generative models operating in latent space. Using an unconditional MNIST flow model, we apply classifier guidance and inpainting by adding velocity corrections during sampling. We also explore PnP-Flow, which satisfies constraints through iterative projection rather than velocity correction.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/flow-where-you-want/";
          
        },
      },{id: "post-from-trajectories-to-operators-a-unified-flow-map-perspective-on-generative-modeling",
        
          title: "From Trajectories to Operators — A Unified Flow Map Perspective on Generative Modeling...",
        
        description: "In this post, we reframe continuous-time generative modeling from integrating trajectories to learning two-time operators (flow maps). This operator view unifies diffusion, flow matching, and consistency models, and suggests a practical diagnostic — semigroup-consistent jumps yield both step-robust generation and low compositional drift. We derive Eulerian/Lagrangian distillation objectives and use inpainting experiments to show why semigroup-consistent jumps can be both step-robust and composition-stable.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/flow-map-learning/";
          
        },
      },{id: "post-understanding-and-fixing-bottlenecks-in-state-space-models-what-recency-and-over-smoothing-tell-us",
        
          title: "Understanding and Fixing Bottlenecks in State Space Models: What Recency and Over-Smoothing Tell...",
        
        description: "This work analyzes how recency bias and hidden-state over-smoothing emerge in modern State Space Models, revealing the bottlenecks that limit their ability to capture long-range dependencies.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/fixing-bottlenecks-in-state-space-models/";
          
        },
      },{id: "post-the-effect-of-feature-resolution-on-embedding-dimension",
        
          title: "The effect of feature resolution on embedding dimension",
        
        description: "High-dimensional data can be compressed into lower-dimensional embeddings while retaining a relatively large amount of relevant information, a phenomenon which, despite its widespread use, we struggle to fully explain. In this post, we use a common property of datasets - a limit on the number of features per data point - to show how a slight uniform dependence between features can be exploited to reduce the required dimensions by at least a third, while sacrificing no information about the features. To do so, we introduce the concepts of dataset resolution and feature composition of a dataset, and analyse how a set of orderings of the dataset affects the types of partitions we can create of the dataset.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/feature-reduction/";
          
        },
      },{id: "post-ai-fundamentals-valuing-ai-agents-amp-data-assets",
        
          title: "AI Fundamentals: Valuing AI Agents &amp; Data Assets",
        
        description: "Large Language Model (LLM) agents now read the world through managed-context pipelines, write to it via tool-calling APIs, and continuously re-wire themselves with fresh experience. Stakeholders therefore need a Generally Accepted Accounting Principles (GAAP) compatible method to price both (i) the agent&#39;s labour-like output and (ii) the data traces that fuel learning. We formalise a single unifying metric - agent Economic Value (AEV)- and demonstrate that these metrics are measurable today. We then extend the template to reinforcement-learning regimes in which grounded rewards equal cash flows. Lastly, we propose a financial settlement layer, which transforms the agent from a passive software user into an active economic participant.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/economic-agents/";
          
        },
      },{id: "post-dllm-rethinking-generation-beyond-autoregressive-models",
        
          title: "dLLM - Rethinking Generation Beyond Autoregressive Models",
        
        description: "Diffusion large language models (dLLMs) provide an alternative to autoregressive Transformers, supporting parallel token generation and flexible infilling. They excel in structured, long-horizon, or data-constrained settings, though challenges remain with output length, denoising, and blockwise generation. Hybrid approaches combining diffusion for reasoning and autoregressive for generation show promise.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/dllm/";
          
        },
      },{id: "post-sample-blog-post",
        
          title: "Sample Blog Post",
        
        description: "Your blog post&#39;s abstract. Please add your abstract or summary here and not in the main body of your text. Do not include math/latex or hyperlinks.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/distill-example/";
          
        },
      },{id: "post-dissecting-non-determinism-in-large-language-models",
        
          title: "Dissecting Non-Determinism in Large Language Models",
        
        description: "The Large Language Models (LLMs) evolve into the backbone of complex decision-making systems, their inherent non-deterministic nature poses a significant threat to the validity of experimental results. This blog explores the impact of stochasticity, prompt brittleness, and LLM-as-a-Judge during both response generation and evaluation. We conclude that understanding these dynamics is essential to prevent misleading conclusions, advocating for consistency oriented practices that treat non-determinism as a critical variable in rigorous experimentation.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/dissecting-non-determinism/";
          
        },
      },{id: "post-discretisation-invariance",
        
          title: "Discretisation invariance",
        
        description: "Discretisation invariance, a recent innovation in scientific machine learning, is a requirement that ensures an architecture can consistently process inputs of different resolutions. In this post, we formally define this property, provide examples, generate datasets, train architectures, and discuss whether discretisation invariance delivers its intended benefits in practice.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/discretisation-invariance/";
          
        },
      },{id: "post-navigating-the-manifold-a-geometric-perspective-on-diffusion-based-inverse-problems",
        
          title: "Navigating the Manifold — A Geometric Perspective on Diffusion-Based Inverse Problems",
        
        description: "This blogpost develops a geometric and probabilistic lens on diffusion priors for inverse problems. We show that a wide range of methods mostly instantiate two operator-splitting paradigms, i.e., posterior-guided sampling and clean-space local-MAP optimization. Through manifold diagrams, Tweedie-based animations, and step-by-step derivations, we explain how these paradigms decouple a pretrained diffusion prior from measurement physics, clarify when they approximate full posterior sampling versus MAP estimation, and distill practical design rules for building robust diffusion-based inverse solvers.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/diffusion-inverse-problems/";
          
        },
      },{id: "post-from-u-nets-to-dits-the-architectural-evolution-of-text-to-image-diffusion-models-2021-2025",
        
          title: "From U-Nets to DiTs: The Architectural Evolution of Text-to-Image Diffusion Models (2021–2025)",
        
        description: "A comprehensive analysis of how diffusion model architectures evolved from U-Net backbones to Diffusion Transformers, transforming text-to-image generation capabilities.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/diffusion-architecture-evolution/";
          
        },
      },{id: "post-destruction-is-a-general-strategy-to-learn-generation-diffusion-39-s-strength-is-to-take-it-seriously-exploration-is-the-future",
        
          title: "Destruction is a General Strategy to Learn Generation; Diffusion&#39;s Strength is to Take...",
        
        description: "I present diffusion models as part of a family of machine learning techniques that withhold information from a model’s input and train it to guess the withheld information. I argue that diffusion&#39;s destroying approach to withholding is more flexible than typical hand-crafted information withholding techniques, providing a rich training playground that could be advantageous in some settings, notably data-scarce ones. I then address subtle issues that may arise when porting reinforcement learning techniques to the diffusion context, and wonder how such exploration problems could be addressed in more diffusion-native ways. I do not have definitive answers, but I do point my fingers in directions I deem interesting. A tutorial follows this thesis, expanding on the destroy-then-generate perspective. A novel kind of probabilistic graphical models is introduced to facilitate the tutorial&#39;s exposition.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/destruction/";
          
        },
      },{id: "post-content-promotion-as-a-strategic-game-how-to-design-agentic-publishers-for-the-evolving-search-ecosystem-in-the-genai-era",
        
          title: "Content Promotion as a Strategic Game: How to Design Agentic Publishers for the...",
        
        description: "With the rise of LLMs, publishers now operate in a dual world where traditional search and chat-like systems coexist. We propose a unified, game-theoretic view of this environment and highlight different tools, such as Multi-Agent Reinforcement Learning, that support the development of competitive content-optimization agents.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/content-promotion-agent-design/";
          
        },
      },{id: "post-defining-and-quantifying-compositional-structure",
        
          title: "Defining and quantifying compositional structure",
        
        description: "Compositionality is thought to be crucial in human cognition and AI, but we lack a scientific understanding of what it is. What kind of data is compositionally structured? Can we mathematically quantify the amount and character of compositional structure? This blog post introduces a novel approach for doing so, building off of existing tools from algorithmic information theory that formalize notions of complexity and structure. The mathematical definition of compositionality that we&#39;ll come to is rigorous, precise, and general, and the hope is that it can inspire novel research directions in AI for uncovering compositional structure in natural data.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/compositionality/";
          
        },
      },{id: "post-chunktabpfn-training-free-long-context",
        
          title: "ChunkTabPFN: Training-free Long Context",
        
        description: "Tabular foundation models such as TabPFN are limited in practice by the memory cost of attention, which grows quadratically with the number of samples and features. While efficient attention backends alleviate this in principle, CUDA grid limits and hardware compatibility gaps prevent their direct application at the scale of real-world tabular datasets. We introduce Chunked TabPFN, an exact tiling strategy that removes these implementation bottlenecks without retraining or approximation, extending TabPFN to 100K+ rows on a single GPU. On the long-context slice of TabArena, we find that — contrary to earlier reports — TabPFN&#39;s performance continues to improve with larger contexts, suggesting that the prior bottleneck was implementation-level memory, not model-level capacity.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/chunked-tabpfn/";
          
        },
      },{id: "post-budget-alignment-making-models-reason-in-the-user-39-s-language",
        
          title: "Budget Alignment: Making Models Reason in the User&#39;s Language",
        
        description: "We explore a two step multilingual alignment recipe for large language models to keep reasoning and answers in the user language while preserving accuracy.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/budget-alignment/";
          
        },
      },{id: "post-the-99-success-paradox-when-near-perfect-retrieval-equals-random-selection",
        
          title: "The 99% Success Paradox: When Near-Perfect Retrieval Equals Random Selection",
        
        description: "For most of the history of information retrieval (IR), search results were designed for human consumers who could scan, filter, and discard irrelevant information on their own. This shaped retrieval systems to optimize for finding and ranking more relevant documents, but not keeping results clean and minimal, as the human was the final filter. However, LLMs have changed that by lacking this filtering ability. To address this, we introduce Bits-over-Random (BoR), a chance-corrected measure of retrieval selectivity that reveals when high success rates mask random-level performance.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/bits-over-random/";
          
        },
      },{id: "post-square-peg-round-hole-plugging-non-sequential-data-into-sequential-language-models",
        
          title: "Square Peg, Round Hole: Plugging Non-Sequential Data into Sequential Language Models",
        
        description: "Autoregressive (AR) models are central to modern generative AI systems, yet their need for an ordered sequence of tokens clashes with modalities that lack an obvious ordering, such as images, graphs, and point clouds. Despite this mismatch, AR models are widely applied beyond language owing to their scalability and controllability. In this post, we articulate exactly what the problem is, and how it can be solved. In short, there are two broad classes of techniques for applying AR models to non-sequential data: selecting a generation order given some fixed tokenization scheme, and redesigning the tokenization itself to simplify next-token prediction. Yet these methods face tradeoffs, particularly between compression (how many bits are used to represent the input) and autoregressive &amp;quot;modelability&amp;quot; (how easy it is to model each next-token conditional distribution in the chosen order). We predict that as data-hungry AI pipelines require new data modalities to train integrated, multi-modal models, these considerations will only grow more crucial. By drawing these connections, we aim to motivate future work on tokenizations tailored to the needs of autoregressive models for arbitrary datatypes.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/autoregressive-tokenization/";
          
        },
      },{id: "post-an-overview-of-subliminal-learning",
        
          title: "An Overview of Subliminal Learning",
        
        description: "In this blog post we survey the current state of subliminal learning research. We conclude by discussing the gaps in the literature which would take these techniques from research interests to potential real world concerns.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/an-overview-of-subliminal-learning/";
          
        },
      },{id: "post-the-adversarial-conditioning-paradox-how-fine-tuning-creates-a-geometric-signature-that-attacks-unknowingly-exploit",
        
          title: "The Adversarial Conditioning Paradox: How Fine-Tuning Creates a Geometric Signature That Attacks Unknowingly...",
        
        description: "Adversarial attacks optimized purely against the softmax output reliably land in geometrically ill-conditioned regions at Layer 12 of fine-tuned BERT — a signature that does not exist before task training. We validate this finding across three attack families (TextFooler, PWWS, DeepWordBug) at N=1,000 each.",
        section: "Posts",
        handler: () => {
          
            window.location.href = "/2026/blog/2026/adversarial-conditioning-paradox/";
          
        },
      },{id: "books-the-godfather",
          title: 'The Godfather',
          description: "",
          section: "Books",handler: () => {
              window.location.href = "/2026/books/the_godfather/";
            },},{id: "news-a-simple-inline-announcement",
          title: 'A simple inline announcement.',
          description: "",
          section: "News",},{id: "news-a-long-announcement-with-details",
          title: 'A long announcement with details',
          description: "",
          section: "News",handler: () => {
              window.location.href = "/2026/news/announcement_2/";
            },},{id: "news-a-simple-inline-announcement-with-markdown-emoji-sparkles-smile",
          title: 'A simple inline announcement with Markdown emoji! :sparkles: :smile:',
          description: "",
          section: "News",},{id: "projects-project-1",
          title: 'project 1',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/1_project/";
            },},{id: "projects-project-2",
          title: 'project 2',
          description: "a project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/2_project/";
            },},{id: "projects-project-3-with-very-long-name",
          title: 'project 3 with very long name',
          description: "a project that redirects to another website",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/3_project/";
            },},{id: "projects-project-4",
          title: 'project 4',
          description: "another without an image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/4_project/";
            },},{id: "projects-project-5",
          title: 'project 5',
          description: "a project with a background image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/5_project/";
            },},{id: "projects-project-6",
          title: 'project 6',
          description: "a project with no image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/6_project/";
            },},{id: "projects-project-7",
          title: 'project 7',
          description: "with background image",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/7_project/";
            },},{id: "projects-project-8",
          title: 'project 8',
          description: "an other project with a background image and giscus comments",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/8_project/";
            },},{id: "projects-project-9",
          title: 'project 9',
          description: "another project with an image 🎉",
          section: "Projects",handler: () => {
              window.location.href = "/2026/projects/9_project/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%79%6F%75@%65%78%61%6D%70%6C%65.%63%6F%6D", "_blank");
        },
      },{
        id: 'social-inspire',
        title: 'Inspire HEP',
        section: 'Socials',
        handler: () => {
          window.open("https://inspirehep.net/authors/1010907", "_blank");
        },
      },{
        id: 'social-rss',
        title: 'RSS Feed',
        section: 'Socials',
        handler: () => {
          window.open("/2026/feed.xml", "_blank");
        },
      },{
        id: 'social-scholar',
        title: 'Google Scholar',
        section: 'Socials',
        handler: () => {
          window.open("https://scholar.google.com/citations?user=qc6CJjYAAAAJ", "_blank");
        },
      },{
        id: 'social-custom_social',
        title: 'Custom_social',
        section: 'Socials',
        handler: () => {
          window.open("https://www.alberteinstein.com/", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
