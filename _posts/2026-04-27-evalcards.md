---
layout: distill
title: EvalCards for Standardized Evaluation Reporting
description: In the age of rapidly released LLMs, evaluation reporting is fragmented, inconsistent, and often misleading. We surveyed the landscape and found three critical crises—reproducibility, accessibility, and governance—that Model Cards alone can't solve. Our solution? EvalCards-- lightweight, standardized evaluation summaries that are easy to write, easy to understand, and impossible to miss. EvalCards are designed to enhance transparency for both researchers and practitioners while providing a practical foundation to meet emerging governance requirements.
date: 2026-04-27
future: true
htmlwidgets: true
#hidden: true

# Mermaid diagrams
# mermaid:
#   enabled: true
#   zoomable: true

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
bibliography: 2026-04-27-evalcards.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Introduction
  - name: Problems with Evaluation Reporting
    subsections:
      - name: Reproducibility Crisis
      - name: Accessibility Crisis
      - name: Governance Crisis
  - name: Problems with Existing Standards
  - name: EvalCards
    subsections:
      - name: Design Principles of EvalCards
      - name: What should an EvalCard contain?
      - name: When should EvalCards be created?
      - name: Where should EvalCards be displayed?
      - name: Model Card vs EvalCard
  - name: EvalCards Case Studies
  - name: Alternative Views
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

Classic scientific scandals often turn on withholding of details around exactly how scientific hypotheses were evaluated. For example, the case of the Piltdown Man, where researchers selectively (mis-)reported crucial contextual details, misled evolutionary sciences for decades <d-cite key="vincent1999piltdown"></d-cite>. Lack of reporting standards can also lead to confusion, even when everyone acts in good faith. To illustrate, in 19th-century chemistry, the lack of agreed conventions on atomic weights left the field in chaos, with the same compounds appearing under conflicting formulas, until the Karlsruhe Congress established common standards <d-cite key="ihde1961karlsruhe"></d-cite>. These (and other similar) episodes <d-cite key="goldacre2009bad"></d-cite> instill the same lesson: without reliable reporting conventions, even important discoveries can distort rather than advance science.

Evaluation - quantitative measurement of a model's performance on a pre-defined task or benchmark - has long been one of the central means of assessing progress in NLP <d-cite key="jones1994towards, church2017emerging, church2019survey, bowman-dahl-2021-will, kiela-etal-2021-dynabench, sainz-etal-2023-nlp"></d-cite>. Despite this, our standards for reporting evaluations have not kept pace <d-cite key="bhatt2021case, belz-etal-2023-non, belz-etal-2025-standard, zhao-etal-2025-sphere"></d-cite>. Such a lack of standards becomes more concerning with the fast adoption of Large Language Models (LLMs) by a wide range of stakeholders, many of whom are not experts and yet heavily depend on such systems to make decisions that impact real-world outcomes <d-cite key="araujo2020ai, bommasani2023holistic"></d-cite>.  As LLMs become embedded in critical domains, responsible deployment is a key consideration <d-cite key="10536000, radanliev2024ethics, orr2024building,tripathi2025ethical"></d-cite> and a major part of this includes transparency on what a model can and cannot do. Based on a survey of recent research in the field of evaluation studies, we identify three critical problems stemming from current reporting practices:  the *Reproducibility Crisis*, the *Accessibility Crisis*, and the *Governance Crisis*. We discuss why current efforts at transparency <d-cite key="mitchell2019model, gebru2021datasheets"></d-cite> need reconsideration. In light of such issues, we propose EvalCards: concise evaluation summaries which are (i) *easy to write*, (ii) *easy to understand*, and (iii) *hard to miss*. We present case studies of three popular models, showing how difficult it was to gather consistent evaluation details when creating sample EvalCards, and also discuss directions of future work.

Our main argument is one for a shift in norms: evaluation reporting is not a marketing exercise but a core component of what it means to release a model responsibly.  While the broader challenge of how to evaluate models remains open and complex <d-cite key="laskar2024systematic, chang2024survey, gao2025llm"></d-cite>, our focus here is narrower but nonetheless critical: improving how evaluations are reported. We hope this work sparks conversation and helps move the field toward a culture of more honest and actionable evaluation disclosure practices.

## Problems with Evaluation Reporting

{% include figure.liquid path="assets/img/2026-04-27-evalcards/eval_card_figure.jpg" class="img-fluid" %}

To ground our analysis of problems in evaluation reporting in NLP and AI, we conducted a survey of recent work with systematic keyword searches related to evaluation and reporting (e.g., evaluation, reporting, disclosure, evaluation artifacts) in ACL Anthology, DBLP, and Google Scholar. We complemented this with a reverse snowball sampling from the most recent broad-scope seminal works in NLP evaluation as seeds <d-cite key="weidinger2025toward, gao2025llm,zhao-etal-2025-sphere,chang2024survey,laskar2024systematic,biderman2024lessons,burnell2023rethink, allen2021evaluation"></d-cite>. We then manually analyzed the works to extract recurring themes of discussion. From this, we identify three overarching crises of reproducibility, accessibility, and governance. 

### Reproducibility Crisis

In machine learning, the broader reproducibility crisis is well discussed <d-cite key="kapoor2023leakage"></d-cite> and also manifests in evaluation reporting <d-cite key="bouthillier2019unreproducible, dodge-etal-2019-show,  belz-etal-2025-standard, zhao-etal-2025-sphere"></d-cite>. More specifically, in the context of model evaluations, prior work repeatedly highlights that published results often cannot be trusted without careful reconstruction of undocumented choices <d-cite key="belz-etal-2021-systematic,burnell2023rethink, belz-etal-2023-non, biderman2024lessons, belz-etal-2025-standard, zhao-etal-2025-sphere"></d-cite>. We conduct some case studies on some recent LLM model releases and discuss three such crucial details that are often inconsistent or missing from evaluation details:

**Target Capability**  Model releases include benchmark names and scores,  but can often fail to specify what each benchmark is intended to measure. This problem can be exacerbated for reported scores on large composite benchmarks such as SuperGLUE <d-cite key="wang2019superglue"></d-cite> or HELM <d-cite key="bommasani2023holistic"></d-cite> that aggregate tasks across domains and do not provide clarity, especially for non-experts, as to what capability is targeted.

**Metric** Reported scores frequently omit which evaluation metric was used, which makes it difficult to assess what a score reflects or to compare performance across models fairly <d-cite key="mizrahi2024state, hu2024unveiling"></d-cite>.

**Prompting Strategy** Prompting, i.e, the way a query is structured and phrased for LLMs, is one of the most significant variables in model performance <d-cite key="hu2024unveiling, sclar2024quantifying, zhuo-etal-2024-prosa, chatterjee-etal-2024-posix"></d-cite>, yet is often absent from reported results. Also, the absence of consistent reporting on a common baseline strategy, such as zero-shot prompting, further hinders meaningful comparison across models <d-cite key="10.1145/3582269.3615599, ousidhoum-etal-2021-probing"></d-cite>.


### Accessibility Crisis

A recurring theme in relevant work is the highly fragmented nature of available information on models. Documentation frameworks such as Model Cards, DataSheets, and FactSheets were introduced precisely to improve accessibility of information about models <d-cite key="mitchell2019model, gebru2021datasheets, arnold2019factsheets, bhardwaj2024machine, luo2025lack"></d-cite>. However, even today, evaluation details are dispersed across academic papers, technical appendices, GitHub READMEs, HuggingFace model cards, and blog posts, each using different terminology and presentation styles. This scattered documentation makes it difficult to locate and compare the evaluation results. This has consequences for both researchers and users.

**For Researchers** When evaluation details are scattered across sources, they are easily overlooked or lost altogether. Important context, such as benchmark versions, question framing, or metric details, may never reach the researchers who rely on these results <d-cite key="belz-etal-2023-non,belz-thomson-2024-2024,belz-etal-2025-standard, belz-etal-2025-2025"></d-cite>.

**For Users** For non-technical users and decision-makers, the problem is compounded by selective reporting, where only strong benchmark results are emphasized or marketed in some sources, further reducing trust in evaluation claims <d-cite key="arnold2019factsheets"></d-cite>. As a result, users may struggle to select appropriate models, a challenge that becomes particularly critical in high-stakes or sensitive deployment contexts <d-cite key="huijgens2024help, 10.1145/3706598.3713240"></d-cite>.

For evaluation to be actionable, it must be consistently visible and easily accessible. Without a standardized way to report evaluations in one place, users are left to piece together incomplete information, undermining efforts to assess a model’s suitability for deployment.

### Governance Crisis

AI legislations across the world today, from US to EU <d-cite key="edwards2021eu, act2024eu, sloane2025systematic, carey2025regulating"></d-cite> and from Singapore to China <d-cite key="pande2023navigating, roberts2021chinese, dong2024meta"></d-cite>, are increasingly concerned with transparency <d-cite key="larsson2020transparency, agrawal2024accountability"></d-cite> and reporting mandates <d-cite key="nagendran2020artificial, laux2024three"></d-cite>. Without standardized evaluation reports, governance of models faces three key problems: 

**Risk Assessment** It becomes challenging to determine model risks when the evaluation methods are not clearly reported <d-cite key="hogan2021ethics, novelli2024taking, reuel2024betterbench, reuel2024open"></d-cite>. 

**Algorithmic Accountability** When developers can selectively report results or omit critical weaknesses, it makes it difficult for external reviewers or regulators to hold systems to consistent standards <d-cite key="shah2018algorithmic, wieringa2020account, horneber2023algorithmic"></d-cite>.

**Compliance Washing** Akin to ethics washing practices <d-cite key="bietti2020ethics"></d-cite>, AI developers can satisfy regulatory requirements by disclosing something---even if that ``something" is incomplete, selectively positive, or methodologically weak <d-cite key="koshiyama2024towards"></d-cite>. Regulatory compliance becomes a box-ticking exercise, undermining the goals of safety, accountability, and public trust <d-cite key="veale2021demystifying"></d-cite>.

These crises highlight that the problem is not only how models are evaluated, but how those evaluations are reported. In the next section, we examine limitations of current standardization efforts and introduce EvalCards as a practical solution for improving evaluation reporting.

## Problems with Existing Standards

**Lack of evaluation focus** Existing documentation frameworks rarely place evaluation at the center. Model Cards, DataSheets, and FactSheets typically treat evaluation results as only one component among many. BenchmarkCards <d-cite key="sokol2024benchmarkcards"></d-cite> focuses on information specific to a single benchmark only, with no reference to models. For large language models, however, evaluation is the primary means by which users, researchers, and regulators can understand capabilities and limitations.

**High effort for developers** Most of the proposed documents, like Model Cards <d-cite key="mitchell2019model"></d-cite>, are lengthy and time-consuming to produce since they require additional analysis like listing out all possible use-cases and users, detailed demographic factor evaluation, intersectional quantitative analyses, etc. This can be especially problematic when many models are being released at a rapid pace. Dedicating extra time to such detailed analysis may not be possible. 

**Limited accessibility for non-experts** For decision-makers, policymakers, and many end-users, these documents are often too technical or jargon-heavy to offer real clarity <d-cite key="mcgregor2025err, crisan2022interactive"></d-cite>. For example, OpenAI's system cards, like the [GPT-4o card](https://openai.com/index/gpt-4o-system-card/), offer in-depth safety and governance information but are often very lengthy and complex. As a result, even when provided, they are underutilized by key stakeholders <d-cite key="blodgett-etal-2024-human"></d-cite>.

**Lack of visibility** While earlier works have talked about model documentation, not many have emphasized the need for visibility. Today, information about models is frequently buried in supplemental materials, obscure repositories, or separate websites---making it difficult to access and easy to overlook <d-cite key="mcgregor2025err"></d-cite>.

## EvalCards

To address the challenges outlined in previous sections, we propose Evaluation Disclosure Cards (EvalCards), a short-form standardized reporting format for model evaluations. In this section, we discuss the design principles of an EvalCard, what it should contain, when it should be created, and where it should be available. 

### Design Principles of EvalCards

Any reporting format must go beyond existing documentation efforts by tackling the practical barriers identified above: focusing on evaluation, reducing the burden on developers, making results clear to a wide range of users, and ensuring that evaluation information is consistently visible wherever models are accessed. We summarize the design principles here:

**Evaluation Focus** Unlike broader documentation frameworks such as Model Cards or DataSheets, which include information about training data, intended use cases, and ethical considerations, EvalCards place evaluation at the center. The goal is not to capture every possible aspect of model development, but to provide clear and standardized details about what was evaluated, how it was evaluated, and under what conditions.

**Easy to Write** For transparency to become standard practice, evaluation reporting must be easy to implement. EvalCards are designed to capture only the essential details of model evaluation, making them quick to produce and maintain. This is especially important for smaller organizations and open-source model developers that lack the resources to run large test suites.

**Easy to Understand** Transparency is meaningless if only a handful of experts can interpret it. Evaluation reports must be designed for broad accessibility, enabling not just researchers but also all those without domain expertise to grasp a model's capabilities and risks quickly.

**Hard to Miss**  As discussed, evaluation details are buried in academic papers, supplementary materials, or hidden deep within repositories. Standardized evaluation reports should be integrated directly into any landing pages where models can be accessed, whether that is on HuggingFace Hub, API dashboards, or third-party model provider repositories. By ensuring that evaluation disclosures are always visible and linked to the model itself, we would create a culture where understanding a model's capabilities and risks becomes a default part of using models, not an optional deep dive.

### What should an EvalCard contain?

**Modalities Evaluated** EvalCards specify which input and output modalities---such as text, image, or audio---the model has been evaluated on.

**Languages Evaluated** As with modalities, clearly stating which languages a model has evaluated on helps define the scope of its real-world applicability. Many models advertise multilingual ``training'' or ``support'', but such claims do not indicate whether those languages have been explicitly tested in any systematic way. Without evaluation, such claims can be misleading <d-cite key="joshi-etal-2020-state, blasi-etal-2022-systematic,talat-etal-2022-reap"></d-cite>

**Capability Evaluation** EvalCards do not prescribe specific benchmarks, but we require developers to explicitly state which core abilities (e.g., summarization, reasoning, factual recall, mathematical problem solving) were evaluated, and to indicate the benchmark chosen for each. Additionally, all reported results should be accompanied by the metric used (e.g., exact match, accuracy, precision@1), zero-shot prompting strategy (to enable better baseline comparison across models), and any alternative prompting strategies tested (e.g., few-shot, chain-of-thought).

**Safety Evaluation** AI models pose well-documented risks, such as bias <d-cite key="dai2024bias"></d-cite>, toxicity <d-cite key="luong-etal-2024-realistic"></d-cite>, and misinformation generation <d-cite key="zhang2024toward, chen2024combating"></d-cite>. These issues are often under-reported or selectively presented in current evaluation practices <d-cite key="burnell2023rethink, mcgregor2025err"></d-cite>. EvalCards should include a dedicated section for such safety risks. As with capability evaluations, developers should specify the safety feature evaluated, the benchmarks used, the metrics applied, and both the zero-shot and any alternative prompting scores.

**Developer Footnotes** In this free-text section of the EvalCard, model developers can choose to have relevant footnotes or include any information they think is relevant for users. 

### When should EvalCards be created?

EvalCards should be generated as part of the initial model release workflow, whether for open-source models or commercial APIs. *First,* model developers are the ones who trained the model, chose the data, designed the architecture, and tuned the objectives. Standardized evaluation reporting works best when it is done by the people who know the model inside out and at the point of release. Also, most of these evaluations align with internal testing already conducted by developers during model validation phases, and direct reporting can prevent multiple runs of the model on the same tests, leading to reduced climate impact. *Second,* most models that require evaluation today---especially large foundation models with hundreds of billions of parameters---are built by organizations with substantial compute resources. If a team can train a model with billions of parameters, it is likely to have enough compute to run a standard suite of evaluations. 

By embedding EvalCard creation into the release pipeline, developers ensure that transparency is delivered upfront, not left to third-party auditors. Furthermore, EvalCards should be updated with each major version change or significant fine-tuning event, reflecting how model behavior may evolve. This keeps users informed of both improvements and potential regressions across the capability and safety dimensions.

### Where should EvalCards be displayed?

Visibility is a core principle of EvalCards: Evaluation summaries should appear where the model appears. This includes:

- **Model Repositories**: EvalCards should be a standard, prominently linked file in Huggingface Hub or Github---similar to a README or license---ensuring that anyone downloading or browsing the model can immediately access it.
- **Commercial Platforms**: For closed-source or hosted models accessed through APIs or user interfaces (e.g., ChatGPT, Google Gemini), EvalCards should be integrated into developer dashboards, product documentation, or user-facing pages. This allows users to review evaluation and safety information before deployment or interaction, supporting informed use and regulatory compliance.
- **Research Publications**: For both open and closed-source models, EvalCards should be present as a section in the main text or appendix of academic papers and any technical blogs as a concise summary of evaluation results---providing a clear alternative to selective performance highlights.

### Model Card vs EvalCard

Model Cards <d-cite key="mitchell2019model"></d-cite> are an early effort at increasing transparency. However, EvalCards are not designed to serve the same purpose and are complementary to Model Cards. Below, we highlight how EvalCards differ from Model Cards and why they are essential in the current landscape:

- **Evaluation Focus:** Model Cards were developed at a time when most models were custom-built, and detailed information about training, design motivations, and use cases was essential. EvalCards elevate evaluation to the primary focus and provides a decision-time snapshot to help users and regulators quickly evaluate if a model is appropriate for their needs.
- **Ease of Adoption:** Model Cards focus on open-ended description of model information, including detailed analysis of ethnographic and ethical considerations. However, it can serve as a barrier to adoption by overwhelming end-users with less technical expertise. EvalCards are designed to be lightweight and easy to adopt, with structured fields that capture high-signal evaluation details with minimal overhead.
- **Visibility:** Model Cards do not specify where or how they should be displayed, and the information is often buried across multiple sources. EvalCards are displayed where the model is accessed---on model hubs, APIs, or UIs.

As the norm shifts towards the use of off-the-shelf generative AI models, evaluation becomes the key requirement for responsible model deployment. 

## EvalCards Case Studies

To implement our idea, we started with case studies of three popular LLMs. We document the issues we found for collecting details for each model's evaluation followed by the creation of an EvalCard for each model. In Fig 1, we create an EvalCard for the model *OLMO-2-1124-7B-Instruct*, from the OLMo project on transparency and open research from the Allen Institute for AI <d-cite key="olmo20252olmo2furious"></d-cite>- which despite being one of the most open models. Metrics and prompting details were dispersed across sources and inconsistent across benchmarks. Accessibility was limited by scattered results throughout the paper, hindering quick assessment. Safety reporting was aggregated into a single score, offering no dataset-level transparency and preventing meaningful analysis of specific risks.

{% include figure.liquid path="assets/img/2026-04-27-evalcards/evalcard_olmo.jpg" class="img-fluid" %}



 In Fig 2, we have an EvalCard for Qwen3-4B-Base, released on the 29th of April 2025. The model's release lacked timely and transparent evaluation details: key results were delayed, language coverage and capability claims were ambiguous, and many benchmarks lacked metrics or prompting information. Safety evaluations were entirely absent, making it difficult for users to reliably assess the model's performance or risks.  

{% include figure.liquid path="assets/img/2026-04-27-evalcards/evalcard_qwen.jpg" class="img-fluid" %}

 
 
 
 In Fig 3, we create an EvalCard for Gemini Flash 2.0 which lacks clear, comprehensive documentation: multilingual support is vaguely described without evidence of evaluated performance, benchmark scores omit metrics and prompting details, and no concrete safety evaluations or red-teaming results are disclosed. As a result, users must rely on incomplete information when assessing the model's actual capabilities and risks.

{% include figure.liquid path="assets/img/2026-04-27-evalcards/evalcard_gemini.jpg" class="img-fluid" %}

These case studies reveal a common thread: while open-source models like OLMo 2 generally provide more information than their closed counterparts, the process of compiling that information remains time-consuming and often incomplete. EvalCards offer a structured way to consolidate key details in one place, reducing the need to spend hours navigating scattered documentation and supplementary sources. More importantly, EvalCards make it immediately visible what's missing—turning the absence of evaluation details from something easy to hide into something impossible to ignore. For model providers, EvalCards serve as both a checklist and a commitment: a clear reminder of what transparency actually requires and a visible demonstration of their willingness to provide it.

## Alternative Views

While we advocate for the adoption of EvalCards, it is important to acknowledge reasonable concerns from different stakeholders. Here, we outline two commonly raised perspectives and address them.

**A Developer's View** Developers may argue that evaluation is already done internally, and publishing it publicly increases workload or reputational risk, especially when performance is uneven. However, as regulatory frameworks like the EU AI Act and US or UK guidelines begin to demand transparency in model capabilities and risks <d-cite key="act2024eu"></d-cite>, structured evaluation disclosures might become a requirement, not a preference. EvalCards offer a proactive way to meet these expectations. Even seeing what has not been evaluated is useful to avoid misinterpretations and build credibility ahead of external audits or compliance checks <d-cite key="raji2020closing, koshiyama2024towards"></d-cite>.


**A Researcher's View** Researchers may note that evaluation is context-sensitive, where evaluations vary by tasks <d-cite key="chang2024survey"></d-cite> or user groups <d-cite key="hershcovich-etal-2022-challenges"></d-cite> and that standards for ``good'' benchmarks are still evolving <d-cite key="reuel2024betterbench,liu-etal-2024-ecbd, blodgett-etal-2024-human"></d-cite>. Introducing a fixed reporting format might seem premature. But EvalCards are not rigid templates. They do not enforce benchmark choices but merely require clarity about what was tested and how. This transparency helps everyone interpret results accurately, compare across models, and build on prior evaluations rather than duplicating or misapplying them.

## Conclusion

Evaluation reporting is a critical but under-prioritized part of responsible NLP and AI. Non-standardized practices create hurdles for researchers, users, and regulators, fueling reproducibility, accessibility, and governance crises. Existing documentation efforts like Model Cards <d-cite key="mitchell2019model"></d-cite> or BenchmarkCards <d-cite key="sokol2024benchmarkcards"></d-cite> have aimed to improved transparency, but they do not put evaluation at the center. EvalCards aim to close this gap with a format that is easy to write, easy to understand, and hard to miss. By making evaluation details visible and consistent, they turn scattered disclosures into a foundation for cumulative research, informed adoption, and accountability. Looking ahead, we highlight three directions to strengthen and extend the EvalCard framework.

**Increased Transparency** EvalCards can help establish a unified pipeline that links evaluation and reporting into a single transparent process by reducing ambiguity across sources <d-cite key="biderman2024lessons"></d-cite>. In the longer term, EvalCards could link to Benchmark Cards <d-cite key="sokol2024benchmarkcards"></d-cite> for each mentioned benchmark, creating a connected reporting ecosystem where model, benchmark, and evaluation details are transparently linked to ensure that both the tests and the results behind model claims are easy to trace and verify.

**Increased Adoption** Widespread adoption will require making EvalCards easy to produce and use. Methods like automated extraction from technical reports <d-cite key="liu-etal-2024-automatic"></d-cite> or community contribution to scores <d-cite key="10855627"></d-cite> can help ease the burden on model developers, while usability testing can help refine design and language for diverse stakeholders <d-cite key="crisan2022interactive"></d-cite>. This will help ensure that EvalCards are not only technically sound but also popular.

**Regulatory Integration** EvalCards can play a key role in bridging technical evaluation and regulatory transparency. They can serve as a standardized reporting format for models within compliance processes, such as regulatory sandboxes under the EU AI Act <d-cite key="lanamaki2025expect"></d-cite> or US AI Risk Management Framework <d-cite key="ai2023artificial"></d-cite>, by offering a consistent way to report model performance and limitations. Partnerships with platforms like HuggingFace or emerging standards bodies could help maintain vetted benchmark sets that align with evolving priorities. Future work in technical governance <d-cite key="reuel2024open, reuel2024position"></d-cite> can explore how EvalCards can be incorporated into regulatory compliance workflows, e.g., by establishing minimum mandatory fields tied to emerging AI regulations.

As model development accelerates, reporting practices must evolve with equal urgency. Evaluation should drive progress, not confusion—but that is only possible when what models can and cannot do is made clear. EvalCards take a small but concrete step toward that goal, embedding transparency into the model release process itself. We hope this work sparks deeper reflection and concrete action toward standardizing how we evaluate and report on models. 

---