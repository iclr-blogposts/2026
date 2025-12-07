---
layout: distill
title: Heuristic-Based Ideation: Guiding LLMs Toward Structured Creativity
description: Large language models hold immense promise for accelerating scientific discovery, yet most LLM-based ideation methods still rely on ad-hoc strategies rather than principled frameworks. This blog introduces Ideation Heuristics, a systematic approach that formalizes 20 cognitive heuristics that structure how researchers generate new ideas. We show that researchers across disciplines find these heuristics highly useful, and we demonstrate how they can be operationalized through Claude skills.
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
bibliography: 2026-04-27-ideation-heuristics.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: Ideation Heuristics: A Framework for Systematic Ideation
    subsections:
      - name: Observation-Based Heuristics
      - name: Reinterpreting Past Research
      - name: Data-Driven Discovery
      - name: Direct Manipulation
      - name: Structured Analytical Approaches
  - name: Researchers Find the Heuristics Useful
  - name: How to Use the Heuristics

---

Large Language Models (LLMs) are rapidly changing how we think, write, and even discover. In science, their potential to accelerate research ideation and hypothesis generation is very exciting<d-cite key=zheng-etal-2025-automation,gridachagentic></d-cite>. Yet, current approaches often rely on ad-hoc strategies (e.g., combining insights from two papers<d-cite key=radensky2024scideator,shen2025understanding></d-cite>), lacking a systematic framework to guide this creative process. In this blog, we introduce Ideation Heuristics, a framework towards structured creativity.

## Ideation Heuristics: A Framework for Systematic Ideation

We formalize the cognitive heuristics underlying creative idea generation. By drawing on heuristics of human ideation, Ideation Heuristics can help LLMs explore the space of possible ideas in a more comprehensive way.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-ideation-heuristics/heuristics.jpg" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Inspired by McGuire<d-cite key=mcguire1997creative></d-cite>, a seminal work on human ideation heuristics, we propose 20 heuristics organized into five categories that capture the diverse ways researchers generate new ideas. Below, we show the detailed heuristics and examples of how they are used in research ideation and hypothesis generation, including human research and LLM-based ideation methods.

### Observation-Based Heuristics
*Drawing inspiration from real-world phenomena and anomalies.*

These heuristics guide researchers in transforming everyday occurrences and reflections into formal inquiries. One strategy is to `investigate deviations from expectations (H1)` by noticing outliers, anomalies, or surprising results and asking why they occur. This often leads to uncovering hidden mechanisms or flawed assumptions. Complementing this is the heuristic to `question the norm (H2)`, which encourages examination of why a widely accepted pattern, theory, or convention exists in the first place. Such questioning can reveal unexamined principles, implicit biases, or alternative explanations for established phenomena. Another strategy is to `juxtapose opposite problems (H3)`, where a deep understanding of an issue emerges from studying its inverse, allowing the contrary problems to suggest solutions for each other.

**In human research:**
> Non-Euclidean geometry<d-cite key=rosenfeld2012history></d-cite> exemplifies H2 by challenging the long-assumed universal truth of Euclid’s parallel postulate, revealing that questioning even the most obvious norms can open entirely new mathematical worlds.

**In LLM-based ideation:**
> HypoGeniC<d-cite key=zhou2024hypothesis></d-cite> and Iterative Hypothesis Refinement<d-cite key=qiuphenomenal></d-cite> exemplify the use of H1 in hypothesis generation by iteratively producing and refining hypotheses from cases that the current hypotheses fail to explain. In this way, anomalies are transformed from obstacles into drivers of discovery.

### Reinterpreting Past Research
*Finding new meaning or opportunities within existing findings.*

This category highlights heuristics for mining existing literature productively. One approach is to `generate theories from conflict (H4)`, which involves identifying and reconciling contradictory findings across studies to form a more comprehensive understanding. Researchers can also `create novel syntheses (H5)` by linking concepts from previously unrelated research, thereby unlocking new insights and theoretical frameworks. Another pathway is to `interpret incidental findings (H6)`, paying close attention to unexpected interactions, side effects, or secondary results that earlier studies did not prioritize. Finally, scholars can `reorganize current knowledge (H7)` by systematically restructuring the literature of a field to expose promising directions for further investigation.

**In human research:**
> A contaminated Petri dish reveals an unexpected bacterial killing zone around Penicillium mold<d-cite key=diggins1999true></d-cite>, an incidental observation that catalyzed the discovery of modern antibiotics, exemplifying H6.

**In LLM-based ideation:**
> A large proportion of existing ideation methods (e.g., SciMON<d-cite key=wang2024scimon></d-cite>, AI Researcher<d-cite key=sican></d-cite>, ResearchAgent<d-cite key=baek2025researchagent></d-cite>) are primarily driven by literature, and several explicitly instantiate these heuristics. Scideator<d-cite key=radensky2024scideator></d-cite> and MOOSE-Chem<d-cite key=yangmoose></d-cite> adopt H5 by creating novel syntheses across papers: Scideator recombines research facets across papers (e.g., pairing the purpose of one study with the mechanism of another) to generate inventive ideas, while MOOSE-Chem combines background knowledge and inspiration knowledge from different papers.

> On the other hand, Chain-of-Ideas<d-cite key=li2024chain></d-cite> and CoQuest<d-cite key=liu2024ai></d-cite> share the idea of H7, structuring literature into either sequential chains or graphs to uncover new gaps and opportunities.

### Data-Driven Discovery
*Identifying patterns or gaps emerging from data itself.*

Researchers may begin with `open-ended qualitative exploration (H8)`, using methods such as ethnography or case studies to gather rich narratives that spark new insights. They can also `assemble novel datasets (H9)` by connecting disparate sources or mining raw data, thereby enabling questions that were previously intractable. In addition, advances in technology make it possible to `explore novel techniques (H10)` in measurement, computation, and analysis, broadening the range of feasible investigations.

Beyond collecting and creating data, researchers can actively interact with it: one strategy is to `think through action (H11)`, building prototypes, running pilot studies, or testing preliminary interventions to obtain rapid feedback and refine ideas. Finally, `computer simulation (H12)` allows the exploration of complex systems through modeling, testing “what-if” scenarios, and generating hypotheses that would be difficult to evaluate directly in the real world.

**In human research:**
> By analyzing qualitative interviews without preset hypotheses, researchers inductively derive new theories of social processes through the methodology of grounded theory<d-cite key=glaser1967discovery></d-cite>, an exemplar of H8.

**In LLM-based ideation:**
> Nexus<d-cite key=gong2024nexus></d-cite> illustrates H9 by linking previously disconnected data sources to create new opportunities for hypothesis generation. Liu et al.<d-cite key=liu2025improving></d-cite> demonstrate H11 through integrating preliminary validation on empirical datasets into the ideation process. MOOSE-Chem3<d-cite key=liu2025moose></d-cite> applies H12 by simulating experimental feedback to refine and rank hypotheses, treating simulation as a proxy for costly laboratory experimentation.

### Direct Manipulation
*Actively tweaking variables, models, or assumptions.*

When you already have a proposition, here are heuristics designed to break conventional thinking patterns. One technique is to `challenge core assumptions (H13)` by systematically questioning foundational beliefs, considering circumstances in which the opposite might hold, or reversing presumed causal directions. Researchers may also `probe the system with controlled changes (H14)`, testing behavior under both small perturbations and large boundary-pushing shifts to expose the system’s structure and limits. Lastly, they can `reconfigure the conceptual framework (H15)` by altering an idea’s core components, such as variables and relationships, to reveal overlooked dynamics. For example, introducing moderators can reshape causal interpretations, while removing well-established effects may uncover hidden patterns.

**In human research:**
> While these heuristics are not yet fully implemented in LLM-based ideation systems, they are deeply rooted in scientific discovery. For example, H13 echoes Albert Einstein’s rethinking of Newtonian absolutes in developing special relativity<d-cite key=einstein1905electrodynamics></d-cite>. Looking forward, such heuristics hold great promise for enriching hypothesis generation by encouraging deeper exploration and conceptual restructuring.

### Structured Analytical Approaches
*Applying structured thinking to deepen and refine emerging ideas.*

A critical step is to `stress-test ideas (H16)`, which involves actively trying to find flaws, counterarguments, or alternative explanations to strengthen the robustness of the core argument. The process can be enhanced by `alternating induction and deduction (H17)`, using iterative cycles of observing specific instances to induce general principles and then using those principles to deduce new, testable hypotheses. For more rigorous theorizing, scholars can `build formal models from core principles (H18)`, translating concepts into mathematical or logical languages to uncover their implications.

Inspiration can also come from `transferring conceptualizations analogously (H19)`, where successful concepts or methods from one field are adapted to provide new perspectives in another field. Finally, the use of `thought-diversifying tools (H20)`, such as checklists and conceptual diagrams, encourages systematic exploration of a problem’s dimensions and generates a broader range of innovative solutions.

**In human research:**
> Early convolutional networks draw analogy from Hubel and Wiesel’s findings about hierarchical feature processing in the mammalian visual system<d-cite key=hubel1977ferrier></d-cite>, exemplifying H19.

**In LLM-based ideation:**
> Popper operationalizes H16 through iterative falsification. Literature Meets Data<d-cite key=liu2025literature></d-cite> demonstrates H17 by alternating between inductive literature-based and deductive data-driven reasoning. Yang et al.<d-cite key=yang2025robust></d-cite> exemplify H18 by formalizing hypotheses via Inductive Logic Programming, while H20 is embodied by meta-frameworks like this blog, systematically organizing heuristics to broaden the landscape of innovation.

Each category represents a different source or stage of inspiration within the research process. The first three often ignite new ideas—starting from what we see, know, or measure—while the latter two help us challenge and refine those ideas into more rigorous hypotheses.

Of course, **research is rarely linear**. Data-driven anomalies might loop back into new observations; reinterpreting an old theory might point us toward a fresh dataset; and structured analysis might reveal hidden assumptions that invite direct manipulation. Rather than separated paths, these heuristics offer a comprehensive toolkit for navigating the ideation process.

## Researchers Find the Heuristics Useful
To assess the real-world usefulness of our proposed Ideation Heuristics, we survey 13 researchers, including 8 postdocs and 5 professors. These participants represent a wide array of academic fields, including computer science, mathematics, natural science (chemistry, biology, material science), and social studies (law, sociology, and business). 

Each participant rates the usefulness of all 20 heuristics within their discipline on a three-point scale: *very useful*, *moderately useful*, or *not useful*.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-ideation-heuristics/survey_results.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

The heuristics successfully capture practical and widely recognized strategies for research ideation across disciplines. Across all responses, **55% of ratings are *very useful***, and an additional **36% are *moderately useful*.** Notably, **13 out of the 20 heuristics** are rated *very useful* by a majority of participants, and **every heuristic** is considered at least *useful* (i.e., *very* or *moderately useful*) by over 60% of respondents. 

While a broad consensus emerges, we also observe meaningful disciplinary differences, reflecting the distinct epistemic styles and methodologies of each field:

- **Computer Scientists** unanimously rate `think through action (H11)` and `reconfigure the conceptual framework (H15)` as *very useful*, underscoring their preference for iterative building, prototyping, and system manipulation.
- **Mathematicians**, by contrast, are the only group to unanimously endorse `alternate induction and deduction (H17)` as *very useful*, aligning with the field’s logical and proof-oriented nature.
- **Natural Scientists** emphasize empirical experimentation, giving highest ratings to `probe the system with controlled changes (H14)`, a heuristic that encapsulates the experimental method itself.
- **Social Scientists** prioritize heuristics tied to critical theory and qualitative reasoning, strongly favoring `open-ended qualitative exploration (H8)` and `challenge core assumptions (H13)`.

## How to Use the Heuristics
To make the ideation heuristics directly actionable, we develop a Claude skill called **`heuristic-ideation`**, inspired by the Claude Scientific Skills project<d-cite key=claude_scientific_skills_2025></d-cite>. The skill is available at <https://github.com/heuristic-ideation/heuristic-ideation-skill>.

You can use the skill by simply zipping the repository and uploading it to Claude. Once enabled, Claude automatically applies the heuristic-based framework during ideation, helping you propose research ideas and hypotheses in a more systematic way. The skill first identifies relevant heuristic categories, then selects specific heuristics by referencing their detailed descriptions, and finally applies them to guide the ideation process.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid path="assets/img/2026-04-27-ideation-heuristics/usage_example.png" class="img-fluid rounded z-depth-1" %}
    </div>
</div>