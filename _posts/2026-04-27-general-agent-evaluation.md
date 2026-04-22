---
layout: distill
title: "Ready For General Agents? Let's Test It."
description: General-purpose agents are emerging, promising seamless deployment across domains. However, we currently do not measure their adaptability to diverse, unseen settings—a core requirement for true generality. We outline the key challenges and chart a path toward a unified evaluation framework designed to guide the development of general agents.

date: 2026-04-27
future: true
htmlwidgets: true
authors:
  - name: Elron Bandel
    url: "https://www.linkedin.com/in/elron/"
    affiliations:
      name: IBM Research
  - name: Asaf Yehudai
    url: "https://www.linkedin.com/in/asaf-yehudai/"
    affiliations:
      name:
  - name: Michal Shmueli-Scheuer
    url: "https://research.ibm.com/people/michal-shmueli-scheuer"
    affiliations:
      name:
bibliography: 2026-04-27-general-agent-evaluation.bib
toc:
  - name: Preliminary
  - name: The Shift to General Agents
  - name: The Promise of General Agents
    subsections:
      - name: Case 1 - Scientific Agents
      - name: Case 2 - SWE Agents
      - name: The Promise
  - name: State of General Agent Evaluation
    subsections:
      - name: Level 1 - Agentic Skills Evaluation
      - name: Level 2 - Domain-Agent Evaluation
      - name: Level 3 - Agentic Cross-Model Evaluation
      - name: Level 4 - Protocol-Centric Agent Evaluation
      - name: Level 5 - General Agent Evaluation
  - name: Challenges in General Agent Evaluation
    subsections:
      - name: Lack of Standardized Agent Interface
      - name: Lack of Standardized Environment Interfaces
      - name: Lack of a Standardized Researcher Interface
  - name: Existing Agent-Environment Protocols insufficiency
  - name: General Agent Evaluation Framework
    subsections:
      - name: A Meta-Protocol for General Agent Evaluation
      - name: Core Characteristics of a General Agent Evaluation Framework
  - name: Conclusions
---

Over the past few years, the NLP community has shifted from building domain-specific systems - such as standalone summarization or translation models - toward developing general-purpose language models <d-cite key="brown2020languagemodelsfewshotlearners"></d-cite><d-cite key="bommasani2022opportunitiesrisksfoundationmodels"></d-cite>. Many see this trend as a contemporary example of Richard Sutton's "bitter lesson" <d-cite key="sutton2019bitter"></d-cite>:

> "The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin."

This transition was not abrupt. It emerged from a long sequence of incremental advances that gradually expanded the scope and capability of models. Along the way, domain-specific solutions and increasingly general methods coexisted, each informing and accelerating the other.

In this blog post, we argue that a similar shift is now unfolding in the field of AI agents: the field is moving from domain-specialized agents toward increasingly general-purpose ones. Agents that can address diverse types of multi-step tasks across different target domains and previously unseen environments.

This development highlights the need for a unified evaluation framework for general-purpose agents that assesses their abilities across environments and compares different architectures. Such a framework is crucial for tracking progress, identifying gaps, and guiding the development of next-generation general agents <d-cite key="bandel2026positionagentic"></d-cite>. It is also essential for evaluating the core of generality itself: an agent's ability to integrate into new environments and perform successfully.

We begin by introducing a shared terminology for discussing domain and general agents, their evaluation, and the agent-to-environment communication protocol ([Sec.2](#preliminary)). Next, we describe how today's domain agents are evolving toward greater generality and outline the effect we expect it to have on general agents ([Sec.3](#the-shift-to-general-agents)). The benefits of general-purpose agents and their advantages through two representative use cases follow in [Sec.4](#the-promise-of-general-agents), motivating the need for evaluation solutions that can assess general agent capabilities. For that end, we survey the current landscape of agent evaluation, presenting a five-level taxonomy ([Sec.5](#state-of-general-agent-evaluation)), and detail the limitations that make existing approaches insufficient for easily evaluating general agents ([Sec.6](#challenges-in-general-agent-evaluation)). We then assess whether existing agentic protocols can address these limitations ([Sec.7](#existing-agent-environment-protocols)). Finally, we outline key requirements that a general agent solution needs to fulfill ([Sec.8](#general-agent-evaluation-framework)).

We hope this blog will help clarify what general agents are, increase awareness of their emergence from domain-specific agents, and highlight the gaps in their evaluation. More broadly, we aim for it to serve as a call to action that inspires a community-wide effort to develop rigorous, scalable, and actionable evaluation frameworks. We believe that as agents become more general and autonomous, such frameworks are essential to guide their progress.

## Preliminary

AI agents are autonomous, goal-driven systems that perform multi-step tasks by interacting with their environment <d-cite key="bandi2025rise"></d-cite>. The environment is the "world" the agent is situated in. The agent can observe and interact with the environment, obtaining observations according to its internal mechanism <d-cite key="cheng2024exploringlargelanguagemodel"></d-cite>. An agent deployed in the environment can interact with it to achieve its goal. Many of those terms were adopted and adjusted from the domain of reinforcement learning to the field of LLM and AI agents.

AI agents are systems composed of interacting algorithmic components for reasoning, planning, memory preservation, code execution, and more. The orchestration of these components, often referred to as the agent's architecture or scaffold, collectively determines the agent's behavior. A large language model typically serves as the central computational element, providing core capabilities for perception, reasoning, and generation. The agent can also have access to external capabilities like code execution and search.

Another important concept is the agent-to-environment communication protocol. This protocol interface describes the way the agent interacts with its environment. Main examples are web browsing, terminal, MCP, and tool schemas.

Most current agents are being developed with a specific domain in mind <d-cite key="Wang_2024"></d-cite><d-cite key="yehudai2025surveyevaluationllmbasedagents"></d-cite>. Common examples are web agents and software engineering agents (SWE agents). In such cases, the agent is restricted to a relevant set of components and tools, and the environment is tailored to the target domain.

To evaluate the capabilities of different domain agents, researchers defined domain-specific benchmarks. Such benchmarks require an environment in which the agent can operate, a set of tasks that describe the agent's goal, and a metric that measures whether the agent has achieved its task. These benchmarks enable the assessment of the efficacy of domain agents and the comparison of different backbone LLMs.

In contrast to these types of agents, general-purpose agents are designed to handle a diverse set of tasks across different environments. This requires them to be adaptable to different kinds of previously unseen environments, each with its own tools, requirements, and specific setup.

In essence, general agents are defined by their capacity to integrate into new problem spaces, absorb their domain knowledge, refine their behavior through interaction, and ultimately master the tasks they encounter.

## The Shift to General Agents

Large language models (LLMs) are general-purpose systems; they are designed to handle diverse tasks. Yet they have a fixed static knowledge of the world and can only interact with it by producing text. To overcome this challenge, researchers advise utilizing tools that allow LLMs to interact with the world (e.g., searching the web, running code). To further advance this ability, researchers also develop designed patterns such as ReAct <d-cite key="yao2023reactsynergizingreasoningacting"></d-cite> and CodeAct <d-cite key="wang2024executablecodeactionselicit"></d-cite> that facilitate a loop of interaction between the LLM and the environment. Such simple agents can be deployed in any environment to achieve multi-step tasks, making them early versions of general agents.

Although these agents are simple, equipping them with the right tools can provide an effective solution. For example, many providers now recommend loop-based agents or agent frameworks as a standard pattern for application development. Even massively used LLM interfaces for both consumer and developer have quietly evolved from a conversation with an LLM to an interaction with an agent equipped with tools for coding and searching.

On the other hand, such agents fall short when compared to more complex and specialized agents on target domains. Such domain agents are utilizing much more structure; they tend to have components for planning, memory, state tracking, tool use, and error handling to support reliable, iterative interaction with the domain environment. They are topping domain-specific leaderboards and providing real-world value. For example, SWE agents are already solving millions of GitHub issues without intervention <d-cite key="PRArena"></d-cite>, and deep research agents are being deployed to millions of users <d-cite key="McKay2025_OpenAIDeepResearch"></d-cite>.

While different domain agents are by design different from one another, they share similar components. If we look at different SWE agents, such as Claude Code, Codex-cli, and different deep research agents such as OpenAI and Perplexity ones, they are all sharing similar algorithmic components <d-cite key="bgauryy_open-docs_2025"></d-cite><d-cite key="langchain_ai_open_deep_research_2025"></d-cite> documented in [open-docs](https://github.com/bgauryy/open-docs). Moreover, each of those components is not specific to its domain, but is a general component that could be used for any target domain. As a result, such domain agents that rely on general components can be easily adopted to other domains and tasks. For example, recently, Anthropic published:
"Over the past several months, Claude Code has become far more than a coding tool. At Anthropic, we've been using it for deep research, video creation, and note-taking, among countless other non-coding applications. In fact, it has begun to power almost all of our major agent loops."
Additionally, they released their [Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) to serve as building blocks for developing other agents <d-cite key="anthropic_claude_agent_sdk_2025"></d-cite>. This demonstrates that domain-specific agents are becoming more proficient and general, allowing them to target a wider set of tasks. This also shows the new type of general agents, composed of plugable general components that can apply to a diverse set of domains across different environments.


## The Promise of General Agents

In all machine learning tasks, simple and effective solutions are better than their specialized counterparts. Such solutions generalize better, are more robust, and less prone to overfitting <d-cite key="shalev2014understanding"></d-cite><d-cite key="haussler1990andrzej"></d-cite>. Specifically, unlike domain agents that can be over-specialized and tailored to a specific task or domain, general agents work with different environments, forcing them to generalize. They receive diverse signals from a wide range of environments, requiring them to be robust. As a result, they tend to be more cost-effective.

Next, we examine two concrete examples of SWE and scientific agents that demonstrate that even simple versions of general agents have a lower cost of development and are more cost-effective. To quantify these qualities, we measure lines of code (LOC) and agent average cost per task.

### Case 1: Scientific Agents

 ASTA Bench <d-cite key="bragg2025astabenchrigorousbenchmarkingai"></d-cite>, an effort towards benchmarking deep scientific research agents, provides a clear test. As shown in [Table 1](#table-1), the specialized ASTA-v0 system scores 53% at \$3.40 per task and contains subsystems exceeding 13,000 lines of code (LOC)[^asta-loc]. Yet the second-best system is a 358-line ReAct general agent (as a baseline architecture) scoring 44% at just \$0.31. On the literature-understanding subtask, ReAct scores 53%; although still below the Asta agent (62%) it is still outperforming both the specialized ASTA Paper Finder (21%) and OpenAI Deep Research (19%).

---
<div id="table-1" markdown="1">

| Agent   | LLMs used | ASTA score | Cost per task | LOC |
|---------|-----------|------------|---------------|-----|
| [ASTA-v0](https://github.com/allenai/asta-bench/tree/f507043) | Claude 4 Sonnet, Gemini 2.5 Flash, O3,<br>GPT 4.1, GPT-4o | 53% | \$3.40 | >13,768[^asta-loc] |
| [ReAct](https://github.com/allenai/asta-bench/blob/f7e25392f4dda167f4e6d46b8c7c080eeeb4cc35/astabench/solvers/react/basic_agent.py)   | GPT-5 | 44% | \$0.31 | 358[^react-code] |

</div>

> [Table 1](#table-1): In scientific tasks, a tiny general agent (ReAct) approaches the performance of a large specialized system (ASTA-v0) at a fraction of the cost and complexity.

---

[^asta-loc]: Based on the python files in https://github.com/allenai/asta-paper-finder/tree/main/agents/mabool/api/mabool.
[^react-code]: Source: https://github.com/allenai/asta-bench/blob/f7e25392f4dda167f4e6d46b8c7c080eeeb4cc35/astabench/solvers/react/basic_agent.py.

### Case 2: SWE Agents

 In SWE-Bench  leaderboard<d-cite key="yang2025swesmith"></d-cite>, the specialized SWE-Agent scores 67%, but the tiny, domain-agnostic Mini SWE-Agent scores 65% while being 30 times smaller and about 7 times cheaper per run ([Table 2](#table-2)). Across both scientific and SWE settings, small general agents of a few hundred lines consistently achieve 70% to 95% of the performance of systems that are thousands of lines [^swebench-leaderboard].


[^swebench-leaderboard]: SWE-Bench verified leaderboard entries used here: SWE-Agent (67%, 2025-05-22) and mini-SWE-Agent (65%, 2025-07-26), source: https://www.swebench.com.

[^sweagent-cost]: Based on the reported per-task costs from the same SWE-Bench verified entries used above.
[^sweagent-loc]: Based on the python files in https://github.com/SWE-agent/SWE-agent/tree/main/sweagent/agent.
[^mini-sweagent-loc]: Source: https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py.

---

<div id="table-2" markdown="1">

| Agent | LLM | SWE-Bench score | Cost per task | LOC |
|-------|-----|-----------------|---------------|-----|
|[SWE-Agent](https://github.com/SWE-agent/SWE-agent/tree/main/sweagent/agent) | Claude 4 Sonnet | 67%[^swebench-leaderboard] | ~\$2.50[^sweagent-cost] | 4,161[^sweagent-loc] |
|[Mini SWE-Agent](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py) | Claude 4 Sonnet | 65%[^swebench-leaderboard] | \$0.37 | 131[^mini-sweagent-loc] |

</div>

> [Table 2](#table-2): In software engineering tasks, a minimal general agent nearly matches a specialized SWE agent while being far smaller and cheaper.

---

### The Promise

As general agents mature, they will become more complex. Yet they hold great promise. Building a general agent can provide a singular solution applied to a wide range of cases. This can make such an effort much more impactful, similar to how OpenAI was able to surpass many task-specific solutions by providing a general-purpose LLM. It also provides a lower cost of development compared to building many domain-specific ones. Additionally, it reduces the development time required for building domain-specific agents, as they can start from a general agent and further adapt it to their target domain, similar to fine-tuning a general-purpose LLM for a specific task.

## State of General Agent Evaluation

There is a shift in the field of AI agents from domain-specific agents to general-purpose ones, and a general agent can be easier to develop compared to many domain-specific agents, more robust, and more cost-effective. This shift raises the need for evaluation solutions that are suitable for evaluating a general agent. This requires a framework that enables running the same agent across different benchmarks and environments to evaluate adaptability. Yet, there is no such solution.

To address this gap, we start by organizing the agent evaluation solutions space into five levels, starting from the specific to the more general. This organization can help us outline the fifth level that describes the missing level for evaluating any general agent across environments.

### Level 1: Agentic Skills Evaluation

The first level focuses on evaluating LLMs on agentic skills, such as reasoning, planning, and tool calling, without embedding them in a dynamic environment. Benchmarks in this level provide a textual prompt and expect a textual response. The model's response is assessed independently of any interaction loop or adaptive environment. Consequently, these evaluations measure whether a model can demonstrate an agentic capability in principle, but they do not assess whether the model can deploy that capability reliably in realistic long-horizon tasks. Nonetheless, they can provide insight into the ability of a model to succeed in certain components.

Representative benchmarks of this level include GSM8K <d-cite key="cobbe2021trainingverifierssolvemath"></d-cite>, which evaluates step-by-step mathematical reasoning; HotPotQA <d-cite key="yang2018hotpotqadatasetdiverseexplainable"></d-cite>, which tests multi-hop question answering; and BFCL <d-cite key="patil2023gorilla"></d-cite>, which measures tool-use capabilities. They were the basis for the reasoning and action design pattern, ReAct, on multi-step tasks. Together, they exemplify evaluations that probe fundamental agentic skills but stop short of testing genuine agentic behavior.

### Level 2: Domain-Agent Evaluation

The second level uses interactive environments, such as web browsers, applications, or terminal interfaces, with a set of tools that allow the agent to interact with them. The agent gets tasks requiring multiple steps that it needs to perform by interacting with the environment. The agent sequence of LLM and tool calls, named the agent trajectory, is then evaluated by an environment-specific metric that assesses whether the agent task was achieved.

These benchmarks provide high value for assessing domain-specific agent capabilities. However, each benchmark often defines its own custom setup, leading to agent logic that is tightly coupled to each environment. As a result, some benchmarks are used to assess LLMs in agentic environments, while others require agents tailored to the specific benchmark, making it hard to compare the same agent across benchmarks.

Representative examples include Tau-Bench <d-cite key="yao2024tau"></d-cite><d-cite key="barres2025tau2"></d-cite> for customer-service scenarios, AppWorld <d-cite key="trivedi-etal-2024-appworld"></d-cite> for multi-application tasks, WebArena <d-cite key="zhou2024webarenarealisticwebenvironment"></d-cite> for browser interactions, TerminalBench <d-cite key="tbench_2025"></d-cite> for Linux command-line tasks, and SWE-Bench <d-cite key="yang2025swesmith"></d-cite> for solving GitHub issues.

### Level 3: Agentic Cross-Model Evaluation

The third level focuses on providing a standardized evaluation harness for reproducible agent evaluations across various domain-specific benchmarks. It can be used for comparing LLMs as the backbone of different agents, or to compare domain-specific agents on a single benchmark. Yet, their standardization does not allow running the same agent across different benchmarks.

A representative example is HAL <d-cite key="hal"></d-cite> compiled nine benchmarks from the second level. Each environment in HAL comes with its own fixed agent setup, and users can easily change the backbone model of an agent or add support to a new domain-specific agent. Hence, this level still does not support comparing different agent architectures across environments as needed for general agent assessment.

### Level 4: Protocol-Centric Agent Evaluation

The fourth level moves beyond fixed agent setups by defining a standardized interaction protocol, such as a unified browser API or terminal interface, that any agent can implement. Instead of each environment defining its own custom agent-to-environment communication protocol, this standardization forces the agent to follow a specific protocol to be consistently evaluated across many environments. This enables comparisons not just across models, but also across different agent architectures.

However, protocol-centric frameworks still impose a specific mode of interaction. Agents built around fundamentally different communication protocols, such as those using the Model Context Protocol (MCP), cannot be evaluated in their native form. They must be forced through the protocol's interface, which can obscure their design and distort performance. For example, [Harbor](https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/base.py) <d-cite key="tbench_2025"></d-cite> evaluates agents through a command-line protocol, preventing MCP-based agents like Claude Code from being tested as intended.

Representative examples include BrowserGym <d-cite key="chezelles2025browsergym"></d-cite>, which standardizes browser interaction across diverse web tasks, and [Harbor](https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/base.py) <d-cite key="tbench_2025"></d-cite>, which provides a unified terminal protocol.

### Level 5: General Agent Evaluation

A fifth level, still missing today, would provide a framework for general agent evaluation<d-cite key="bandel2026generalagentevaluation"></d-cite><d-cite key="bandel2026positionagentic"></d-cite>. It will enable the evaluation of the same agent across environments without being forced to use a specific communication protocol. It would reveal how an agent actually performs in realistic settings, how design choices influence outcomes, and how well agents generalize across diverse tasks and environments. Achieving this level of flexibility and fairness remains an open challenge, with the lack of a unified, protocol-agnostic evaluation paradigm as a central barrier to progress toward genuinely general-purpose agents.

---

<div id="table-3" markdown="1">

| Level | Cross-model | Agentic environment interaction | Cross-environment | Cross-agent | Protocol-agnostic | Examples |
|-------|-------------|---------------------------------|-------------------|-------------|-------------------|----------|
| 1: Agentic model skills | Yes | No | No | No | No | BFCL, GSM8K, HotPotQA |
| 2: Interactive agentic model | Yes | Yes | No | No | No | Tau-Bench, AppWorld, WebArena, TerminalBench |
| 3: Cross-model harness | Yes | Yes | Yes | No | No | HAL |
| 4: Protocol-centric | Yes | Yes | Yes | Yes | No | BrowserGym, Harbor |
| 5: General agent evaluation | Yes | Yes | Yes | Yes | Yes | (missing) |

</div>

> [Table 3](#table-3): Comparison of the five levels of agent evaluation, highlighting cross-model, cross-environment, cross-agent, and protocol coverage.

---

## Challenges in General Agent Evaluation

Benchmarking general agents in different benchmarking environments is challenging; existing benchmarks were typically designed with a specific agent domain and goal in mind, such as a user-facing conversational agent, a computer-using autonomous coder, or a web-navigation agent. These design choices led to incompatible setups, preventing the development of a single, unified evaluation protocol that any agent can be seamlessly integrated into. Below, we outline key challenges that must be resolved to provide a unified standard for general agent evaluation.

### Lack of Standardized Agent Interface

Many benchmarks implicitly assume the tested agent possesses certain built-in, domain-specific capabilities. For example:

- [Tau-Bench](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/base.py) <d-cite key="yao2024tau"></d-cite><d-cite key="barres2025tau2"></d-cite> assumes an agent that can inherently message or converse with a user:

```python
class BaseAgent(ABC, Generic[AgentState]):
    """Base agent class that defines the common interface for agents."""
    @abstractmethod
    def generate_next_message(..., message: UserMessage | ToolMessage | MultiToolMessage..) -> ... AssistanceMessage:
        ...
```

- [WebArena](https://github.com/web-arena-x/webarena/blob/main/agent/agent.py) <d-cite key="zhou2024webarenarealisticwebenvironment"></d-cite> assumes an agent whose entire perceptual and action space is mediated through a browser interface controlled by predefined actions:

```python
class Agent:
    """Base class for the agent"""
    def next_action(..., trajectory: Trajectory,...) -> Action:
        # ActionType: NONE, SCROLL, KEY_PRESS, MOUSE_CLICK, KEYBOARD_TYPE, MOUSE_HOVER, CLICK, TYPE, HOVER,
        # PAGE_FOCUS, NEW_TAB, GO_BACK, GO_FORWARD, GOTO_URL, PAGE_CLOSE, ...
        ...
```

- [Terminal Bench](https://github.com/laude-institute/harbor/blob/main/src/harbor/agents/base.py) <d-cite key="tbench_2025"></d-cite> assumes an agent whose interaction interface is a computer with a command line:

```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(..., environment: BaseEnvironment) -> None:
        ...
class BaseEnvironment(ABC):
    @abstractmethod
    async def exec(..., command: str, ...) -> ExecResult:
        """Executes a command in the environment..."""
```

These assumptions are mutually incompatible. A web-browsing agent cannot converse with a user, while a chat-oriented agent cannot click on a web element. Such rigid, benchmark-specific communication protocols make cross-environment evaluation impossible without substantial ad hoc engineering.

---

<div id="figure-1" markdown="1">

{% include figure.liquid path="assets/img/2026-04-27-general-agent-evaluation/benchmark_agent_cross.png" class="img-fluid" %}

</div>

> [Figure 1](#figure-1): Illustration of agents in different benchmarks and their incompatibility with the environment constraints of other benchmarks. Tau-Bench assumes user messaging, which does not align with TerminalBench or WebArena.

---

### Lack of Standardized Environment Interfaces

Benchmarks rarely specify, in an agent-agnostic way, what task the agent must perform, what information it should receive, or what actions it can take and how those actions affect the environment. As shown in [Table 4](#table-4), developers are often left to infer or invent these elements themselves.

SWE-Bench <d-cite key="yang2025swesmith"></d-cite> exemplifies this lack of specification. Although the benchmark defines issues the agent should solve, it does not supply standard instructions on how the issues should be solved, how the solution will be validated, or how the submitted solution should be structured. Without an explicit environment interface for agents solving the benchmark, every user must design their own, de-facto creating different setups for different agents, making evaluations difficult to compare.

Other benchmarks present the opposite issue: they do communicate environment semantics, but only in a form tailored to a specific agent architecture. [Tau-Bench](https://github.com/sierra-research/tau2-bench/blob/main/src/tau2/agent/llm_agent.py) <d-cite key="yao2024tau"></d-cite><d-cite key="barres2025tau2"></d-cite> is a clear example. Instead of offering a standalone, environment-level description of how an agent should behave, key instructions appear only in the reference conversational LLM agent:

```python
AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
"""
```

While this instruction block expresses important aspects of the task, it mixes them with assumptions specific to a particular agent design-such as turn-by-turn interaction rules, usage of tools and JSON output formatting. These constraints make sense for a conversational LLM but do not generalize to other types of agents, such as code-act agents.

Across both cases, the core issue is the same: current benchmarks do not provide a standardized, agent-agnostic interface that cleanly communicates the task, available information, and action space. Without such an interface, researchers are forced to do additional work to interpret or construct these components themselves. This, in turn, creates room for inconsistencies across agent evaluations and introduces integration errors. Finally, the lack of standardization prevents testing from scaling to many environments, which is essential for evaluating an agent's ability to adapt to many different environments.

### Lack of a Standardized Researcher Interface

Evaluating general agents across many environments requires more than a common agent-environment integration interface. To support seamless experimentation-without hours spent integrating each new environment-researcher-facing interfaces must also be standardized and simplified. Today, every environment demands its own setup, scripts, and output formats. Researchers repeatedly lose time to these inconsistencies and risk introducing avoidable integration errors.

One example of this fragmentation is the process of collecting and interpreting results. Benchmarks report outcomes in different formats, store them in different locations, and follow different conventions. As shown in [Table 4](#table-4), even the basic metrics differ across three representative benchmarks-not only in which quantities are tracked, but also in terminology and aggregation standards used. A simple notion like success appears as a Boolean resolved flag in SWE-bench, a success field in AppWorld, and an implicit reward of 1 in Tau-bench. Interaction cost is reported as agent cost and user cost in Tau-bench but omitted entirely in AppWorld and SWE-bench.

Even when benchmarks measure conceptually similar quantities, they adopt incompatible formats, naming conventions, and aggregation methods.

These gaps underscore the need for consolidation and standardization to make large-scale evaluation of general agents feasible. This is not just a convenience - given the growing complexity of modern benchmarks, it is the only viable path to scalable general agent research.

---

<div id="table-4" markdown="1">

| Metric type | Tau-Bench | AppWorld | SWE-Bench |
|-------------|-----------|----------|-----------|
| Success (bool) | Reward = 1 | success | Resolved |
| Score (float) | Reward | Score | - |
| Termination | Termination reason | - | - |
| Duration | Duration | - | - |
| Number of interactions | Num messages | Steps | - |
| Agent cost | Agent cost | - | - |
| Environment cost | User cost | - | - |
| Task ID | Task ID | Task ID | Instance ID |
| Logs | Message log | Task logs | Test logs |
| Success rate (benchmark) | Avg reward | Task goal completion | Resolved counts |
| Cost aggregate (benchmark) | Avg agent cost | - | - |

</div>

> [Table 4](#table-4): Metrics differ across benchmarks, with incompatible names and formats even for basic success and cost reporting.

---

These gaps underscore the need for consolidation and standardization to make large-scale evaluation of general agents feasible.

## Existing Agent-Environment Protocols Insufficiency

Several protocols have recently emerged to standardize interaction, discovery, and data exchange in agentic systems. Some of these can, in principle, support evaluation. For example, A2A <d-cite key="a2a_protocol"></d-cite> provides a standardized way for agents to discover each other's capabilities, negotiate interaction modes, and coordinate on collaborative tasks, which could inform how benchmarks deliver tasks to agents. MCP <d-cite key="anthropic_mcp_2024"></d-cite> defines a structured way for agents to access external tools, data resources, and prompt templates. In an evaluation setting, a benchmark could expose its environment via an MCP server.

In this section, we ask two questions:

1. Should the research community adopt a single protocol for general-agent evaluation?
2. Do current protocols satisfy the practical needs of evaluation today?

Regarding the first question, we caution against prematurely standardizing evaluation processes on a single protocol. Lock-in at this stage risks constraining innovation as agent capabilities and demands still evolve. More fundamentally, evaluation must remain capable of assessing everything-including the protocol itself-making early commitment counterproductive.

To address the second question, we use MCP as a case study for whether existing protocols can support end-to-end evaluation workflows.

MCP defines three core primitives: tools (invocable operations), resources (exposed data/content), and prompts (parameterized templates), along with mechanisms for streaming events and updates. While MCP provides a promising foundation for unifying agent-environment interaction, several gaps prevent it from serving as a complete solution for general-agent evaluation:

- Missing Support for Benchmark Task Semantics. Benchmarks center around tasks-the defined goal an agent is supposed to achieve. MCP does not offer a built-in way to represent or communicate such tasks. One could require that tasks always appear in either prompts, resources, or events, but doing so would essentially create a new protocol on top of MCP, showing that MCP by itself is not enough.
- Missing Support for Evaluation Workflows. Evaluation requires more than interaction; it depends on standardized metrics reporting, aggregation, logging, experiment tracking, and reproducibility. MCP is intentionally agnostic to these needs. While one could build evaluation workflows around MCP sessions, doing so would again amount to defining an additional protocol on top of MCP, rather than using MCP itself as the benchmarking standard.
- Inconsistent Ecosystem Adoption. Support for MCP remains uneven: many frameworks implement tool calling but not resources or prompts, resulting in inconsistent capabilities and substantial integration overhead ([Table 5](#table-5)). This fragmentation makes it difficult for benchmark developers to ensure that agents can reliably interact with their environments, and equally hard for agent developers to obtain consistent baselines across benchmarks.

---

<div id="table-5" markdown="1">

| Agent / MCP Integration | Tools | Resources | Prompts |
|-------------------------|-------|-----------|---------|
| [Smolagents](https://huggingface.co/docs/smolagents/reference/tools#smolagents.ToolCollection.from_mcp) | Yes | No | No |
| [Llama Stack](https://llamastack.github.io/docs/building_applications/tools#model-context-protocol-mcp) | Yes | No | No |
| [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/ref/mcp/server/) | Yes | No | Yes |
| [Codex CLI](https://developers.openai.com/codex/mcp/) | Yes | No | No |
| [Claude Code](https://www.anthropic.com/news/claude-code-remote-mcp) | Yes | Yes | No |

</div>

> [Table 5](#table-5): MCP integration across common agent frameworks. In many cases, only partial protocol components are implemented.

---

These limitations indicate that while protocols like MCP and A2A lay important groundwork, they do not yet meet the full requirements of standardized general-agent evaluation.

Since no current protocol perfectly fits evaluation needs, the next section explores how a shared evaluation framework can still be built.

## General Agent Evaluation Framework

Advancing general-purpose agents requires an evaluation framework that is itself general: capable of supporting different environments, diverse agent architectures, and multiple communication protocols. Yet standardizing such evaluation is intrinsically difficult. Environments vary widely, implicit assumptions fragment the space, and existing protocols address only parts of the challenge. Even promising standards such as MCP provide useful building blocks but remain incomplete. These pressures motivate the need for a unifying layer that can support evaluating any agent on any benchmark without restriction to a specific communication protocol.

### A Meta-Protocol for General Agent Evaluation

At the core of such a framework is a meta-protocol: an abstract, protocol-agnostic layer that defines the semantics of evaluation independently of any concrete agent-environment protocol. Current benchmarks implicitly bind evaluation to a specific communication protocol, thereby entangling agent performance with protocol-specific setup.

The meta-protocol needs to specify how tasks, actions, observations, documentation, and termination conditions are represented while allowing different communication protocols. Such a protocol can allow communication interfaces, such as web browsing, terminal, MCP, and tool schemas, without altering the evaluation semantics.

By enabling protocol modularity rather than protocol uniformity, a meta-protocol allows agents to be evaluated in their native interaction modes and makes cross-protocol comparisons meaningful. Under this abstraction, the practical requirements of a general agent evaluation framework can be defined cleanly and without protocol-specific assumptions.

### Core Characteristics of a General Agent Evaluation Framework

- Environment-agnostic agent interface. The agent-facing interface should function across any environment. Environments expose a minimal, standardized schema for actions, observations, tasks, and documentation, with all assumptions explicit and discoverable rather than embedded in reference agents.
- Agent-agnostic environment interface. Environments should not assume a specific agent architecture, reasoning style, or protocol. Simple ReAct agents, memory-augmented agents, MCP-based agents, browser agents, and command-line agents should all integrate without modification.
- Plug-and-play modularity. Models, agent architectures, interaction protocols, and environments should be swappable without altering the evaluation protocol. This enables controlled comparisons across dimensions within a unified setup.
- Standardized reporting and transparency. The framework should define consistent reporting conventions, including metrics for success, robustness, interaction efficiency, and cost. Observability layer support for logging and full trajectory tracking.


Such a framework would let us measure the core property that makes general agents uniquely valuable: adaptability to new environments with minimal re-engineering.

## Conclusions

Ultimately, an evaluation framework for general agents must measure the core capability that defines general agents: operating effectively in unfamiliar environments and solving tasks without environment-specific manual tuning. This adaptability enables agents to scale across domains and deliver broad practical value. A meta-level evaluation protocol is therefore essential-one that can allow testing any agent in any environment without intervention, ensuring fair and consistent comparison. Without such a framework, the field will have limited ability to reliably measure progress and to guide the development of general-purpose agents.

---

<figure id="figure-2" class="text-figure" markdown="1">

### Checklist for General Agent Evaluation Framework

- **Environment-Agnostic Agent Interface**
  - [ ] Works across any environment.
  - [ ] Standardized actions, observations, tasks, and documentation.
  - [ ] No hidden assumptions or reference-agent logic.

- **Agent-Agnostic Environment Interface**
  - [ ] Compatible with any agent architecture or control loop.
  - [ ] Neutral to reasoning style, memory, tools, and workflow.
  - [ ] Independent of protocol (MCP, browser API, CLI, etc.).

- **Strict Environment--Agent Separation**
  - [ ] Environments expose only actions, observations, state semantics, and documentation.
  - [ ] No strategy hints, behavioral instructions, or scaffolding.
  - [ ] No environment helpers compensating for agent capabilities.

- **Discovery-Oriented Evaluation**
  - [ ] Explicit mechanisms for discovering schema and domain knowledge.
  - [ ] Metrics for discovery efficiency and generalization.

- **Standardized Reporting and Transparency**
  - [ ] Unified reporting protocol across environments.
  - [ ] Metrics for success, cost, efficiency, robustness, and discovery quality.
  - [ ] Full trajectory and action logging for reproducibility.

- **Cross-Dimensional Comparability**
  - [ ] Supports comparisons across models, agents, environments, and protocols.
  - [ ] Separates model, architecture, protocol, and environment effects.

- **Efficiency and Practicality**
  - [ ] Cost-aware evaluation with token and compute tracking.
  - [ ] Scalable execution (parallelism, batching, lightweight environments).

- **Extensibility and Future-Proofing**
  - [ ] Modular integration of new environments and task families.
  - [ ] Agnostic to future agent protocols and architectural innovations.
  - [ ] Meta-protocol supports evaluation of emerging standards.

</figure>

> [Figure 2](#figure-2): Checklist of requirements for a general agent evaluation framework.

---
