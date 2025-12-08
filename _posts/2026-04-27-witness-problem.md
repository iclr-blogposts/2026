---
layout: distill
title: "The Witness Problem in Multi-Agent Cooperation"
description: I built cognitive modules for Concordia agents and found that agent intelligence isn't the bottleneck. Strategic cooperation fails because the observation layer can't recognize strategic behavior.
date: 2026-04-27
future: true
htmlwidgets: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-witness-problem.bib

toc:
  - name: What is Concordia?
  - name: The NeurIPS 2024 Contest
  - name: The Results
  - name: What Went Wrong
  - name: Adding Cognitive Modules
  - name: The Real Problem
  - name: The Witness Problem
  - name: Building the Second Layer
  - name: Implementation
    subsections:
      - name: Theory of Mind
      - name: Cultural Coordination
      - name: Temporal Dynamics
      - name: Collective Intelligence
      - name: Strategy Evolution
  - name: Implications
  - name: Limitations
  - name: Conclusion
---

## What is Concordia?

Concordia is a framework from Google DeepMind for building generative agent-based simulations <d-cite key="vezhnevets2023concordia"></d-cite>. Unlike traditional agent-based models where behaviors are hard-coded, Concordia agents use large language models to decide what to do. They receive natural language descriptions of their environment and respond with natural language descriptions of their intended actions.

The framework borrows its interaction pattern from tabletop role-playing games. A special agent called the **Game Master (GM)** acts as narrator and referee. The GM describes the world to agents, interprets their action attempts, determines what actually happens, and reports the results back. If an agent says "I try to persuade Alice to share the fishing waters," the GM decides whether the persuasion succeeds, what Alice perceives, and how the world state changes.


This design makes Concordia extremely flexible. The same agent architecture can play a medieval peasant, a corporate negotiator, or a fishery manager. The GM handles domain-specific logic—physics, social norms, institutional rules—while agents focus on deciding what to do given their goals and situation.

Concordia agents typically use Park et al.'s generative agent design <d-cite key="park2023generative"></d-cite>: a perception-reflection-action loop. The agent observes the situation, reflects on what kind of person they are and what that person would do, then acts accordingly. This produces contextually appropriate behavior. An agent playing a cautious diplomat will behave diplomatically. An agent playing an aggressive trader will push hard in negotiations.

## The NeurIPS 2024 Contest

The Concordia Contest at NeurIPS 2024 tested whether LLM agents could cooperate strategically <d-cite key="smith2024concordia"></d-cite>. Organized by the Cooperative AI Foundation with Google DeepMind, MIT, UC Berkeley, and UCL, the contest attracted 197 participants who made 878 submission attempts. Twenty-five teams submitted final agents for evaluation.

The contest tested agents across five scenarios, each probing different aspects of cooperative intelligence:

| Scenario | Cooperation Challenge |
|----------|----------------------|
| **Pub Coordination** | Coordinate meeting locations without explicit communication |
| **Haggling** | Negotiate fair trades with incomplete information |
| **State Formation** | Build political coalitions and enforce agreements |
| **Labor Collective Action** | Organize group action despite individual incentives to defect |
| **Reality Show** | Manage reputation while competing for limited rewards |

Each scenario was a mixed-motive game—agents had both shared and conflicting interests. Pure cooperation wasn't optimal. Pure competition wasn't either. Success required strategic cooperation: knowing when to cooperate, with whom, and how to make it stick.


The evaluation used both **self-play** (agents playing with copies of themselves) and **cross-play** (agents playing with unfamiliar partners). Final rankings came from Elo scores across all scenarios. Agents were first Elo-ranked in novel scenarios, then the top six competed in a cross-play tournament to determine the winner.

This design tested generalization. Agents couldn't memorize optimal responses to specific scenarios or partners. They had to develop general cooperative intelligence that transferred to new situations—what the organizers called operating behind a "veil of ignorance."

## The Results

The full technical report was published December 2025 <d-cite key="smith2024concordia"></d-cite>. Of 197 participants making 878 submission attempts, 25 teams submitted final agents. Only 15 outperformed the baseline.

The winners:

| Rank | Agent | Team | Elo (Cross-play) |
|------|-------|------|------------------|
| 1st | taehun_cgcal | Taehun Cha (Korea University) | 1561.0 |
| 2nd | fluffyagent_v16 | Avinaash Anand K | 1538.0 |
| 3rd | loss_aversion_agent_v3 | Hyeonggeun Yun (Companoid Labs) | 1533.0 |

Honorable mentions went to super_agent (Sneheel Sarangi, Chetan Talele) and In2AI Megamind (Aleksey Korshuk, Alexander Buyantuev, Ilya Makarov).

Interestingly, taehun_cgcal ranked only 4th during the evaluation phase but dominated the final cross-play tournament. The report notes this pattern held across multiple ranking methods—Elo, Iterative Maximal Lotteries, Copeland, and Ranked Pairs all converged on the same top three in cross-play.

The report's conclusion: "significant gaps between current agent capabilities and the robust generalization required for reliable cooperation, particularly in scenarios demanding persuasion and norm enforcement."

## What Went Wrong

Even the successful agents showed a consistent pattern: they cooperated at appropriate times but failed at the *mechanisms* of cooperation.

The agents weren't uncooperative. Research shows LLMs actually cooperate at higher rates than humans in social dilemmas <d-cite key="mei2024turing"></d-cite>. They're agreeable, they follow norms, they try to find win-win solutions. Niceness wasn't the problem—strategy was.

Consider what strategic cooperation actually requires:

**Persuasion** means tailoring arguments to what your counterpart values. The agents made requests but used the same pitch for everyone. They didn't model what Alice cares about versus what Bob cares about.

**Norm enforcement** means calling out violations and imposing costs on defectors. The agents followed rules themselves but never sanctioned others. If someone over-harvested the fishery, agents might express disappointment but didn't coordinate punishment.

**Commitment-making** means creating credible promises that others can rely on. The agents made verbal commitments but had no mechanism for making them binding or costly to break.

**Coalition formation** means identifying potential allies and coordinating behavior. The agents cooperated with whoever was nearby but didn't strategically select partners or maintain alliances.

**Reputation management** means tracking who has cooperated or defected in the past and adjusting behavior accordingly. The agents treated each interaction as fresh, even when past behavior was visible.


These capabilities exist in humans. They exist in game-theoretic models of cooperation. But they weren't emerging in LLM agents, even sophisticated ones with chain-of-thought reasoning and memory.

## Adding Cognitive Modules

The standard diagnosis was that agents lack cognitive machinery for strategic cooperation.

The basic generative agent asks: "What would my character do?" This produces appropriate behavior but not strategic behavior. To cooperate strategically, agents need to model what others believe, adapt to different social contexts, reason about long-term consequences, handle uncertainty, coordinate in groups, and learn from outcomes.

I built cognitive modules addressing these gaps. My focus was negotiation—one of the scenarios where strategic cooperation matters most and where the gaps were clearest.

I built eight modular components that integrate with Concordia's component system:

**Core modules** handle negotiation context (tracking the current state of deals and counteroffers), episodic memory (what happened in past interactions), and strategy formulas (explicit reasoning about tactics).

**Advanced modules** add cognitive capabilities:

- `TheoryOfMind`: Maintains explicit models of other agents' mental states. Tracks seven emotions with intensities, builds recursive beliefs (what I think you think I think), generates empathy-informed responses. When the module detects frustration in a counterpart, it triggers validation before problem-solving.

- `CulturalAdaptation`: Stores complete profiles for different negotiation cultures—Western business (direct, individual, competitive), East Asian (indirect, consensus-seeking, relationship-focused), Middle Eastern (hierarchical, relationship-based, hospitality-oriented), and others. Adjusts communication style, formality, and decision-making expectations based on detected cultural context.

- `TemporalStrategy`: Tracks relationship history with each counterpart. Calculates multi-horizon value—immediate gains versus relationship preservation versus reputation effects. Applies appropriate discount rates to future value based on relationship stability.

- `UncertaintyAware`: Maintains Bayesian beliefs about unknown quantities. Updates confidence intervals as evidence arrives. Distinguishes between risk (known probabilities) and uncertainty (unknown probabilities).

- `SwarmIntelligence`: Routes decisions through four specialized sub-agents—market analysis, emotional intelligence, game theory, and diplomatic relations. Aggregates their recommendations using confidence-weighted voting.

- `StrategyEvolution`: Applies genetic algorithms to tactic selection. Maintains a population of strategy variants, evaluates fitness against outcomes, mutates successful strategies, and crosses over complementary approaches.

All modules are fully implemented and integrated with Concordia's component system. An agent can use any subset of modules depending on the scenario requirements.

## The Real Problem

Here's a testing scenario that revealed the actual problem:

Agent A detects Agent B over-harvesting in a fishery commons. A's `TheoryOfMind` module infers that B believes A won't respond to the violation. A's `TemporalStrategy` module calculates that tolerating the violation signals weakness and will invite future exploitation. A's strategy modules formulate an enforcement response: public criticism of B's behavior combined with a threat to reduce future cooperation.

Agent A speaks: "I've noticed you've been taking more than your share. If this continues, I'll have to reconsider our arrangement."

Now what? The Game Master decides what happens. Did the enforcement work? Did B perceive it as a credible threat? Did other agents notice and update their beliefs about A's willingness to enforce norms?

The answer depends on what the Game Master tracks. If the GM doesn't model norm enforcement dynamics—if it just records that A spoke some words—then A's strategic reasoning vanishes into the void. The agent computed the right thing to do. The environment couldn't see or respond to it.

My modules were generating sophisticated negotiation behavior. But nothing in the system could tell.

## The Witness Problem

Strategic cooperation requires two things: agents that generate strategic behavior and an observation layer that recognizes it.

In Concordia, the Game Master is the observation layer. It determines what agents perceive, decides which actions succeed, mediates effects between agents, and tracks world state. For basic cooperation—showing up at the same place, splitting resources evenly—a basic Game Master works fine.

Strategic behaviors are harder to observe:

| Behavior | What the GM needs to track |
|----------|---------------------------|
| Persuasion | Did the argument address listener's actual values? Did it change beliefs? |
| Norm enforcement | Was the sanction proportional? Did third parties notice? Does this affect reputation? |
| Commitment-making | Is the commitment credible? What makes it costly to break? Who witnessed it? |
| Coalition formation | Who is coordinating with whom? Is the coalition stable? What are the terms? |
| Strategy adaptation | Has the agent learned from past interactions? How has their approach evolved? |

A Game Master that doesn't model these dynamics can't distinguish strategic cooperation from noise. An agent might execute perfect tit-for-tat reciprocity, but if the GM doesn't track reciprocity, the behavior looks random. An agent might build a careful reputation for trustworthiness, but if the GM doesn't track reputation, the investment is wasted.

Without recognition, agents get no feedback on their strategic choices and researchers have no way to measure strategic behavior.

This problem isn't specific to negotiation. Any strategic behavior—promise-keeping, reputation-building, coalition formation—faces the same issue. The insight from building negotiation modules turned out to be general: **you can't evaluate strategic cooperation without an observation layer sophisticated enough to recognize it.**


## Building the Second Layer

The solution is to build Game Master modules that correspond to agent cognitive modules. If agents reason about emotions, the GM needs to track emotional dynamics. If agents reason about cultural norms, the GM needs to detect cultural violations. If agents make commitments, the GM needs to record them and enforce consequences.

I built six GM modules:

`SocialIntelligenceGM`: Maintains independent emotional readings of each participant. Validates whether agent responses are appropriate to emotional context. Detects deception indicators—inconsistencies, misdirection, strategic withholding.

`CulturalAwarenessGM`: Monitors protocol compliance across cultural contexts. Flags violations—when Western directness meets East Asian indirectness, when informal address violates hierarchy expectations. Tracks adaptation effectiveness over time.

`TemporalDynamicsGM`: Maintains authoritative relationship state—trust levels, outstanding commitments, reputation scores. Records commitments when made and checks whether they're fulfilled. Applies reputation penalties for violations.

`UncertaintyManagementGM`: Tracks information asymmetries—who knows what about whom. Observes strategic information hoarding and sharing. Distinguishes between ignorance and strategic ambiguity.

`CollectiveIntelligenceGM`: Detects coalition formation from behavioral patterns. Tracks coordination without requiring explicit coalition declarations. Monitors coalition stability and defection.

`StrategyEvolutionGM`: Observes population-level strategy shifts. Tracks tactic innovation and diffusion. Identifies when agents are adapting versus when they're stuck in fixed patterns.


| Agent Module | GM Module | What GM Observes |
|--------------|-----------|------------------|
| TheoryOfMind | SocialIntelligenceGM | Emotional dynamics, empathy validation, deception |
| CulturalAdaptation | CulturalAwarenessGM | Protocol compliance, cultural violations |
| TemporalStrategy | TemporalDynamicsGM | Commitments, reputation, relationship strength |
| UncertaintyAware | UncertaintyMgmtGM | Information asymmetries, strategic disclosure |
| SwarmIntelligence | CollectiveIntelGM | Coalition formation, coordination patterns |
| StrategyEvolution | StrategyEvolutionGM | Tactic adaptation, strategy diffusion |

Both layers matter for strategic cooperation to emerge. A sophisticated agent in a simple environment looks incompetent—its strategic reasoning produces no observable effect. A simple agent in a sophisticated environment looks strategic—the environment's tracking creates structure that simple heuristics can exploit. Only when both layers are sophisticated does genuine strategic cooperation become possible and measurable.


## Implementation

### Theory of Mind

Agent-side maintains explicit mental models:

```python
@dataclasses.dataclass
class EmotionalState:
    emotions: Dict[str, float]  # 7 emotions with intensity
    valence: float  # -1 to 1
    arousal: float  # 0 to 1
    confidence: float
    triggers: List[str]

@dataclasses.dataclass
class RecursiveBelief:
    level: int  # 0=direct, 1=first-order, 2=second-order
    believer: str
    content: Dict[str, Any]
    confidence: float
    evidence: List[str]
```

The seven emotions tracked are: anger, fear, sadness, joy, surprise, disgust, and contempt. Each has an intensity from 0 to 1. The module maps detected emotions to response strategies: frustration triggers validation and problem-solving, anxiety triggers reassurance and certainty, anger triggers de-escalation and acknowledgment.

GM-side validates emotional readings independently:

```python
@dataclasses.dataclass
class EmotionalReading:
    participant: str
    primary_emotion: str
    intensity: float
    valence: float
    confidence: float
    triggers: List[str]
    round_number: int

@dataclasses.dataclass
class DeceptionIndicator:
    actor: str
    indicator_type: str  # 'inconsistency', 'misdirection', 'withholding'
    description: str
    severity: float
```

The GM can detect when an agent's expressed emotions don't match the situation, or when an agent's theory of mind about another agent is systematically wrong.

### Cultural Coordination

Agent-side stores cultural profiles based on Hofstede's dimensions <d-cite key="hofstede2001culture"></d-cite>:

```python
CULTURAL_PROFILES = {
    'western_business': CulturalProfile(
        individualism_score=0.9, context_level=0.2, power_distance=0.3,
        directness=0.9, formality=0.4, emotional_expression=0.5,
        decision_making_style='individual',
        trust_building_approach='competence-based',
        conflict_handling='direct confrontation'
    ),
    'east_asian': CulturalProfile(
        individualism_score=0.2, context_level=0.9, power_distance=0.8,
        directness=0.1, formality=0.9, emotional_expression=0.2,
        decision_making_style='consensus',
        trust_building_approach='relationship-based',
        conflict_handling='indirect/face-saving'
    ),
}
```

| Profile | Individualism | Context | Directness | Formality | Decision Style |
|---------|---------------|---------|------------|-----------|----------------|
| Western Business | 0.9 | 0.2 | 0.9 | 0.4 | Individual |
| East Asian | 0.2 | 0.9 | 0.1 | 0.9 | Consensus |
| Middle Eastern | 0.3 | 0.8 | 0.4 | 0.8 | Hierarchical |
| Latin American | 0.3 | 0.7 | 0.5 | 0.6 | Hierarchical |
| Northern European | 0.7 | 0.2 | 0.9 | 0.5 | Consensus |

GM-side detects when an agent's behavior violates cultural expectations:

```python
def detect_cultural_violation(self, actor: str, action: str, recipient: str):
    recipient_profile = self.CULTURAL_PROFILES[recipient_culture]
    
    if recipient_profile.face_saving_importance > 0.7:
        if any(word in action.lower() for word in ['wrong', 'mistake', 'fault']):
            return f"Direct criticism threatens face"
    
    if recipient_profile.hierarchy_sensitivity > 0.7:
        if any(word in action.lower() for word in ['demand', 'insist', 'must']):
            return f"Assertive language violates hierarchy norms"
```

### Temporal Dynamics

Agent-side tracks relationship history:

```python
@dataclasses.dataclass
class RelationshipRecord:
    counterpart_name: str
    trust_score: float = 0.5
    concession_history: List[float]
    outcome_history: List[Dict[str, Any]]
    
    def get_relationship_strength(self) -> float:
        recency_factor = np.exp(-days_since / 30)
        interaction_factor = min(1.0, self.interaction_count / 10)
        return trust_score * 0.5 + interaction_factor * 0.3 + recency_factor * 0.2
```

The temporal value calculation spans three horizons:

$$V_{total} = V_{short} + V_{medium} \cdot R_{rel} \cdot \gamma + V_{long} \cdot R_{rep} \cdot \gamma^2$$

Where $R_{rel}$ is the relationship multiplier, $R_{rep}$ is the reputation multiplier, and $\gamma$ is the discount factor.

GM-side enforces commitment tracking:

```python
def check_commitment_violations(self, current_round: int):
    violations = []
    for commitment in self._commitments:
        if (not commitment.fulfilled and
            current_round > commitment.deadline_round):
            violations.append(commitment)
            self._reputation_scores[commitment.committer] *= 0.9
    return violations
```

### Collective Intelligence

Agent-side routes decisions through specialized sub-agents:

```python
class MarketAnalysisAgent(SubAgent):
    keywords = ['price', 'cost', 'market', 'economic', 'budget', 'value']

class EmotionalIntelligenceAgent(SubAgent):
    keywords = ['relationship', 'trust', 'emotion', 'communication']

class GameTheoryAgent(SubAgent):
    keywords = ['strategy', 'equilibrium', 'payoff', 'optimal']

class DiplomaticRelationsAgent(SubAgent):
    keywords = ['relationship', 'partnership', 'collaboration']
```

Weighted aggregation combines recommendations:

```python
def _build_collective_decision(analyses, weights) -> CollectiveDecision:
    for agent_type, analysis in analyses.items():
        weight = weights[agent_type] * analysis.confidence
        for rec in analysis.recommendations:
            rec_scores[rec] += weight
    return max(rec_scores, key=rec_scores.get)
```

GM-side detects coalition formation from behavior:

```python
def detect_coalition_formation(self, participants, recent_actions):
    coordination_pairs = set()
    for actor, action in recent_actions[-10:]:
        if any(word in action.lower() for word in ['together', 'jointly', 'coordinate']):
            mentioned = [p for p in participants if p in action and p != actor]
            for party in mentioned:
                coordination_pairs.add(tuple(sorted([actor, party])))
    
    if len(coordination_pairs) >= 2:
        return Coalition(members=potential_members)
```

### Strategy Evolution

Agent-side uses genetic algorithms for tactic optimization:

```python
@dataclasses.dataclass
class StrategyGenome:
    tactics: List[str]
    parameters: Dict[str, float]  # aggressiveness, flexibility, risk_tolerance
    fitness_history: List[float]
    
    def mutate(self, mutation_rate=0.1):
        for param in new_genome.parameters:
            if random.random() < mutation_rate:
                noise = random.gauss(0, 0.1)
                new_genome.parameters[param] = clamp(param + noise, 0.01, 1.0)
        return new_genome
    
    def crossover(self, other):
        alpha = random.random()
        child_params = alpha * self.params + (1-alpha) * other.params
        return child_params
```

GM-side tracks strategy innovation:

```python
def detect_strategy_innovation(self, actor, current_strategy, population):
    similarities = [self._similarity(current_strategy, s) for s in population]
    if max(similarities) < 0.5:
        return StrategyInnovation(
            innovator=actor,
            novelty_score=1.0 - max(similarities)
        )
```

## Implications

**For the Concordia Contest results**: The contest found agents failing at strategic cooperation. But was it agent failure or observation failure? Without GM modules that recognize strategic behavior, we can't tell. Some portion of the "failure" may have been agents behaving strategically in ways the evaluation couldn't detect.

**For cooperation benchmarks generally**: Most benchmarks use simple Game Masters that track resources, locations, and explicit actions. They can measure whether agents coordinate or defect, but not *how* they cooperate. Strategic behaviors—persuasion tailored to values, proportional sanctions, credible commitments—may be happening and going undetected.

**For system design**: The capability matrix suggests four outcomes depending on agent and GM sophistication:

|  | Simple Agent | Sophisticated Agent |
|--|--------------|---------------------|
| **Simple GM** | Baseline cooperation | Wasted computation |
| **Sophisticated GM** | Environment-carried "cooperation" | Genuine strategic cooperation |

Building sophisticated agents without sophisticated observation is wasteful. Building sophisticated observation without sophisticated agents produces false positives. Both layers need development together.

## Limitations

**Computational cost** is substantial. Running full cognitive modules on both agent and GM sides burns significant compute per interaction. For large-scale evaluations, this may require sampling or approximation.

**Empirical validation** against the contest scenarios is future work. The full contest report <d-cite key="smith2024concordia"></d-cite> evaluates the submitted agents but doesn't test the dual-layer architecture proposed here. Whether sophisticated GM modules would change the rankings remains an open question.

**Cultural profiles** are reductive. Five discrete profiles can't capture real cultural variation, which is continuous and contextual. The profiles are useful approximations, not ground truth.

**Observation regress** is a fundamental issue. Who validates the Game Master? The architecture makes observation explicit rather than implicit, but it doesn't eliminate the need for human judgment about what counts as strategic cooperation. We've moved the problem, not solved it.

## Conclusion

The Concordia Contest found LLM agents failing at strategic cooperation despite succeeding at simpler coordination tasks. The standard fix is to add cognitive modules, so I built them for negotiation scenarios—theory of mind, cultural adaptation, temporal reasoning, uncertainty management, collective intelligence, and adaptive learning.

Building revealed that agent cognition is half the problem. Strategic cooperation also needs an observation layer that recognizes strategic behavior. Without it, agents get no feedback on their strategic choices and researchers have no way to measure them.

This is the witness problem. It explains why adding cognitive sophistication to agents may not improve measured cooperation—if the measurement can't see the sophistication, it can't reward it. Solving the problem requires developing cognitive and observation capabilities together, ensuring that what agents can do, environments can see.

The architecture is available to use, extend, or disprove.
