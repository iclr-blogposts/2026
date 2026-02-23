---
layout: distill
title: "Getting SAC to Work on a Massive Parallel Simulator: An RL Journey With Off-Policy Algorithms"
description: This post details how to get the Soft-Actor Critic (SAC) and other off-policy reinforcement learning algorithms to work on massively parallel simulators (e.g., Isaac Sim with thousands of robots simulated in parallel). In addition to tuning SAC for speed, the post also explores why SAC fails where PPO succeeds, highlighting a common problem in task design that many codebases share.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

# Mermaid diagrams
mermaid:
  enabled: false
  zoomable: false

# Anonymize when submitting
# authors:
#  - name: Anonymous

authors:
   - name: Antonin Raffin
     url: https://araffin.github.io/
     affiliations:
       name: German Aerospace Center (DLR)

# must be the exact same name as your blogpost
bibliography: 2026-04-27-sac-massive-sim.bib

# Add a table of contents to your post.
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - please use this format rather than manually creating a markdown table of contents.
toc:
  - name: "A Suspicious Trend: PPO, PPO, PPO, ..."
  - name: Why It Matters? - Fine Tuning on Real Robots
  - name: (The Path of Least Resistance) Hypothesis
  - name: The Hunt Begins
  - name: PPO Gaussian Distribution
  - name: SAC Squashed Gaussian
  - name: Quick Fix
  - name: "Transition: What Does That Mean for the RL Community?"
  - name: Tuning for Speed (Part II)
  - name: Defining Proper Action Bound - Extracting the Limits with PPO
  - name: "Need for Speed or: How I Learned to Stop Worrying About Sample Efficiency"
  - name: Does it work? - More Environments
  - name: Solving Harder Environments
  - name: Conclusion
    # subsections:
    #   - name: "Appendix: Affected Papers/Code"
    #   - name: "Appendix: Note on Unbounded Action Spaces"
    #   - name: "Appendix: What I Tried That Didn't Work"
    #   - name: "Appendix: SB3 PPO (PyTorch) vs. SBX PPO (Jax) - A Small Change in the Code, a Big Change in Performance"
---


<!--This post details how I managed to get the Soft-Actor Critic (SAC) and other off-policy reinforcement learning algorithms to work on massively parallel simulators (think Isaac Sim with thousands of robots simulated in parallel).
If you follow the journey, you will learn about overlooked details in task design and algorithm implementation that can greatly impact performance.-->

Spoiler alert: [quite a few papers/code](#appendix-affected-paperscode) are affected by the problem described below.

This post is divided into two main parts.
The first part analyzes why SAC does not work out of the box in Isaac Sim environments (until the [quick fix section](#quick-fix)).
The [second part](#tuning-for-speed-part-ii) discusses how to tune SAC for speed and make it perform as good as PPO.

## A Suspicious Trend: PPO, PPO, PPO, ...

The story begins a few months ago when I saw another paper using the same recipe for learning locomotion: train a PPO<d-cite key="schulman2017proximal"></d-cite> agent in simulation using thousands of environments in parallel and domain randomization, then deploy it on the real robot.
This recipe has become the standard since 2021, when ETH Zurich and NVIDIA showed that it was possible to learn locomotion in minutes on a single workstation<d-cite key="rudin2022learning"></d-cite>.
The codebase and the simulator (called Isaac Gym<d-cite key="makoviychuk2021isaac"></d-cite> at that time) that were published became the basis for much follow-up work<d-footnote>Like the <a href="https://www.youtube.com/watch?v=7_LW7u-nk6Q">BD-1 Disney robot</a></d-footnote>.

As an RL researcher interested in learning directly on real robots, I was curious and suspicious about one aspect of this trend: why is no one trying an algorithm other than PPO<d-footnote>I was not the only one asking why SAC doesn't work: <a href="https://forums.developer.nvidia.com/t/poor-performance-of-soft-actor-critic-sac-in-omniverseisaacgym/266970">nvidia forum</a> <a href="https://www.reddit.com/r/reinforcementlearning/comments/lcx0cm/scaling_up_sac_with_parallel_environments/">reddit1</a> <a href="https://www.reddit.com/r/reinforcementlearning/comments/12h1faq/isaac_gym_with_offpolicy_algorithms">reddit2</a></d-footnote>?
PPO benefits from fast and parallel environments<d-cite key="berner2019dota"></d-cite>, but PPO is not the only deep reinforcement learning (DRL) algorithm for continuous control tasks, and there are alternatives like SAC<d-cite key="haarnoja2018soft"></d-cite> or TQC<d-cite key="kuznetsov2020tqc"></d-cite> that can lead to better performance<d-cite key="huang2023openrlbenchmark"></d-cite>.

So I decided to investigate why practitioners do not use these off-policy algorithms, and maybe why they don't work with massively parallel simulators.


## Why It Matters? - Fine Tuning on Real Robots

If we could make SAC work with these simulators, then it would be possible to train in simulation and fine-tune on the real robot using the same algorithm (PPO is too sample-inefficient to train on a single robot).

By using other algorithms, it might also be possible to get better performance.
Finally, it is always good to better understand what works or not and why.
As researchers, we tend to publish only positive results, but a lot of valuable insights are lost in our unpublished failures.

<!--<div style="max-width: 50%; margin: auto;">
include figure.liquid path="https://araffin.github.io/slides/tips-reliable-rl/images/bert/real_bert.jpg" class="img-fluid"
<p style="font-size: 12pt; text-align:center;">The DLR bert elastic quadruped</p>
</div>-->


## (The Path of Least Resistance) Hypothesis

Before digging any further, I had some hypotheses as to why PPO was the only algorithm used:
- PPO is fast to train (in terms of computation time) and was tuned for the massively parallel environment.
- As researchers, we tend to take the path of least resistance and build on proven solutions (the original training code is open source, and the simulator is freely available) to get new, interesting results<d-footnote>Yes, we tend to be lazy.</d-footnote>.
- Some peculiarities in the environment design may favor PPO over other algorithms. In other words, the massively parallel environments might be optimized for PPO.
- SAC/TQC and derivatives are tuned for sample efficiency, not fast wall clock time. In the case of massively parallel simulation, what matters is how long it takes to train, not how many samples are used. They probably need to be tuned/adjusted for this new setting.

Note: During my journey, I will be using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)<d-cite key="raffin2021sb3"></d-cite> and its fast Jax version [SBX](https://github.com/araffin/sbx).

## The Hunt Begins

There are now many massively parallel simulators available (Isaac Sim, Brax, MJX, Genesis, ...), here, I chose to focus on Isaac Sim because it was one of the first and is probably the most influential one.

<!--As with any RL problem, starting simple is the [key to success](https://www.youtube.com/watch?v=eZ6ZEpCi6D8).-->
<!--<video controls src="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/ppo_trained.mp4">
</video>-->
{% include video.liquid path="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/ppo_trained.mp4" class="img-fluid rounded z-depth-1" controls=true %}
<div class="caption">
    A PPO agent trained on the <code>Isaac-Velocity-Flat-Unitree-A1-v0</code> locomotion task. <br>
    Green arrow is the desired velocity, blue arrow represents the current velocity
</div>

As with any RL problem, starting simple is the key to success <d-footnote>Also known as <a href="https://en.wikipedia.org/wiki/John_Gall_(author)#Gall's_law">Gall's law</a></d-footnote>.
Therefore, I decided to focus on the `Isaac-Velocity-Flat-Unitree-A1-v0` locomotion task first, because it is simple but representative.
The goal is to learn a policy that can move the Unitree A1 quadruped in any direction on flat ground, following a commanded velocity (the same way you would control a robot with a joystick).
The agent receives information about its current task as input (joint positions, velocities, desired velocity, ...) and outputs desired joint positions (12D vector, three joints per leg).
The robot is rewarded for following the correct desired velocity (linear and angular) and for other secondary tasks (feet air time, smooth control, ...).
<!--([truncation](https://www.youtube.com/watch?v=eZ6ZEpCi6D8))-->
An episode ends when the robot falls over and is timed out after 1000 steps<d-footnote>The control loop runs at <a href="https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L302">50 Hz</a>, so after 20 seconds</d-footnote>.

<!--After some [quick optimizations](https://github.com/isaac-sim/IsaacLab/pull/2022) (SB3 now runs 4x faster, at 60 000 fps for 2048 envs with PPO), I did some sanity checks.-->
To begin, I did some sanity checks.
I ran PPO with the [tuned hyperparameters](https://github.com/isaac-sim/IsaacLab/blob/f52aa9802780e897c184684d1cbc2025fafcef4a/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/agents/sb3_ppo_cfg.yaml) found in the repository, and it was able to quickly solve the task.
In 5 minutes, it gets an average episode return of ~30 (above an episode return of 15, the task is almost solved).
Then I tried SAC and TQC, with default hyperparameters (and observation normalization), and, as expected, it didn't work.
No matter how long it was training, there was no sign of improvement.

Looking at the simulation GUI, something struck me: the robots were making very large random movements.
Something was wrong.

<!--<video controls src="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/limits_train.mp4">
</video>-->
{% include video.liquid path="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/limits_train.mp4" class="img-fluid rounded z-depth-1" controls=true %}
<div class="caption">
    SAC out of the box on Isaac Sim during training.
</div>

Because of the very large movements, my suspicion was towards what action the robot is allowed to take.
Looking at the code, the RL agent commands a (scaled) [delta](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab/isaaclab/envs/mdp/actions/joint_actions.py#L134) with respect to a default [joint position](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py#L112):
```python
# Note desired_joint_pos is of dimension 12 (3 joints per leg)
desired_joint_pos = default_joint_pos + scale * action
```

Then, let's look at the action space itself (I'm using `ipdb` to have an interactive debugger):
```python
import ipdb; ipdb.set_trace()
>> vec_env.action_space
Box(-100.0, 100.0, (12,), float32)
```
Ah ah!
The action space defines continuous actions of dimension 12 (nothing wrong here), but the limits $$[-100, 100]$$ are surprisingly large, e.g., it allows a delta of +/- 1432 deg!! in joint angle when [scale=0.25](https://github.com/isaac-sim/IsaacLab/blob/f1a4975eb7bae8509082a8ff02fd775810a73531/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/rough_env_cfg.py#L30), like for the Unitree A1 robot.
<!--To understand why [normalizing](https://www.youtube.com/watch?v=Ikngt0_DXJg) the action space [matters](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) (usually a bounded space in $$[-1, 1]$$), we need to dig deeper into how PPO works.-->
To understand why normalizing the action space [matters](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html) (usually a bounded space in $$[-1, 1]$$), we need to dig deeper into how PPO works.

## PPO Gaussian Distribution

Like many RL algorithms, PPO relies on a probability distribution to select actions<d-cite key="shengyi2022the37implementation"></d-cite>.
During training, at each timestep, it samples an action $$a_t \sim N(\mu_\theta(s_t), \sigma^2)$$ from a Gaussian distribution in the case of continuous actions<d-footnote>This is not true for the PPO implementation in Brax which uses a squashed Gaussian like SAC.</d-footnote>.
The mean of the Gaussian $$\mu_\theta(s_t)$$ is the output of the actor neural network (with parameters $$\theta$$) and the standard deviation is a [learnable parameter](https://github.com/DLR-RM/stable-baselines3/blob/55d6f18dbd880c62d40a276349b8bac7ebf453cd/stable_baselines3/common/distributions.py#L150) $$\sigma$$, usually [initialized](https://github.com/leggedrobotics/rsl_rl/blob/f80d4750fbdfb62cfdb0c362b7063450f427cf35/rsl_rl/modules/actor_critic.py#L26) with $$\sigma_0 = 1.0$$.

This means that at the beginning of training, most of the sampled actions will be in $$[-3, 3]$$ (from the [Three Sigma Rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule)):

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/gaussian.svg" class="img-fluid" %}
<!--<img style="max-width:80%" src="assets/img/2026-04-27-sac-massive-sim/gaussian.svg"/>-->
<div class="caption">
    The initial Gaussian distribution used by PPO for sampling actions.
</div>


Back to our original topic, because of the way $$\sigma$$ is initialized, if the action space has large bounds (upper/lower bounds >> 1), PPO will almost never sample actions near the limits.
In practice, the actions taken by PPO will be far from them.
Now, let's compare the initial PPO action distribution with the Unitree A1 action space:

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/gaussian_large_bounds.svg" class="img-fluid" %}
<div class="caption">
    The same initial Gaussian distribution, but with the perspective of the Unitree A1 action space $$[-100, 100]$$
</div>

For reference, we can plot the action distribution of PPO after training<d-footnote>The code to record and plot action distribution is in the <a href="#appendix-plot-action-distribution">Appendix</a></d-footnote>:
<!--<d-footnote>The code to record and plot action distribution is on <a href="https://gist.github.com/araffin/e069945a68aa0d51fcdff3f01e945c70">GitHub</a></d-footnote>-->


{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/dist_actions_trained_ppo.svg" class="img-fluid" %}
<div class="caption">
    Distribution of actions for PPO after training (on 64 000 steps).
</div>

The min/max values per dimension:
```python
>> actions.min(axis=0)
array([-3.6, -2.5, -3.1, -1.8, -4.5, -4.2, -4. , -3.9, -2.8, -2.8, -2.9, -2.7])
>> actions.max(axis=0)
array([ 3.2,  2.8,  2.7,  2.8,  2.9,  2.7,  3.2,  2.9,  7.2,  5.7,  5. ,  5.8])

```

Again, most of the actions are centered around zero (which makes sense, since it corresponds to the quadruped initial position, which is usually chosen to be stable), and there are almost no actions outside $$[-5, 5]$$ (less than 0.1%): PPO uses less than 5% of the action space!

Now that we know that we need less than 5% of the action space to solve the task, let's see why this might explain why SAC doesn't work in this case<d-footnote>Action spaces that are too small are also problematic. See <a href="https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html">SB3 RL Tips and Tricks</a>.</d-footnote>.

<!-- Note: if in rad, 3 rad is already 171 degrees (but action scale = 0.25, so ~40 deg, action scale = 0.5 for Anymal). -->

## SAC Squashed Gaussian

SAC and other off-policy algorithms for continuous actions (such as DDPG, TD3, or TQC) have an additional transformation at the end of the actor network.
In SAC, actions are sampled from an unbounded Gaussian distribution and then passed through a [$$tanh()$$](https://pytorch.org/docs/stable/generated/torch.nn.Tanh.html) function to squash them to the range $$[-1, 1]$$.
SAC then linearly rescales the sampled action to match the action space definition, i.e. it transforms the action from $$[-1, 1]$$ to $$[\text{low}, \text{high}]$$<d-footnote>Rescale from [-1, 1] to [low, high] using <code>action = low + (0.5 * (scaled_action + 1.0) * (high - low))</code>.</d-footnote>.

What does this mean?
Assuming we start with a standard deviation similar to PPO, this is what the sampled action distribution looks like after squashing<d-footnote>Common PPO implementations clip the actions to fit the desired boundaries, which has the effect of oversampling actions at the boundaries when the limits are smaller than ~4.</d-footnote>:

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/squashed_vs_gaussian.svg" class="img-fluid" %}
<div class="caption">
    The equivalent initial squashed Gaussian distribution.
</div>

And after rescaling to the environment limits (with PPO distribution to put it in perspective):

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/squashed_rescaled.svg" class="img-fluid" %}
<div class="caption">
    The same initial squashed Gaussian distribution but rescaled to the Unitree A1 action space $$[-100, 100]$$
</div>

As you can see, these are two completely different initial distributions at the beginning of training!
The fact that the actions are rescaled to fit the action space boundaries explains the very large movements seen during training.
Also, it explains why it was impossible for SAC to learn anything useful.

## Quick Fix

When I discovered that the action limits were way too large, my first reflex was to re-train SAC, but with only 3% of the action space, to more or less match the effective action space of PPO.
Although it didn't reach PPO performance, there was finally some sign of life (an average episodic return slightly positive after a while).

Next, I tried to use a neural network similar to the one used by PPO for this task and reduce SAC exploration by having a smaller entropy coefficient<d-footnote>The entropy coeff is the coeff that does the trade-off between RL objective and entropy maximization.</d-footnote> at the beginning of training.
Bingo!
SAC finally learned to solve the task!

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/learning_curve.svg" class="img-fluid" %}
<div class="caption">
    Learning curve on the Unitree A1 task using 1024 envs.
</div>

{% include video.liquid path="https://b2drop.eudat.eu/public.php/dav/files/z5LFrzLNfrPMd9o/sac_trained_cut_1.mp4" class="img-fluid rounded z-depth-1" controls=true %}
<div class="caption">
    Trained SAC agent after the quick fix.
</div>

SAC Hyperparameters (the ones not specified are [SB3 defaults](https://github.com/araffin/sbx/blob/8238fccc19048340870e4869813835b8fb02e577/sbx/sac/sac.py#L54-L64)):
```python
sac_hyperparams = dict(
    policy_kwargs={
        # Similar to PPO network tuned for Unitree A1 task
        "activation_fn": jax.nn.elu,
        "net_arch": [512, 256, 128],
    },
    # When using 2048 envs, gradient_steps=512 corresponds
    # to an update-to-data ratio of 1/4
    gradient_steps=512,
    ent_coef="auto_0.006",
)
```

<!--That's all, folks? - -->

## Transition: What Does That Mean for the RL Community?

When I discovered the large action limits problem, I was curious to see how widespread it was in the community.
After a quick search, it turns out that a lot of papers/code are affected<d-footnote>A notable exception are Brax-based environments because their PPO implementation uses a squashed Gaussian, so the boundaries of the environments had to be properly defined.</d-footnote> by this large boundary problem (see a non-exhaustive [list of affected papers/code below](#appendix-affected-paperscode)).

Although the initial choice of bounds may be a conscious and convenient one (no need to specify the real bounds, PPO will figure it out), it seems to have worked a bit by accident for those who built on top of it, and probably discouraged practitioners from trying other algorithms.

<!--My recommendation would be to always have properly defined action bounds, and if they are not known in advance, you can always [plot the action distribution](https://gist.github.com/araffin/e069945a68aa0d51fcdff3f01e945c70) and adjust the limits when iterating on the environment design.-->
My recommendation would be to always have properly defined action bounds.
If they are not known in advance, you can [plot the action distribution](#appendix-plot-action-distribution) and adjust the limits when iterating on the environment design <d-footnote>More on that very soon ;)</d-footnote>.

## Tuning for Speed (Part II)

Although SAC can now solve the locomotion task on flat ground, it takes more time to train, is not consistent, and the performance is slightly below PPO's.
In addition, SAC's learned gaits are not as pleasing as PPO's, for example, SAC agents tend to keep one leg up in the air...

<!--By limiting the action space to 3% of the original size, and quickly tuning SAC (bigger network, reduced initial exploration rate), I could get SAC to learn to solve the Unitree A1 task on a flat surface in minutes.-->

<!--[Part II](../tune-sac-isaac-sim/) explores these aspects (and more environments), reviews SAC design decisions (for example, try to remove the squashed Gaussian), and tunes it for speed, but for now, let's see what this means for the RL community.-->

The second part of this post explores these aspects<d-footnote>I also present the ideas that didn't work and could use help (open problems) at the end of this post.</d-footnote>, as well as more complex environments.
It also details how to automatically tune SAC for speed (i.e., minimize wall clock time), to learn as fast as PPO.

<!--This second post details how I tuned the Soft-Actor Critic (SAC) algorithm to learn as fast as PPO in the context of a massively parallel simulator (thousands of robots simulated in parallel).
If you read along, you will learn how to automatically tune SAC for speed (i.e., minimize wall clock time), how to find better action boundaries, and what I tried that didn't work.-->

<!--## In the Previous Episode...

In the [first part](../sac-massive-sim/), I stopped at the point where we could detect some signs of life from SAC (it was learning something).-->

<!--However, SAC took more time to train than PPO (12 minutes vs. 6 minutes) and did not reach PPO's performance level.
Luckily, I still had several ideas for improving SAC's training speed and performance<d-footnote>I present the ones that didn't work and could use help (open problems) at the end of this post.</d-footnote>.-->

## Defining Proper Action Bound - Extracting the Limits with PPO

First, let's define the action space more precisely.
Correctly defining the boundaries of the action space is important for both the convergence speed and the final performance.
A larger action space gives the agent more flexibility, which can lead to better performance, but slower learning.
Conversely, a smaller action space can accelerate learning, though it may result in suboptimal solutions.

Thus, rather than simply restricting the action space to a small percentage of the original, I [recorded](#appendix-plot-action-distribution) the actions taken by a trained PPO agent and took the 2.5th and 97.5th percentiles for the new limits.
In other words, the new action space contains 95% of the actions commanded by a trained PPO agent<d-footnote>I repeat the same process for any new environment where those boundaries would not work</d-footnote>:
```python
# np.percentile(actions, 2.5, axis=0)
low = np.array([-2.0, -0.4, -2.6, -1.3, -2.2, -1.9, -0.7, -0.4, -2.1, -2.4, -2.5, -1.7])
# np.percentile(actions, 97.5, axis=0)
high = np.array([1.1, 2.6, 0.7, 1.9, 1.3, 2.6, 3.4, 3.8, 3.4, 3.4, 1.9, 2.1])
```

## Need for Speed or: How I Learned to Stop Worrying About Sample Efficiency

The second aspect I can improve is the hyperparameters of the SAC algorithm.
The default hyperparameters of the SAC algorithm are optimized for sample efficiency.
While this is ideal for learning directly on a single real robot<d-cite key="haarnoja2018learning"></d-cite>, it is suboptimal for training thousands of robots in simulation.

[Previously](#quick-fix), I quickly tuned SAC by hand to get it up and running.
This was sufficient for obtaining initial results, but it would be very time-consuming to continue tuning manually to reach PPO's performance level.
That's why I turned to automatic hyperparameter [optimization](https://github.com/optuna/optuna).

<!--If you are not familiar with automatic hyperparameter tuning, I wrote two blog posts about it:
- [Automatic Hyperparameter Tuning - A Visual Guide (Part 1)](../hyperparam-tuning/)
- [Automatic Hyperparameter Tuning - In Practice (Part 2)](../optuna/) shows how to use the [Optuna library](https://github.com/optuna/optuna) to put these techniques into practice-->

### New Objective: Learn as Fast as Possible

Since I'm using a massively parallel simulator, I no longer care about how many samples are needed to learn something but how quickly it can learn, regardless of the number of samples used.
In practice, this translates to an objective function that looks like this:
```python
def objective(trial: optuna.Trial) -> float:
    """Optimize for best performance after 5 minutes of training."""
    # Sample hyperparameters
    hyperparams = sample_sac_params(trial)
    agent = sbx.SAC(env=env, **hyperparams)
    # Callback to exit the training loop after 5 minutes
    callback = TimeoutCallback(timeout=60 * 5)
    # Train with a max budget of 50_000_000 timesteps
    agent.learn(total_timesteps=int(5e7), callback=callback)
    # Log the number of interactions with the environments
    trial.set_user_attr("num_timesteps", agent.num_timesteps)
    # Evaluate the trained agent
    env.seed(args_cli.seed)
    mean_reward, std_reward = evaluate_policy(agent, env, n_eval_episodes=512)
    return mean_reward
```

The agent is evaluated after five minutes of training, regardless of how many interactions with the environment were needed (the `TimeoutCallback` forces the agent to exit the training loop).


### SAC Hyperparameters Sampler

Similar to PPO, many hyperparameters can be tuned for SAC.
After some trial and error, I came up with the following sampling function (I've included comments that explain the meaning of each parameter):
```python
def sample_sac_params(trial: optuna.Trial) -> dict[str, Any]:
    # Discount factor
    gamma = trial.suggest_float("gamma", 0.975, 0.995)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.002, log=True)
    # Initial exploration rate (entropy coefficient in the SAC loss)
    ent_coef_init = trial.suggest_float("ent_coef_init", 0.001, 0.02, log=True)
    # From 2^7=128 to 2^12 = 4096, the mini-batch size
    batch_size_pow = trial.suggest_int("batch_size_pow", 7, 12, log=True)
    # How big should should the actor and critic networks be
    # net_arch = trial.suggest_categorical("net_arch", ["default", "simba", "large"])
    # I'm using integers to be able to use CMA-ES,
    # "default" is [256, 256], "large" is [512, 256, 128]
    net_arch_complexity = trial.suggest_int("net_arch_complexity", 0, 3)
    # From 1 to 8 (how often should we update the networks, every train_freq steps in the env)
    train_freq_pow = trial.suggest_int("train_freq_pow", 0, 3)
    # From 1 to 1024 (how many gradient steps by step in the environment)
    gradient_steps_pow = trial.suggest_int("gradient_steps_pow", 0, 10)
    # From 1 to 32 (the policy delay parameter, similar to TD3 update)
    policy_delay_pow = trial.suggest_int("policy_delay_pow", 0, 5)
    # Polyak coeff (soft update of the target network)
    tau = trial.suggest_float("tau", 0.001, 0.05, log=True)

    # Display true values
    trial.set_user_attr("batch_size", 2**batch_size_pow)
    trial.set_user_attr("gradient_steps", 2**gradient_steps_pow)
    trial.set_user_attr("policy_delay", 2**policy_delay_pow)
    trial.set_user_attr("train_freq", 2**train_freq_pow)
    # Note: to_hyperparams() does the convertions between sampled value and expected value
    # Ex: converts batch_size_pow to batch_size
    # This is useful when replaying trials
    return to_hyperparams({
        "train_freq_pow": train_freq_pow,
        "gradient_steps_pow": gradient_steps_pow,
        "batch_size_pow": batch_size_pow,
        "tau": tau,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "policy_delay_pow": policy_delay_pow,
        "ent_coef_init": ent_coef_init,
        "net_arch_complexity": net_arch_complexity,
    })
```

### Replay Ratio

A metric that will be useful to understand the tuned hyperparameters is the replay ratio.
The replay ratio (also known as update-to-data ratio or UTD ratio) measures the number of gradient updates performed per environment interaction or experience collected.
This ratio represents how often an agent updates its parameters relative to how much new experience it gathers.
For SAC, it is defined as `replay_ratio = gradient_steps / (num_envs * train_freq)`.

In a classic setting, the replay ratio is usually greater than one when optimizing for sample efficiency.
That means that SAC does at least one gradient step per interaction with the environment.
However, since collecting new data is cheap in the current setting, the replay ratio tends to be lower than 1/4 (one gradient step for every four steps in the environment).

### Optimization Result - Tuned Hyperparameters

To optimize the hyperparameters, I used Optuna's CMA-ES sampler<d-cite key="takuya2019optuna"></d-cite> for 100 trials<d-footnote>Here, I only optimized for the Unitree A1 flat task due to limited computation power. It would be interesting to tune SAC directly for the "Rough" variant, including `n_steps` and gSDE train frequency as hyperparameters.</d-footnote> (taking about 10 hours with a population size of 10 individuals).
Afterward, I retrained the best trials to filter out any lucky seeds<d-cite key="raffin2022learning"></d-cite>, i.e., to find hyperparameters that work consistently across different runs.

This is what the optimization history looks like. Many sets of hyperparameters were successful:

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/optuna_sac.png" class="img-fluid" %}
<div class="caption">
    Hyperparameter optimization history
</div>

These are the tuned hyperparameters of SAC found by the CMA-ES sampler while optimizing for speed:
```yaml
batch_size: 512
buffer_size: 2_000_000
ent_coef: auto_0.009471776840423638
gamma: 0.983100250213744
gradient_steps: 32
learning_rate: 0.00044689099625712413
learning_starts: 2000
policy: MlpPolicy
policy_delay: 8
policy_kwargs:
  net_arch: [512, 256, 128]
  activation_fn: !!python/name:isaaclab_rl.sb3.elu ''
  optimizer_class: !!python/name:optax._src.alias.adamw ''
  layer_norm: true
tau: 0.0023055560568780655
train_freq: 1
```

Compared to the default hyperparameters of SAC, there are some notable changes:
- The network architecture is much larger (`[512, 256, 128]` vs. `[256, 256]`), but similar to that used by [PPO in Isaac Sim](https://github.com/isaac-sim/IsaacLab/blob/f52aa9802780e897c184684d1cbc2025fafcef4a/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/a1/agents/rsl_rl_ppo_cfg.py#L21).
- The lower replay ratio (RR ≈ 0.03 for 1024 environments, or three gradient steps every 100 steps in an environment) and higher policy delay (update the actor once every eight critic updates) make it faster, as less time is taken for gradient updates.
- The discount factor is lower than the default value of 0.99, which favors shorter-term rewards.

Here is the result in video and the associated learning curves<d-footnote>The results are plotted for only five independent runs (random seeds). This is usually insufficient for RL due to the stochasticity of the results. However, in this case, the results tend to be consistent between runs (limited variability). I observed this during the many runs I did while debugging (and writing this blog post), so the trend is likely correct, even with a limited number of seeds.</d-footnote>:

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/learning_curve_unitree.svg" class="img-fluid" %}
<div class="caption">
    Learning curve on the Unitree A1 task using 1024 envs.
</div>

{% include video.liquid path="https://b2drop.eudat.eu/public.php/dav/files/ATn25xMbccroaiQ/sac_unitree_a1_tuned.mp4"  class="img-fluid rounded z-depth-1" controls=true %}
<div class="caption">
    Trained SAC agent after automatic tuning.
</div>

With these tuned hyperparameters, SAC learns faster, achieves higher performance, and the learned gaits look better (no more feet in the air!).
What more could you ask for?


## Does it work? - More Environments

<!-- So far, I have only optimized and tested the hyperparameters in one environment. -->
<!-- The goal is to make it work in any locomotion environment. -->

After it successfully learned in the flat Unitree A1 environment, I tested the same hyperparameters (with the same recipe<d-footnote>I updated the limits for each family of robots. The PPO percentiles technique worked nicely.</d-footnote>) on the GO1, GO2, Anymal-B, and Anymal-C environments, as well as the flat [Disney BD-X](https://github.com/louislelay/disney_bdx_rl_isaaclab) environment, and ... it worked!

{% include video.liquid path="https://b2drop.eudat.eu/public.php/dav/files/ATn25xMbccroaiQ/isaac_part_two.mp4"  class="img-fluid rounded z-depth-1" controls=true %}
<div class="caption">
    Trained SAC agent in different environments, using the same tuned hyperparameters.
</div>


<!-- In those environments, SAC learns as fast as PPO but is more sample-efficient. -->

Then, I trained SAC on the "rough" locomotion environments, which are harder environments where the robot has to learn to navigate steps and uneven, accidented terrain (with additional randomization).
And ... it worked partially.

## Solving Harder Environments

### Identifying the problem: Why it doesn't work?

In the "Rough" environment, the SAC-trained agent exhibits inconsistent behavior.
For example, one time the robot successfully climbs down the pyramid steps without falling; at other times, however, it does nothing.
Additionally, no matter how long it is trained, SAC does not seem to be able to learn to solve the "inverted pyramid", which is probably one of the most challenging tasks:

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/inverted_pyramid.jpg" class="img-fluid" %}
<div class="caption">
    The inverted pyramid task.
</div>

I isolated this task by training SAC only on the inverted pyramid.
Upon further inspection, it appeared to be an exploration problem; that is, SAC never experiences successful stepping when executing random movements.
<!--This reminded me of SAC failing on the [mountain car problem](https://github.com/rail-berkeley/softlearning/issues/76) because the exploration was inconsistent (the default high-frequency noise is usually a [bad default](https://openreview.net/forum?id=TSuSGVkjuXd) for robots).-->
This reminded me of SAC failing on the [mountain car problem](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/) because the exploration was inconsistent (the default high-frequency noise is usually a bad default<d-cite key="raffin2021gsde"></d-cite> for robots).

### Improving Exploration and Performance

To test this hypothesis, I simplified the problem by [lowering the step](https://github.com/isaac-sim/IsaacLab/blob/f52aa9802780e897c184684d1cbc2025fafcef4a/source/isaaclab/isaaclab/terrains/config/rough.py#L32) of the inverted pyramid.
I also used a more consistent exploration scheme: generalized State-Dependent Exploration (gSDE)<d-cite key="raffin2021gsde"></d-cite>.
<!--(developed during my PhD to train RL directly on real robots).-->

In its simplest form, gSDE repeats the noise vector for $$n$$-steps, instead of sampling it at every timestep.
In other words, instead of selecting actions following $$a_t = \mu_\theta(s_t) + \epsilon_t$$<d-footnote>$$\mu_\theta(s_t)$$ is the actor network output, which represents the mean of the Gaussian distribution.</d-footnote> and sampling $$\epsilon_t \sim N(0, \sigma^2)$$ at every step during exploration, gSDE samples $$\epsilon \sim N(0, \sigma^2)$$ once and keeps $$\epsilon$$ constant for $$n$$-steps.
The robot could finally learn to partially solve this task with this improved exploration.
<!-- (note: gSDE also allowed to have better performance on the flat terrain, maybe my PhD was useful ^^?) -->
<!-- Not alone: https://github.com/younggyoseo/FastTD3/issues/26#issuecomment-3388978359-->

{% include video.liquid path="https://b2drop.eudat.eu/public.php/dav/files/ATn25xMbccroaiQ/sac_rough_anymal_c.mp4"  class="img-fluid rounded z-depth-1" controls=true %}
<div class="caption">
    Trained SAC agent with gSDE and n-step return in the "Rough" Anymal-C environment.
</div>

There was still a big gap in final performance between SAC and PPO.
<!--To close the gap, I drew inspiration from the recent FastTD3<d-cite key="seo2025fasttd3"></d-cite> paper and implemented [n-step returns](https://github.com/DLR-RM/stable-baselines3/pull/2144) for all off-policy algorithms in SB3.-->
To close the gap, I drew inspiration from the recent FastTD3<d-cite key="seo2025fasttd3"></d-cite> paper and implemented n-step returns.
Using `n_steps=3` allowed SAC to finally solve the hardest task<d-footnote>Although there is still a slight performance gap between SAC and PPO, after reading the FastTD3 paper and conducting my own experiments, I believe that the environment rewards were tuned for PPO to encourage a desired behavior. In other words, I suspect that the weighting of the reward terms was adjusted for PPO. To achieve similar performance, SAC probably needs different weights. However, this is beyond the scope of this already lengthy blog post.</d-footnote>!

In summary, here are the additional manual changes I made to the hyperparameters of SAC compared to those optimized automatically:
```yaml
# Note: we must use train_freq > 1 to enable gSDE
# which resamples the noise every n steps (here every 10 steps)
train_freq: 10
# Scaling the gradient steps accordingly, to keep the same replay ratio:
# 32 * train_freq = 320
gradient_steps: 320
use_sde: True
# N-step return
n_steps: 3
```

And here are the associated learning curves (plotting the current curriculum level on the y-axis<d-footnote>I'm plotting the current state of the terrain curriculum (the higher the number, the harder the task/terrain) as the reward magnitude doesn't tell the whole story for the "Rough" task.</d-footnote>):

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/learning_curve_rough.svg" class="img-fluid" %}
<div class="caption">
    Learning curve on the Anymal-C "Rough" task using 1024 envs (except for PPO).
</div>

{% include figure.liquid path="assets/img/2026-04-27-sac-massive-sim/learning_curve_rough_efficiency.svg" class="img-fluid" %}
<div class="caption">
    Learning curve in term of sample-effiency on the Anymal-C "Rough" task using 1024 envs (except for PPO).
</div>

In those plots, you can see the effect of gSDE and the use of n-step returns.
SAC is also much more sample-efficient than PPO.

<!-- Note: sde allow to have better performance without linear schedule -->

## Conclusion

This concludes the long journey I started a few months ago to make SAC work on a massively parallel simulator.
During this adventure, I addressed a common issue that prevents SAC-like algorithms from working in these environments: the use of an unbounded action space.

In the end, with a proper action space and tuned hyperparameters, SAC is now competitive with PPO<d-footnote>Although there is still a slight performance gap between SAC and PPO, after reading the FastTD3 paper and conducting my own experiments, I believe that the environment rewards were tuned for PPO to encourage a desired behavior. In other words, I suspect that the weighting of the reward terms was adjusted for PPO. To achieve similar performance, SAC probably needs different weights. However, this is beyond the scope of this already lengthy blog post.</d-footnote> in terms of training time (while being much more sample-efficient) on a large collection of locomotion environments.
I hope my voyage encourages others to use SAC in their experiments and unlock fine-tuning on real robots after pretraining in simulation.


## Appendix: Affected Papers/Code
Please find here a non-exhaustive list of papers/code affected by the large bound problem:
<!-- - [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground/blob/0f3adda84f2a2ab55e9d9aaf7311c917518ec25c/mujoco_playground/_src/locomotion/go1/joystick.py#L239) -->
<!-- https://github.com/Argo-Robot/quadrupeds_locomotion/blob/45eec904e72ff6bafe1d5378322962003aeff88d/src/go2_env.py#L173 -->
<!-- https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot.py#L85 -->
- [IsaacLab](https://github.com/isaac-sim/IsaacLab/blob/c4bec8fe01c2fd83a0a25da184494b37b3e3eb61/source/isaaclab_rl/isaaclab_rl/sb3.py#L154)
- [Learning to Walk in Minutes](https://github.com/leggedrobotics/legged_gym/blob/17847702f90d8227cd31cce9c920aa53a739a09a/legged_gym/envs/base/legged_robot_config.py#L164 )
- [One Policy to Run Them All](https://github.com/nico-bohlinger/one_policy_to_run_them_all/blob/d9d166c348496c9665dd3ebabc20efb6d8077161/one_policy_to_run_them_all/environments/unitree_a1/environment.py#L140)
- [Genesis env](https://github.com/Argo-Robot/quadrupeds_locomotion/blob/45eec904e72ff6bafe1d5378322962003aeff88d/src/go2_train.py#L104)
- [ASAP Humanoid](https://github.com/LeCAR-Lab/ASAP/blob/c78664b6d2574f62bd2287e4b54b4f8c2a0a47a5/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml#L161)
- [Agile But Robust](https://github.com/LeCAR-Lab/ABS/blob/9b95329ffb823c15dead02be620ff96938e4d0a3/training/legged_gym/legged_gym/envs/base/legged_robot_config.py#L169)
- [Rapid Locomotion](https://github.com/Improbable-AI/rapid-locomotion-rl/blob/f5143ef940e934849c00284e34caf164d6ce7b6e/mini_gym/envs/base/legged_robot_config.py#L209)
- [Deep Whole Body Control](https://github.com/MarkFzp/Deep-Whole-Body-Control/blob/8159e4ed8695b2d3f62a40d2ab8d88205ac5021a/legged_gym/legged_gym/envs/widowGo1/widowGo1_config.py#L114)
- [Robot Parkour Learning](https://github.com/ZiwenZhuang/parkour/blob/789e83c40b95fdd49fda7c1725c8c573df42d2a9/legged_gym/legged_gym/envs/base/legged_robot_config.py#L169)

You can probably find many more by looking at [works that cite the ETH paper](https://scholar.google.com/scholar?cites=8503164023891275626&as_sdt=2005&sciodt=0,5).

- Seems to be fixed in [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour/blob/d2ffe27ba59a3229fad22a9fc94c38010bb1f519/legged_gym/legged_gym/envs/base/legged_robot_config.py#L120) (clip action 1.2)
- Almost fixed in [Walk this way](https://github.com/Improbable-AI/walk-these-ways/blob/0e7236bdc81ce855cbe3d70345a7899452bdeb1c/scripts/train.py#L200) (clip action 10)

<!--
Related:
- [Parallel Q Learning (PQL)](https://github.com/Improbable-AI/pql) but only tackles classic MuJoCo locomotion envs -->

## Appendix: Note on Unbounded Action Spaces

<!--While discussing this blog post with [Nico Bohlinger](https://github.com/nico-bohlinger), he raised another point that could explain why people might choose an unbounded action space.-->

While discussing this blog post with a fellow researcher, they raised another point that could explain why people might choose an unbounded action space.

In short, policies can learn to produce actions outside the joint limits to trick the underlying [PD controller](https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller) into outputting desired torques.
For example, when recovering from a strong push, what matters is not to accurately track a desired position, but to quickly move the joints in the right direction.
This makes training almost invariant to the chosen PD gains.

<!--<details>
  <summary>Full quote</summary>

  So in theory you could clip the actor output to the min and max ranges of the joints, but what happens quite often is that these policies learn to produce actions that sets the target joint position outside of the joint limits.

  This happens because the policies don't care about the tracking accuracy of the underlying <a href="https://en.wikipedia.org/wiki/Proportional%E2%80%93integral%E2%80%93derivative_controller">PD controller</a>, they just want to command: in which direction should the joint angle change, and by how much.

  In control, the magnitude is done through the P and D gains, but we fix them during training, so when the policy wants to move the joints in a certain direction very quickly (especially needed during recovery of strong pushes or strong domain randomization in general), it learns to command actions that are far away to move into this direction quickly, i.e. to produce more torque.

  It essentially learns to trick the PD control to output whatever torques it needs. Of course, this also depends on the PD gains you set; if they are well chosen, actions outside of the joint limits are less frequent. A big benefit is that this makes the whole training pipeline quite invariant to the PD gains you choose at the start, which makes tuning easier.
</details>-->


## Appendix: What I Tried That Didn't Work

While preparing this blog post, I tried many things to achieve PPO performance and learn good policies in minimal time.
Many of the things I tried didn't work, but they are probably worth investigating further.
I hope you can learn from my failures, too.

### Using an Unbounded Gaussian Distribution

One approach I tried was to make SAC look more like PPO.
In part one, PPO could handle an unbounded action space because it used a (non-squashed) Gaussian distribution (vs. a squashed one for SAC).
However, replacing SAC's squashed Normal distribution with an unbounded Gaussian distribution led to additional problems.

Without layer normalization in the critic, it quickly diverges (leading to Inf/NaN).
It seems that, encouraged by the entropy bonus, the actor pushes toward very large action values.
It also appears that this variant requires specific tuning (and that state-dependent std may need to be replaced with state-independent std, as is done for PPO).

If you manage to reliably make SAC work with an unbounded Gaussian distribution, please reach out!

<!-- Note: tried with both state-dependent std and independent std -->

<!-- TODO: try with fixed std? more tuning, tune notably the target entropy, any other mechanism to avoid explosion of losses/divergence? -->

### KL Divergence Adaptive Learning Rate

One component of PPO that allows for better performance is the learning rate schedule (although it is not critical, it eases hyperparameter tuning).
It automatically adjusts the learning rate to maintain a constant KL divergence between two updates, ensuring that the new policy remains close to the previous one (and ensuring that the learning rate is large enough, too).
It should be possible to do something similar with SAC.
However, when I tried to approximate the KL divergence using either the log probability or the extracted Gaussian parameters (mean and standard deviation), it didn't work.
The KL divergence values were too large and inconsistent.
SAC would probably need a trust region mechanism as well.

Again, if you find a way to make it work, please reach out!

### En Vrac - Other Things I Tried

- penalty to be away from action bounds (hard to tune)
- action space schedule (start with a small action space, make it bigger over time, tricky to schedule, and didn't improve performance)
- linear schedule (`learning_rate = LinearSchedule(start=5e-4, end=1e-5, end_fraction=0.15)`), it helped for convergence when using `n_steps=1` and `use_sde=False`, but was not needed at the end


## Appendix: Plot Action Distribution

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecEnvWrapper

sns.set_theme()


class PlotActionVecEnvWrapper(VecEnvWrapper):
    """
    VecEnv wrapper for plotting the taken actions.
    """

    def __init__(self, venv, plot_freq: int = 10_000):
        super().__init__(venv)
        # Action buffer
        assert isinstance(self.action_space, spaces.Box)
        self.n_actions = self.action_space.shape[0]
        self.actions = np.zeros((plot_freq, self.num_envs, self.n_actions))
        self.n_steps = 0
        self.plot_freq = plot_freq

    def reset(self):
        return self.venv.reset()

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return obs, rewards, dones, infos

    def step_async(self, actions):
        self.actions[self.n_steps % self.plot_freq] = actions
        self.n_steps += 1
        if self.n_steps % self.plot_freq == 0:
            self.plot()
        self.venv.step_async(actions)

    def plot(self) -> None:
        # Flatten the env dimension
        actions = self.actions.reshape(-1, self.n_actions)
        n_steps = self.num_envs * self.n_steps
        # Create a figure with subplots for each action dimension
        n_rows = min(2, self.n_actions // 2 + 1)
        n_cols = max(self.n_actions // 2, 1)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 10))
        fig.suptitle(
            f"Distribution of Actions per Dimension after {n_steps} steps", fontsize=16
        )

        # Flatten the axes array for easy iteration
        if n_rows > 1:
            axes = axes.flatten()
        else:
            # Special case, n_actions == 1
            axes = [axes]

        # Plot the distribution for each action dimension
        for i in range(self.n_actions):
            sns.histplot(actions[:, i], kde=True, ax=axes[i], stat="density")
            axes[i].set_title(f"Action Dimension {i+1}")
            axes[i].set_xlabel("Action Value")
            axes[i].set_ylabel("Density")

        # Adjust the layout and display the plot
        plt.tight_layout()
        plt.show()


vec_env = make_vec_env("Pendulum-v1", n_envs=2)
wrapped_env = PlotActionVecEnvWrapper(vec_env, plot_freq=5_000)

# from sbx import PPO
# from sbx import SAC
# policy_kwargs = dict(log_std_init=-0.5)

model = PPO("MlpPolicy", wrapped_env, gamma=0.98, verbose=1)
model.learn(total_timesteps=1_000_000)
```


<!--## Acknowledgement

I would like to thank Anssi, Leon, Ria and Costa for their feedback =).-->

<!-- All the graphics were made using [excalidraw](https://excalidraw.com/). -->
