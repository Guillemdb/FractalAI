## Table of Contents

**This project is still under active development.**

1. [Introduction](#introduction)
2. [How it works](#how-it-works)
4. [Parameters](#parameters)
5. [Architecture](#architecture)
6. [Comparison against MCTS](#installation)
7. [Benchmarks](#benchmarks)
    1. [Other Monte carlo methods](#other-monte-carlo-methods)
    2. [Current state of the art](#current-state-of-the-art)

8. [Bibliography](#bibliography)


# Introduction

Fractal Monte Carlo (FMC) is a new model-free RL agent derived from first principles based upon the
Fractal AI (FAI) theory. FMC calculates a utility function that approximates the probability distribution over
the potential action space, and builds a policy by discounting expected rewards. This report explains how.
   
FMC has been extensively tested in a wide range of environments, in both continuous and discrete 
action spaces. This report will focus on the performance
of the Agent in Atari-2600 games, as provided by the Python library [OpenAI Gym](https://arxiv.org/pdf/1606.01540.pdf), under heavily
bounded computational resources. The agent presented outperforms the current state of the art on 
14/24 Atari games tested, and shows to be significantly more efficient than similar methods,
using either pixels or RAM as input.

# How it works

*We provide links to the specific lines of the code where the described parts of the algorithm take
place. For example, [(L45)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L45) will be a reference to the line 45 of the file 
[fractalmc.py](fractalai/fractalmc.py)*

When calculating an action, FMC will construct a tree that consists of potential trajectories that
describe the future evolution of the system. This tree, called causal cone, is expanded by a
swarm [(L58, 105)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L58-L105) of walkers that populates its leaf nodes. 
The swarm will undergo an iterative process [(L322)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L322) in order to make the tree grow efficiently.
When a maximum amount of computation has been reached, the utility of each action will be considered
proportional to the number of walkers that populate leaf nodes originating
from the same action [(120, 126)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L120-L126).

The causal cone, unlike in MCTS, is not a static tree of all possible actions that will be explored.
Instead the causal cone is a tree data structure that changes at every time step by applying random
perturbations [(L142)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L142), and letting the swarm move freely among different leaf nodes of the tree.
[(L199)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L199)

In order to evolve the swarm, we first initialize the walkers at the root state, perturb them, and store
the action chosen [(L136)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L136). Then use the following algorithm to make it evolve until the maximum number of samples allowed is reached:

1. Measure the euclidean distance between all the observations of all the walkers, and the observation of
another walker chosen at random [(L164)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L164). This will create an stochastic measure of diversity, that when incorporated into the virtual reward formula, will favor the diversity among the states in
the swarm.

2. Normalize the values so all the walkers' distances fall into the [0, 1] range [(L175)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L175). 
Normalize the rewards to be in range [1, 2]. [(L184)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L184). This allows us to get rid of problems with
the scale of both distances and rewards, and assures that the value of the virtual distance will be bounded.

3. Calculate the virtual reward for each walker. [(L177)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L177). This value represents an stochastic
measure of the importance of a given walker with respect to the whole swarm. It combines both an
exploration term (distance) with an exploitation term (reward) that is weighted by the balance
coefficient, which represents the current trade-off between exploration and exploitation, and helps
modeling risk [(L304)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L304).

4. Each walker of the swarm compares itself to another walker chosen at random [(L215)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L215), and gets assigned a probability of moving to the leaf node where the other walker is located [(L219)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L219).
 
5. Determine if a walker is dead [(L209-212)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L209-L212). Then decide if the walker will clone or not
depending on its death condition and clone probability [(L220-223)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L220-L223). The death condition is a flag set
by the programmer that lets us incorporate arbitrary boundary conditions to the behaviour of the agent.
The death flag helps the swarm avoiding undesired regions of the state space.

6. Move the walkers that are cloning to theirs target leaf nodes [(L224-228)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L224-L228). This allows for
recycling the walkers that either fall out of the desired domain (dead) or have been poorly valued
with respect to the whole swarm. It also partially avoids exploring regions of the state space that are either
too crowded (low diversity) or have a very poor reward.

7. Choose an action for each walker and step the environment (perturbation).
The swarm will evolve and explore new states. This is how you make the causal cone grow [(L142)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L142).
The fact that we are choosing between cloning and exploring allows for a non-uniform growth of the 
causal cone's time horizon. A walker can clone to a leaf node which has a different
depth than its current leaf node, meaning that jumps forward and backwards in time are allowed.

8. GOTO 1 until the maximum number of samples is reached. By iterating each time, we are
redistributing the "useless" walkers to more promising leaf nodes, and perturbing the states located
in the regions considered to have the highest utility. After several iterations, the density
distribution of the walkers should match the reward density distribution of the state space.

9. Approximate the utility for each action according to [(120, 126)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L120-L126). Take the action with more
utility. Note that we are just counting how many states in the swarm took the same action in the
root node.

After deciding an action, the swarm will update its parameters: the number of walkers, and the
number of times it will sample the state space to build the next causal cone. This update will be
adjusted by a non linear feedback loop, with the objective of keeping the mean depth of the cone
as close as possible to a desired time horizon [(L262, 280, 290)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L262-L290).

# Parameters

### Set by the programmer

- **Fixed steps**: It is the number of consecutive times that we will apply an action to the
environment when we perturb it choosing an action. Although this parameter actually depends on the
environment, we can use it to manually set the frequency at which the agent will play. Taking more
consecutive fixed steps per action allows for exploring further into the future at the cost of longer reaction times.

- **Time Horizon**: This value represents how far we need to look into the future when taking an
action. A useful rule of thumb is **Time Horiozon = Nt / Fixed steps**, where **Nt** is the number
of frames that it takes the agent to loose one life (die) since the moment it performs the actions
that inevitably lead to its death. This parameters, multiplied by the fixed_steps, determines the time horizon of the bigger
potential well that the agent should be able to escape.

- **Max states**: This is the maximum number of walkers that can be part of the Swarm. This number
is related to "how thick" we want the resulting causal cone to be. The algorithm will try to use
the maximum number of walkers possible. 

- **Max samples**: This is the maximum number of times that we can make a perturbation when using
a swarm to build a causal cone. It is a superior bound, the algorithm will try to use as few
samples as possible in order to meet the defined **time horizon**. It is a nice way to set how fast you need to
take an action in the worst case. A reasonable value is **max walkers** \* **time horizon** \* ***N***,
being ***N=5*** a number that works well in Atari games, but highly depends on the task.
