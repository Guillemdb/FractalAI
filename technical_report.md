## Table of Contents

**This project is still under active development.**

1. [Introduction](#introduction)
2. [How it works](#how-it-works)
3. [Definitions](#definitions)
    1. [Causal cone](#causal-cone)
    2. [Swarm](#swarm)
    3. [Virtual reward](#virtual-reward)
    4. [Dead](#cloning)
    5. [Distance](#distance)
    6. [Normalization](#normalization)
4. [Pseudo code](#pseudo-code)
5. [Architecture](#architecture)
6. [Comparison against MCTS](#installation)
7. [Benchmarks](#benchmarks)
    1. [Other Monte carlo methods](#other-monte-carlo-methods)
    2. [Current state of the art](#current-state-of-the-art)

8. [Bibliography](#bibliography)


# Introduction

Fractal Monte Carlo (FMC) is a new model-free reinforcement learning (MFRL)[#Citation Needed]()
agent derived from first principles of the Fractal AI theory[#CN](). FMC calculates an approximation
to a probability distribution over the action space that acts as an utility function. In this
report, we will show how our method outperforms equivalent approaches based on discounting expected
rewards[#CN]() to construct a policy.
   
FMC has been extensively tested in a wide range of environments, in both continuous and discrete 
action spaces. This report will focus on the performance
of the Agent in Atari games, as provided by the Python library OpenAI Gym[#CN](), under heavily
bounded computational resources. The agent presented outperforms the current state of the art
and shows to be significantly more efficient than similar methods, using either pixels or RAM as input.

# How it works

FMC uses a [Swarm](#swarm) to construct a [causal cone](#causal-cone) of possible future
trajectories of the agent through the state space. The leaf nodes in this cone, will be used
to estimate the the utility value distribution over actions. This utility distribution will be used
as a policy, and the highest utility action will be chosen.

The causal cone is calculated using an iterative algorithm that makes random perturbations
to a group of swarm. FMC intends to distribute the swarm of the Swarm matching de reward density
distribution of the state space sampled.

This is accomplished thanks to the following properties of the algorithm:

- FMC uses an scalar called [virtual reward](#virtual-reward) to weight both the physical
distribution of the Swarm across the state space, and the reward of the different swarm.

- The algorithm recycles useless trajectories in the cone by [cloning](#cloning) any state of the
Swarm to another randomly chosen state. An state will clone or not depending on the relationship 
of virtual rewards between itself and the chosen state.

- After calculating an action, the algorithm updates most of its internal parameters to adjust
itself dynamically to changes in the environment. The corrections to the parameters needed,
are calculated based on how asymmetric the shape of the swarm is. The asymmetry is measured with
respect to the spatial distribution of the swarm, their reward distribution, and their distribution
across time.

- It can discard arbitrary large sections of the search space by manually adding boundary conditions.
These boundary conditions are modeled defining a [dead flag]() for each state, that allows to modify the
cloning probability.

- FMC avoids getting stuck in local optima thanks to weighting not only the reward, but also the
spatial distribution of the states in the swarm.

- The algorithm is parallelizable and could scale almost linearly with the maximum number of
samples allowed. All the internal operations (distances, virtual reward calculations, cloning,
etc...) involve only one state, or a given state and another randomly chosen state. 
They all have a computational complexity that is lower than quadratic, although the specific
scalability will depend on the implementation.

- It is tolerant to rewards and distances that are unbounded, or vary widely in scale thanks to a 
[normalization]() process that takes place before the virtual reward is calculated.


# Definitions
In this section we define the different concepts that are required for understanding the agent. When
available, an indication of the line in where the concept is used in the file [fractalai.py]() is provided.


##### Causal cone
 It represents the graph formed by all the paths traversed by the swarm of the
 swarm during the evolutionary process that they undergo. All the paths share a common origin and
 form a graph with a tree like structure. At any moment, the swarm of the swarm will be distributed
 across the leaf nodes of the tree.

##### Swarm
It is the mathematical structure that represents a group of swarm that undergo an
evolution process through random perturbations. [See lines 58, 105]()

##### Virtual reward
 Is an scalar number that represents the instantaneous utility of a given
state during the evolution of the swarm. [Defined at _virtual_reward (Line 175)]()

##### Cloning
 It is an operation that allows us to move any state of the swarm from one leaf of the
cone to another. This allows to make the cone grow efficiently by redistributing
the swarm of the swarm across the state space, assigning more computational resources to areas where
the state space has higher rewards, and to areas that are significantly different from the others
scanned by the swarm. [Defined at _clone() (L197)]()

##### Dead
 This is an arbitrary condition set by the programmer that affects the clone probability.
Given to swarm, **A** and another state chosen at random **B**: 

  - If **B** is dead (meets the condition) -> the probability of **A** cloning **B** to is 0.
  - If **A** is dead and **B** not dead -> **A** will clone to **B** with probability 1.
 
 This mechanism allows for efficient scanning when facing boundary conditions.
 [See lines 208, 220, 221]()
    
##### Distance
 This is an informative measure of how different the observations of two different
swarm are. We call it distance because in this specific implementation we are using the euclidean
distance between two observations. At each iteration of the algorithm, each state will calculate its distance
to another state chosen at random. [Defined at L162 fractalai.py]()


##### Normalization
It is a transformation applied to both the distances and the rewards, which is applied before
calculating the virtual reward. It consists in two steps:

  - Normalizing the distances of all the states so they fall in the range [0,1]. The lowest distance
will be assigned a value of 0 and the highest a value of one. [L173]()

  - Normalizing the rewards to be in the range [1, 2]. [L182]()
