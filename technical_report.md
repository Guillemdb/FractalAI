## Table of Contents

**This project is still under active development.**

1. [Introduction](#introduction)
2. [How it works](#how-it-works)
3. [Definitions](#definitions)
    1. [Reinforcement Learning](#inherited-from-rl)
        1.  [Environment, actions, and rewards](#environment)
        3.  [State and State space](#state-and-state-space)
        2.  [Q value](#q-value)
    2. [Parameters](#parameters)
    3. [Operators](#operators)
    4. [Internal variables](#internal-variables)
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

FMC uses a [Swarm](#swarm) to construct a Direct Acyclic Graph (DAG) of possible future
trajectories of the agent through the state space. The leaf nodes in this DAG, will be used
to estimate the the utility value distribution over actions. This utility distribution will be used
as a policy, and the highest utility action will be chosen.

The DAG is calculated using an iterative algorithm that makes random perturbations to a group of
states. FMC intends to distribute the states of the Swarm matching de reward density
distribution of the state space sampled.

This is accomplished thanks to the following properties of the algorithm:

- FMC uses an scalar called [entropic reward](#entropic-reward) to weight both the physical
distribution of the Swarm across the state space, and the reward of the different states.

- The algorithm recycles useless trajectories in the DAG by [cloning](#cloning) any state of the
Swarm to another randomly chosen state. An state will clone or not depending on the relationship 
of entropic rewards between itself and the chosen state.

- After calculating an action, the algorithm updates most of its internal parameters to adjust
itself dynamically to changes in the environment. The corrections to the parameters needed,
are calculated based on how asymmetric the shape of the swarm is. The asymmetry is measured with
respect to the spatial distribution of the states, their reward distribution, and their distribution
across time.

- It can discard arbitrary large sections of the search space by manually adding boundary conditions.
These boundary conditions are modeled defining a [dead flag]() for each state, that allows to modify the
cloning probability.

- FMC avoids getting stuck in local optima thanks to weighting not only the reward, but also the
spatial distribution of the states in the swarm.

- The algorithm is parallelizable and could scale almost linearly with the maximum number of
samples allowed. All the internal operations (distances, entropic reward calculations, cloning,
etc...) involve only one state, or a given state and another randomly chosen state. 
They all have a computational complexity that is lower than quadratic, although the specific
scalability will depend on the implementation.

- It is tolerant to rewards and distances that are unbounded, or vary widely in scale thanks to a 
[renormalization process]() that takes place before the entropic reward is calculated.


# Definitions

- **Swarm**: 

- **Entropic reward**:

- **Cloning**:

- **Dead**:

- **Distance**: