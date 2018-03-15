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
4. [Parameters](#pseudo-code)
4. [Pseudo code](#pseudo-code)
5. [Architecture](#architecture)
6. [Comparison against MCTS](#installation)
7. [Benchmarks](#benchmarks)
    1. [Other Monte carlo methods](#other-monte-carlo-methods)
    2. [Current state of the art](#current-state-of-the-art)

8. [Bibliography](#bibliography)


# Introduction

Fractal Monte Carlo (FMC) is a new model-free RL agent derived from first principles based upon the
FAI theory. FMC calculates a utility function that approximates the probability distribution over
the potential action space. And builds a policy by discounting expected rewards. This report explains how.
   
FMC has been extensively tested in a wide range of environments, in both continuous and discrete 
action spaces. This report will focus on the performance
of the Agent in Atari games, as provided by the Python library OpenAI Gym[#CN](), under heavily
bounded computational resources. The agent presented outperforms the current state of the art on 
14/24 Atari games tested, and shows to be significantly more efficient than similar methods,
using either pixels or RAM as input.

# How it works

*We provide links to the specific lines of the code where the described parts of the algorithm take
place. For example, [(L45)]() will be a reference to the line 45 of the file 
[fractalmc.py](fractalai/fractalmc.py)*

When calculating an action, FMC will construct a tree that consists of potential trajectories that
describe the future evolution of the system. This tree, called causal cone, is expanded by a
swarm [(L58, 105)]() of walkers that populates its leaf nodes. 
The swarm will undergo an iterative process[(L322)]() in order to make the tree grow efficiently.
When a maximum amount of computation has been reached, the utility of an action will be considered
proportional to the number of walkers that populate leaf nodes originating
from the same action[(120, 126)]().

The causal cone, unlike MCTS is not a static tree of all possible actions that will be explored.
Instead the causal cone is a tree data structure that changes at every time step by applying random
perturbations[(L142)](), and letting the swarm move freely among different leaf nodes of the tree.
[(L199)]()

In order to evolve the swarm, first initialize the walkers at the root state, perturb them and store
the action chosen[(L136)](). Then use the following algorithm to make it evolve until the maximum
number of samples allowed is reached:

1. Measure the euclidean distance between all the observations of all the walkers, and the observation of
another walker chosen at random.[(L164)]() This will create an stochastic measure of diversity, that
when incorporated into the virtual reward formula, will favor the diversity among the states in
the swarm.

2. Normalize the values so all the walkers' distances fall into the [0, 1] range.[(L175)](). 
Normalize the rewards tobe in range [1, 2]. [(L184)](). This allows us to get rid of problems with
the scale of both distances and rewards, and assures that the value of the virtual distance will be
bounded.

3. Calculate the virtual reward for each walker. [(L177)](). This value represents an stochastic
measure of the importance of a given walker with respect to the whole swarm. It combines both an
exploration term (distance), with an exploitation term reward that is weighted by the balance
coefficient, which represents the current trade-off between exploration and exploitation, and helps
modeling risk.[(L304)]()

4. Each walker of the swarm compares itself to another walker chosen at random[(L215)](),
 and gets assigned a probability of moving to the leaf node where the other walker is located[(L219)]().
 
5. Determine if a walker is dead[(L209-212](). Then decide if the walker will clone or not
depending on its death condition and clone probability.[(L220-223]()

6. Move the walkers that are cloning to the target leaf node. [(L224-228]()

6. Perturb the walkers that did not clone. This will make the cone evolve.[(L142)]()

7. GOTO 1. until the maximum number of samples is reached.

8. Approximate the utility for each action according to [(120, 126)](). Take the action with more
utility.

After deciding an action, the swarm will update its parameters: the number of walkers, and the
number of times it will sample the state space to build the next causal cone. This update will be
adjusted by a non linear feedback loop, with the objective of keeping the mean depth of the cone
as close as possible to a desired time horizon.[(L262, 280, 290]()


# Definitions
This can be used as a reference section for the concepts in which the algorithm is based. When
available,indication of the line in where the concept is used in the file [fractalai.py]() is provided.


##### Causal cone
 It represents the graph formed by all the paths traversed by the states of the
 swarm during the evolutionary process that they undergo. All the paths share a common origin and
 form a graph with a tree like structure. At any moment, the states of the swarm will be distributed
 across the leaf nodes of the tree.

##### Swarm
It is the mathematical structure that represents a group of states that undergo an
evolution process through random perturbations. [See lines 58, 105]()

##### Virtual reward
 Is an scalar number that represents the instantaneous utility of a given
state during the evolution of the swarm. [Line 175]()

##### Cloning
 It is an operation that allows us to move any state of the swarm from one leaf of the
cone to another. This allows to make the cone grow efficiently by redistributing
the states of the swarm across the state space, assigning more computational resources to areas where
the state space has higher rewards, and to areas that are significantly different from the others
scanned by the swarm. [Defined at _clone() (L197)]()

##### Dead
 This is an arbitrary condition set by the programmer that affects the clone probability.
Given two states, **A** and another state chosen at random **B**: 

  - If **B** is dead (meets the condition) -> the probability of **A** cloning **B** to is 0.
  - If **A** is dead and **B** not dead -> **A** will clone to **B** with probability 1.
 
 This mechanism allows for efficient scanning when facing boundary conditions. In Atari games, the
 dead condition is triggered when the Agent loses one life or it gets a cumulative reward in a leaf
 of the cone, which is lower than the reward at the root node of the cone. 
 [See lines 208, 220, 221]()
    
##### Distance
 This is an informative measure of how different the observations of two different
states are. We call it distance because in this specific implementation we are using the euclidean
distance between two observations. At each iteration of the algorithm, each state will calculate its distance
to another state chosen at random. [Defined at L162 fractalai.py]()


##### Normalization
It is a transformation applied to both the distances and the rewards, which is applied before
calculating the virtual reward. It consists in two steps:

  - Normalizing the distances of all the states so they fall in the range [0,1]. The lowest distance
will be assigned a value of 0 and the highest a value of one. [L173]()

  - Normalizing the rewards to be in the range [1, 2]. [L182]()


# Parameters

### Set by the programmer

- **Fixed steps**: It is the number of consecutive times that will will apply an action to the
environment when we perturb it choosing an action. Although this parameter actually depends on the
Environment, we can use it to manually set the frequency at which the Agent will play. Taking more
consecutive actions allows for exploring further in the future at the cost of less reaction time.

- **Time Horizon**: This value represents "how far we need to look into the future when taking an
action". A useful rule of thumb is **Time Horiozon = Nt / Fixed steps**, where **Nt** is the number
of frames that it takes the agent to loose one life, (die) since the moment it performs the actions
that inevitably lead to its death. This parameters determines the time horizon of the bigger
potential well that the Agent should be able to escape.

- **Max states**: This is the maximum number of states that can be part of the Swarm. This number
is related to "how thick" we want the resulting causal cone to be. The algorithm will try to use
the maximum number of states possible unless it detects it is wasting computation.

- **Max samples**: This is the maximum number of times that we can sample an action when using
a Swarm to build a causal cone. It is a superior bound, and the algorithm will try to use less
samples to meet the defined **time horizon**. It is a nice way to limit how fast you need to
take an action. A reasonable value could be **max walkers** \* **time horizon** \* ***N***,
being ***N=5*** a number that works well in Atari games, but it depends on what you are trying
to accomplish.

### Internal parameters

You cannot directly set these parameters, but they .

- **Mean samples**:  It is the mean number of samples
taken each state. In and ideal case it would be **Max states \* Time horizon** samples. If its
lower it means that we are not having any trouble sampling the state space, but if its higher it
means that the states tend to die and more computation has been be required. 

- **Balance**:

- **Swarm size**