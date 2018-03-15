## Table of Contents

**This project is still under active development.**

1. [Abstract](#abstract)
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
    2. [Current state of the art](#other-monte-carlo-methods)

8. [Bibliography](#bibliography)


# Abstract

Fractal Monte Carlo (FMC) is a new model-free reinforcement learning (MFRL)[#Citation Needed]()
agent derived from first principles of the Fractal AI theory[#CN](). FMC calculates an approximation
to a probability distribution over the action space that acts as an utility function. In this
report, we will show how our method outperforms equivalent approaches based on discounting expected
rewards[#CN]().
   
FMC has been extensively tested in a wide range of environments, in both continuous and discrete 
action spaces, and using either pixels or RAM as input. This report will focus on the performance
of the Agent in Atari games, as provided by the Python library OpenAI Gym[#CN](), under heavily
bounded computational resources. The agent presented outperforms the current state of the art
and shows to be significantly more efficient than similar methods.

# How it works

We use a cellular automaton-inspired mathematical structure called Swarm to construct a Direct
Acyclic Graph (DAG) of possible future trajectories of the agent through the state space. This leaf
node in this DAG, (or causal cone, as called in [#CN]()) will be used to estimate the the utility 
value distribution over actions.



# Definitions
