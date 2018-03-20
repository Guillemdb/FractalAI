# Introduction to Fractal AI theory

This is the fastest way to get an idea of what our work is about. Here you will find qualitative
explanations of the inner workings of our techniques, and possible applications to real world
problems.

## Table of Contents

- [What is FAI?](#what-is-fai-about)
- [Swarm Wave](#swarm-wave)
- [Fractal Monte Carlo](#fractal-monte-carlo)
- [Domain of application](#domain-of-aplication)
- [Combining FAI and RL](#comining-fai-and-rl)
- [Using FAI to train Robots](#bibliography)
- [Black box optimization](#black-box-optimization)



## What is FAI?

[Fractal AI](https://arxiv.org/pdf/1803.05049.pdf) is a theory derived from first principles that
allows to control swarms of walkers. It is a set of rules to move, compare, and extract information
from a swarm. Deriving mathematical tools from FAI to solve a specific problem is about defining
how you want the swarm to behave in a given state space, in a way that the information extracted
from it serves a purpose.

In the context of FAI, Fractal Monte Carlo is just one of the possible tools that can be derived to
solve specific problems. The Atari games, where we have perfect information, are just the simplest
case to test the algorithm, but it is not the general case.

We believe that swarms with different behaviours could be used as an alternative to traditional
calculus techniques in many other problems. How to derive new tools from FAI that apply to real
world problems would become a research topic, but some of the tools already derived suggest that
FAI could compete efficiently with alternative techniques. 


These short videos show some of our tools represented in a similar fashion that the
[Deep-Neuroevolution](https://eng.uber.com/deep-neuroevolution/) blog post from Uber.
We show swarms serving different purposes such as: 

* Forming a tree like structure that approximates a
[long-term energy gradient](https://youtu.be/eLGTo0RfFi4?t=29s).
* Exploring the space to find [shortest paths](https://youtu.be/AoiGseO7g1I?t=44s).
* Sampling chaotic state spaces [Video 1](https://youtu.be/HLbThk624jI?t=30s)
[Video 2](https://youtu.be/OFhBKZ0l6fw?t=1m19s). 
* Behaving like a [wave function](https://youtu.be/PdyfWIlLTCs?t=10s) that avoids boundary
conditions, that can also be used to draw a tree of trajectories.
* Exploring different scales of an energy landscape in a
[global optimization](http://entropicai.blogspot.com.es/2016/02/serious-fractal-optimizing.html?m=0)
problem.
* Solving an stochastic integral to [find a path](https://youtu.be/0t7jI9WdTWI) that discounts
expectations over 1000 time steps.

## Swarm Wave

## Fractal Monte Carlo

FMC applied to Atari is a toy example, however since we applied FMC in the context of Markov
decision processes, we do not actually need a perfect model. According to FAI, FMC is an algorithm
to efficiently explore functions that contain information on the input-output relation for a
system, given a bound in computational resources. The swarm that FMC uses is not meant to be
applied only when we have access to perfect models, but also in any mathematical function that
does not require to dynamically adjust the step size(dt) when being integrated. 

In this case we showed that it works on the ALE interface, which can provide either a deterministic
or an stochastic environment, but it could be possible to alter the environment that the swarm uses
by adding noise, (to observations, actions, rewards, distance measure, or boundary conditions) and
FMC will still perform nicely. Unfortunately, when not assuming a perfect model we experience a
penalty in performance, although FMC is capable of successfully filtering out different kinds of
noise.

## Domain of Application

## Combining FAI and RL

The techniques presented in this repository do not tackle "learning" in any way. They are just
tools for generating data efficiently.

It is true that we do not know the true state of the “real world”, so FMC (as for any other MCTS
variation) cannot be applied directly in production models. However, we do not know of any
application of RL that trains an agent directly in the real world, without some pre-training using
a simplified computer model of the real environment. We are aware that the use of a training
environment is considered a “hack” to avoid breaking robots, but as long as it is needed, we
propose taking this hack one step further.

In the case of having access to a training environment, FMC could be used to overcome one of the
current bottlenecks of RL: efficiently generating high quality samples.

It could be possible to generate millions of games with record scores within an hour. Once trained,
the model would not longer need FMC to act on the real world. The data generated could be used in
some of the following ways:

* Training an embedding that learns to capture the internal dynamics of the environment. The
embedding could be used as input for a DQN instead of cropping the screen differently in each game,
or as a weight initialization to avoid dependence on initial conditions.

* Traditional reward discounting techniques or policy gradients could be applied on rollouts
generated by FMC, instead of rollouts generated by the same network that is being trained. We
hypothesize that this could help stabilize learning, and dealing with unbounded rewards.

* A network trained in a fully supervised manner on samples generated by FMC, and then used as a
baseline for a traditional RL agent, maybe could help in improving performance.

* Improving already existing applications where MCTS is used, such as AlphaZero, and AlphaChem. The
data presented suggests that using FMC instead of MCTS could result in a boost on performance.

The hacks proposed could work directly with dm_control, gym, and virtually with any function that
can be sampled.



## Using FAI to train robots

FMC can also be applied to continuous control tasks. It is possible to run environments from
the dm_control library, although without proper reward shaping and defining boundary conditions,
each run takes hours. All the tasks but humanoid have been solved in minutes using custom
boundary conditions (and sometimes reward shaping).

Sergio built a custom training environment to test our methods in different continuous control
tasks. In those 2D toy examples that we used for testing and debugging purposes, we were able to
sample [high quality trajectories](https://www.youtube.com/watch?v=tLsu0On61CI&list=PLEXwXLT-a6beFPzal3OznPQC0pieccAle&index=29)
in tasks which involved up to [36 degrees of freedom](https://www.youtube.com/watch?v=XD9Fumzf57Y).
Given that those trajectories were sampled with an early algorithm (also derived using FAI) that is
about two orders of magnitude less efficient than than FMC, we believe that the ideas proposed for
discrete mdps could also be applied to robots.

## Black box optimization

Although the first tool we have published is FMC, we do not believe it is the most suitable
algorithm for training a policy that can be applied to real world problems. We think that FAI
really shines as a black box optimization technique, such as similar alternatives like Evolutionary
Strategies.

There are at least two ways of leveraging FAI as a black box technique for improving RL models:
As a meta parameter optimization tool, where we evolve the parameters of a population of RL models,
or as an alternative to [Evolutionary Strategies](https://arxiv.org/pdf/1703.03864.pdf) to directly
train the weights of the network.

We have data that suggest that FAI techniques beat ES in many kinds of low dimensional global
optimization problems. [GAS](https://arxiv.org/pdf/1705.08691.pdf) Shows how an early algorithm
derived from FAI that incorporates a tabu search-like strategy, and a multi-scale exploration
mechanism that outperforms alternative metaheuristic techniques. We tested GAS in optimizing
Lennard-Jones energy landscapes up to 80 degrees of freedom, but we couldn’t manage to benchmark
the same problem against other techniques, because we couldn’t get any of them to work on those
problems.

We believe that trying ES with a novelty search-inspired algorithm, derived from first principles
of FAI, could prove useful in training large networks. For example, deriving an algorithm with the
following properties is pretty straightforward:

* We can balance the spatial distribution to match the reward distribution of the action space using
the virtual reward. If done right, this should weight the gradients to favor a more controlled
evolution process.

* It is possible to add a more sophisticated tabu-like memory like we did in
[GAS](https://arxiv.org/pdf/1705.08691.pdf), in a similar way that novelty search does. The
diversity with respect to previously explored solutions could bebalanced in the virtual reward
formula, at the same time we balance the exploration/exploitation information of the walkers.

* We could perturbate the weights either using some standard technique like mirrored sampling, or
any other technique that proves to work in this kind of problem.

* In order to integrate the weights it would be possible to use any trick such as applying ranks,
or using different standard optimizers the same way ES does. FAI allows to be combined with
virtually any mathematical technique, but it is also possible to use an integration technique
derived from FAI, depending on the specific behaviour desired for the walkers.

FAI-derived tools are also really simple to escalate, and can greatly benefit from an increase in
computational resources.
