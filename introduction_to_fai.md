# Introduction to Fractal AI theory

This is the fastest way to get an idea of what our work is about. Here you will find qualitative
explanations of the inner workings of our techniques, and possible applications to real world
problems.

## Table of Contents

- [What is FAI?](#what-is-fai?)
- [Swarm Wave](#swarm-wave)
- [Fractal Monte Carlo](#fractal-monte-carlo)
- [Domain of application](#domain-of-application)
- [Combining FAI and RL](#combining-fai-and-rl)
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

Swarms with different behaviours could be used as an alternative to traditional
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

According to FAI, FMC is an algorithm to efficiently explore functions that contain information on
the input-output relation for a system, given a bound in computational resources.  It is meant to be
a robust path-search algorithm that efficiently approximates path integrals formulated as a Markov
decision process. FMC calculates each step of the path independently, but uses information extracted
from previous time steps to adjust its parameters.

### Domain of Application



FMC applied to Atari is a toy example, however since we applied FMC in the context of Markov
decision processes, we do not actually need a perfect model.  The swarm that FMC uses is not meant to be
applied only when we have access to perfect models, but also in any mathematical function that
does not require to dynamically adjust the step size(dt) when being integrated. 

In this case we showed that it works on the ALE interface, which can provide either a deterministic
or an stochastic environment, but it could be possible to alter the environment that the swarm uses
by adding noise, (to observations, actions, rewards, distance measure, or boundary conditions) and
FMC will still perform nicely. Unfortunately, when not assuming a perfect model we experience a
penalty in performance, although FMC is capable of successfully filtering out different kinds of
noise.

### How it works

*We provide links to the specific lines of the code where the described parts of the algorithm take
place. For example, [(L45)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L45)
will be a reference to the line 45 of the file [fractalmc.py](fractalai/fractalmc.py)*

When calculating an action, FMC will construct a tree that consists of potential trajectories that
describe the future evolution of the system. This tree, called causal cone, is expanded by a
swarm [(L58, 105)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L58-L105)
of walkers that populates its leaf nodes. The swarm will undergo an iterative process [(L322)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L322)
in order to make the tree grow efficiently. When a maximum amount of computation has been reached,
the utility of each action will be considered proportional to the number of walkers that populate
leaf nodes originating from the same action [(120, 126)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L120-L126).

The causal cone, unlike in MCTS, is not a static tree of all possible actions that will be explored.
Instead the causal cone is a tree data structure that changes at every time step by applying random
perturbations [(L142)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L142),
and letting the swarm move freely among different leaf nodes of the tree.
[(L199)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L199)

In order to evolve the swarm, we first initialize the walkers at the root state, perturb them, and store
the action chosen [(L136)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L136).
Then use the following algorithm to make it evolve until the maximum number of samples allowed is reached:

1. Measure the euclidean distance between all the observations of all the walkers, and the observation of
another walker chosen at random [(L164)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L164).
This will create an stochastic measure of diversity, that when incorporated into the virtual
reward formula, will favor the diversity among the states in the swarm.

2. Normalize the values so all the walkers' distances fall into the [0, 1] range [(L175)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L175). 
Normalize the rewards to be in range [1, 2]. [(L184)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L184).
This allows us to get rid of problems with the scale of both distances and rewards, and assures
that the value of the virtual distance will be bounded.

3. Calculate the virtual reward for each walker. [(L177)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L177).
This value represents an stochastic measure of the importance of a given walker with respect
to the whole swarm. It combines both an exploration term (distance) with an exploitation
term (reward) that is weighted by the balance coefficient, which represents the current trade-off
between exploration and exploitation, and helps modeling risk [(L304)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L304).

4. Each walker of the swarm compares itself to another walker chosen at random [(L215)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L215),
and gets assigned a probability of moving to the leaf node where the other walker is located [(L219)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L219).
 
5. Determine if a walker is dead [(L209-212)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L209-L212).
Then decide if the walker will clone or not depending on its death condition and clone probability [(L220-223)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L220-L223).
The death condition is a flag set by the programmer that lets us incorporate arbitrary boundary
conditions to the behaviour of the agent. The death flag helps the swarm avoiding undesired
regions of the state space.

6. Move the walkers that are cloning to theirs target leaf nodes [(L224-228)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L224-L228).
This allows for recycling the walkers that either fall out of the desired domain (dead) or have
been poorly valued with respect to the whole swarm. It also partially avoids exploring regions of
the state space that are either too crowded (low diversity) or have a very poor reward.

7. Choose an action for each walker and step the environment (perturbation).
The swarm will evolve and explore new states. This is how you make the causal cone grow [(L142)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L142).
The fact that we are choosing between cloning and exploring allows for a non-uniform growth of the 
causal cone's time horizon. A walker can clone to a leaf node which has a different
depth than its current leaf node, meaning that jumps forward and backwards in time are allowed.

8. GOTO 1 until the maximum number of samples is reached. By iterating each time, we are
redistributing the "useless" walkers to more promising leaf nodes, and perturbing the states located
in the regions considered to have the highest utility. After several iterations, the density
distribution of the walkers should match the reward density distribution of the state space.

9. Approximate the utility for each action according to [(120, 126)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L120-L126).
Take the action with more utility. Note that we are just counting how many states in the swarm
took the same action in the root node.

After deciding an action, the swarm will update its parameters: the number of walkers, and the
number of times it will sample the state space to build the next causal cone. This update will be
adjusted by a non linear feedback loop, with the objective of keeping the mean depth of the cone
as close as possible to a desired time horizon [(L262, 280, 290)](https://github.com/FragileTheory/FractalAI/blob/6b62d79559364c222025dbf3da669f0ac8a38c09/fractalai/fractalmc.py#L262-L290).

### Parameters

#### Set by the programmer

- **Fixed steps**: It is the number of consecutive times that we will apply an action to the
environment when we perturb it choosing an action. Although this parameter actually depends on the
environment, we can use it to manually set the frequency at which the agent will play. Taking more
consecutive fixed steps per action allows for exploring further into the future at the cost of
longer reaction times.

- **Time Horizon**: This value represents how far we need to look into the future when taking an
action. A useful rule of thumb is **Time Horiozon = Nt / Fixed steps**, where **Nt** is the number
of frames that it takes the agent to loose one life (die) since the moment it performs the actions
that inevitably lead to its death. This parameters, multiplied by the fixed_steps, determines the
time horizon of the bigger potential well that the agent should be able to escape.

- **Max states**: This is the maximum number of walkers that can be part of the Swarm. This number
is related to "how thick" we want the resulting causal cone to be. The algorithm will try to use
the maximum number of walkers possible. 

- **Max samples**: This is the maximum number of times that we can make a perturbation when using
a swarm to build a causal cone. It is a superior bound, the algorithm will try to use as few
samples as possible in order to meet the defined **time horizon**. It is a nice way to set how
fast you need to take an action in the worst case. A reasonable value is **max walkers** \* **time horizon** \* ***N***,
being ***N=5*** a number that works well in Atari games, but highly depends on the task.


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

## Other tasks solved

Besides Atari games, we have also used our theory to solve different continuous control environments involving task such as:

- **Collecting rocks with a spaceship** ([Video](https://www.youtube.com/watch?v=HLbThk624jI) and
 [blog post](http://entropicai.blogspot.com.es/2016/04/understanding-mining-example.html)): 
    This agent can catch rocks using a hook that behaves like an elastic band. We are capable of
     sampling low  probability trajectories in such chaotic space state.
       
       
- **Multi agent environments**: It is aso possible to control multi agent environments, like
 [Maintaining a formation](https://www.youtube.com/watch?v=J9kW1lhT06A),
 [cooperating to achieve a shared goal](https://www.youtube.com/watch?v=DsvSH3cNhnE),
  or [fighting](http://entropicai.blogspot.com.es/2015/05/tonight-four-of-my-new-fractal-minded.html) against each other.
 A nice property of our methods is that their computational cost scales near linearly with the number of agents. 
       

- **Stochastic simulations**: It can even [handle uncertainty in a continuous domain](http://entropicai.blogspot.com.es/2015/06/passing-asteroids-test.html?m=0).
You can also check this on Atari games by setting the clone_seeds parameter of the agent to False.


- **Multi objective and multi agent path finding**: This technique can also be applied to path finding problems. [Video 1](https://www.youtube.com/watch?v=AoiGseO7g1I),
 [Video 2](https://www.youtube.com/watch?v=R61FRUf-F6M), [Blog Post](http://entropicai.blogspot.com.es/search/label/Path%20finding).


- **General optimization**: Here you can find a [visual representation](http://entropicai.blogspot.com.es/2016/02/serious-fractal-optimizing.html?m=0)
 of how the GAS algorithm explores the state space.