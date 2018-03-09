# Fractalai: An approach to General Artificial Intelligence

### Sergio Hernández Cerezo and Guillem Duran Ballester
Under construction

## Table of Contents

- [Abstract](#abstract)
- [Running the code](#installation)
- [Benchmarks](#benchmarks)
- [Additional Resources](#additional-resources)
  * [Theoretical foundations](#theoretical-foundations)
  * [Blog](#blog)
  * [YouTube](#youtube)
  * [Related papers](#related-papers)
- [FAQ](#faq)
- [About](#about)


## Abstract

Fractal AI is a theoretical framework for general artificial intelligence, 
that can be applied to any kind of Markov decision process.
In this repository we are are presenting a new Agent, derived from the first principles of the theory,
 which is capable of solving Atari games several orders of magnitude more efficiently than 
 similar techniques, like Monte Carlo Tree Search. 

The code provided shows how it is now viable to beat some of the current state of the art benchmarks on Atari games,
using a less than 1000 samples to calculate one action. Fractal AI makes it possible to generate a huge database of
 top performing examples with very little amount of computation required, transforming Reinforcement Learning into a 
 supervised problem.
 
 The algorithm presented is capable of solving the exploration vs exploitation dilemma, while
 maintaining control over any aspect of the behavior of the Agent. From a mathematical perspective, 
 Fractal AI also offers a new way of measuring intelligence, and complexity in any kind of state space, 
 thus giving rise to a new kind of risk control theory.

## Installation

This code release relies aims for simplicity and self-explainability. 
So it should be pretty straightforward to run in Python 3. Python 2 is not supported.

It only needs numpy and [gym["atari"]](https://github.com/openai/gym), although we also recommend
 installing the Jupyter Notebook for running the example.

#### Installing dependencies
 
First, install the dependencies explained on the gym documentation.

>To install the full set of environments, you'll need to have some system
packages installed. We'll build out the list here over time; please let us know
what you end up installing on your platform.
>In case you want to run the notebook:
>
>  ``pip3 install jupyter``
>
>On OSX:
>
>   ``brew install cmake boost boost-python sdl2 swig wget``
>
>On Ubuntu 14.04:
>
>    ``sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev 
>xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig``

#### Cloning and installing the repository

``git clone git@github.com:Guillem-db/FractalAI.git``

``cd FractalAI``

``sudo pip3 install -e .``


## Benchmarks
>It doesn't matter how beautiful your theory is, it doesn't matter how smart you are. 
>
>If it doesn't agree with experiment, it's wrong.
>
> **Richard P. Feynman**

We know we are making some claims that may be found difficult to believe. That is why we have tried to make it very
easy to verify that Fractal AI works. 

The following benchmarks have been calculated in a laptop using a single thread implementation. 
Some of them can be replicated in real time, and others require up to 20k samples per action, but anyone running this code
should be able to get similar performance.


|Game          |FAI Score|% vs Best AI|% vs MCTS|Mean samples per action|N repeat action|Time horizon|Max samples per action|Max states|
| :----------: |:------: | :--------: |--- |--- |--- |--- |--- |--- |
|alien         |19380  |271.89%     |NaN|1190|5|25|2000|50|
|amidar        |4306   |194.40%     |NaN|1222|5|25|2000|50|
|assault       |1280   |17.06%      |NaN|1317|5|25|2000|50|
|asteroids     |76270  |160.94%     |NaN|2733|2|20|5000|NaN|
|beam rider    |2160   |12.64%      |29.86%|4052|5|25|2000|100|
|boxing        |100    |104.17%     |NaN|2027|3|30|2000|150|
|breakout      |36     |7.98%       |8.87%|5309|5|30|20000|150|
|centipede     |529355 |4,405.05%   |NaN|1960|1|20|6000|100|
|double dunk   |20     |400.00%     |NaN|5327|3|50|8000|NaN|
|enduro        |471    |28.05%      |59.77%|NaN|5|15|2000|50|
|ice hockey    |52     |5,200.00%   |NaN|12158|3|50|20000|250|
|ms pacman     |58521  |372.91%     |NaN|5129|10|20|20000|250|
|qbert         |35750  |189.66%|189.66%|2728|5|20|5000|NaN|
|seaquest      |3180   |7.56%|97.64%|6252|5|25|20000|250|
|space invaders|3605   |80.29%|153.14%|4261|2|35|6000|NaN|
|tennis        |16     |177.78%|NaN|2437|4|30|5000|NaN|
|video pinball |496681 |64.64%|NaN|1083|2|30|5000|NaN|
|wizard of wor |93090  |785.44%|NaN|2229|4|35|5000|NaN|

In the following Google Sheet we are logging the performance of our Agent relative to the current alternatives.
If you find that some benchmark is outdated, or you are not capable of replicating any of our results, please
open an issue and we will update the document.

> [Fractal AI Performance Sheet](https://docs.google.com/spreadsheets/d/1JcNw2L0YL_I2iGZPJ0bNKJshlTaqMuEl5CP2W5zie6M/edit?usp=sharing)

#### Or just run the example and check by yourself!

## Additional Resources

### Theoretical foundations
[A Fragile Theory of Intelligence](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing):
This document explains the fundamental principles of the Fractal AI theory in which our Agent is based. 
We tried very hard to build our own solution, so we worked all the fundamental principles completely from scratch.
This means that it should contain anything you need to understand the theory without further reading required.
Any comment on how to explain things more clearly will be welcome.

### Blog
 [Sergio's Blog: EntropicAI](http://entropicai.blogspot.com.es/):
 Here we have documented and explained the evolution of our research process for developing this algorithm,
 and some experiments where we tried to apply our theory to other fields than reinforcement learning.
 
### YouTube

[Sergio's YouTube channel](https://www.youtube.com/user/finaysergio/videos)
Here you can find some videos of what we accomplished over the years. Among other things, you can find videos 
recorded using a custom library, which can be used to create different task in continuous control environments,
  and visualizations of how the Agent samples the state space.

### Related Papers

[GAS Paper](https://arxiv.org/abs/1705.08691):
 We tried to publish a paper describing an application of our theory to general optimization,
but it was not published because our method "lacked scientific relevance", and there was no need for more algorithms that were not proven to work at a huge scale.
As we lack the resources to deploy our methods at a bigger scale, we were unable to meet the requirements for publishing. 

There are better ways to apply our theory to general optimization, but it is a nice example of why code is better than math to explain our theory. When you try to formalize it, 
it gets really non-intuitive.

[Causal Entropic Forces by Alexander Wiessner Gross](http://alexwg.org/publications/PhysRevLett_110-168702.pdf): 
The fundamental concepts behind this paper inspired our research. We develop our theory aiming to calculate future entropy faster,
 and being able to leverage the information contained in the Entropy of any state space, together with any potential function.
 

## Conjectures on some of its properties


We cannot provide any formal proof about this algorithm, because we don't know any tool suitable for analyzing
 the far-from-equilibrium regime in which the Agent operates. These are just conjectures and they could be wrong.
 
 Any suggestion about how to prove our conjectures will be welcome. This list is non-exhaustive and it will be updated.
 
 **State Swarm**: Structure consisting of different states that interact with each other in order to build a cusal cone. The agent uses a Swarm to build a causal cone used to approximate the Q values of each action.
 
- ***It is possible to prove that this algorithm is unprovable with any known mathematical tools.***
 Maybe someone can proof that FAI is unprovable.
![Improvable](assets/improvable.png)   
 
- ***A State Swarm can leverage efficiently both the information contained in the physical structure of a given State Space (Entropy/Exploration), and the potential field associated with each state.***

 This means that we are not taking only into account "how good an state is", but also "How different an state is with respect to the others", effectively solving the exploration vs exploitation problem.
 
- ***This algorithm tends to achieve symmetry. In the limit, a swarm of states will be distributed proportionally to the space-time reward distribution of the space state.*** 

 If we fix all the states in the Swarm to share time, states distribution in each slice of the causal cone  will be proportional to its reward density distribution in the limit.
 If we do not fix the time, a Swarm will automatically adjust to also distribute the states symmetrically with respect to the time horizon.
  
- ***Given a uniform prior, this algorithm will never perform worse than random. And it will only perform randomly when no information is available in the sampled Swarm of states that we can levare to build a more efficient causal cone.*** 

    Yes, I have read about the No Free Lunch Theorem, and I think this is an exception.

- ***P vs NP is not the right question to ask***. If we happen to be right, and complexiti is better measured using our methods, there would be NP hard problems which should be possible to solve in polynomial time. Our complexity measure can classify some P, and NP problems in the same category.

 
## FAQ

## About
