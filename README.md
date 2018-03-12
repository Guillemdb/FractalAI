# Fractal AI: A fragile theory of intelligence
### Sergio Hernández Cerezo and Guillem Duran Ballester

![Boxing-v0](assets/boxing.gif "Boxing-v0 76 samples per action") ![MsPacman-v0](assets/mspacman.gif "MsPacman-v0 154 samples per action") ![Tennis-v0](assets/tennis.gif "Tennis-v0 1250 samples per action")   ![Centipede-v0](assets/centipede.png
  "Centipede-v0 1960 samples per action") ![MontezumaRevenge-v0](assets/montezuma.gif
 "MontezumaRevenge-v0 5175 samples per action")



>“Once you start doubting, just like you’re supposed to doubt, you ask me if the science is true.
 You say no, we don’t know what’s true, we’re trying to find out and everything is possibly wrong.”
>
>  **Richard P. Feynman**, *The Pleasure of Finding Things Out*: The Best Short Works of Richard P. Feynman 

## Table of Contents

- [Abstract](#abstract)
- [Running the code](#installation)
- [Benchmarks](#benchmarks)
- [Additional Resources](#additional-resources)
  * [Theoretical foundations](#theoretical-foundations)
  * [Blog](#blog)
  * [YouTube](#youtube)
  * [Related papers](#related-papers)
- [Other tasks solved](#other-tasks-solved)
- [Other applications](#other-applications)
- [Conjectures on some of its properties](#conjectures-on-some-of-its-properties)
- [Cite us](#cite-us)
- [FAQ](#faq)
- [About](#about)
- [Bibliography](#bibliography)


## Abstract


[Fractal AI](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing) 
is a theoretical framework for general artificial intelligence. It allows to derive new mathematical
 tools that constitute the foundations for a new kind of stochastic calculus, by modelling
  information using cellular automaton-like structures instead of smooth functions.

In this repository we are presenting a new Agent, derived from the first principles of the theory,
 which is capable of solving Atari games several orders of magnitude more efficiently than other 
 similar techniques, like Monte Carlo Tree Search. 

The code provided shows how it is now possible to beat some of the current state of the art
 benchmarks on Atari games, using less than 1000 samples to calculate each one of the actions.
  Among other things, Fractal AI makes it possible to generate a huge database of
 top performing examples with very little amount of computation required, transforming 
 Reinforcement Learning into a supervised problem.
 
 The algorithm presented is capable of solving the exploration vs exploitation dilemma, while
 maintaining control over any aspect of the behavior of the Agent. From a general approach, 
 new techniques presented here have direct applications to other areas such as: Non-equilibrium
 thermodynamics, chemistry, quantum physics, economics, information theory, and non-linear
 control theory.
  
  
## Installation

This code release aims for simplicity and self-explainability. 
It should be pretty straightforward to run in Python 3. Python 2 is not supported.

It only needs numpy and [gym["atari"]](https://github.com/openai/gym) **[[1](#bibliography)]**, although we also recommend
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
>xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig libav-tools``

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

The following benchmarks have been calculated in a laptop using a single thread implementation. 
Some of them can be replicated in real time, and others require up to 20k samples per action, but anyone running this code
should be able to get similar performance.

We show performance with respect to benchmarks widely accepted in the reinforcement learning community. **[[4](#bibliography)]**, **[[5](#bibliography)]**, **[[6](#bibliography)]**, **[[7](#bibliography)]**, **[[8](#bibliography)]**


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Game</th>
      <th>Mean samples per action</th>
      <th>FAI Score</th>
      <th>% Best AI</th>
      <th>% Human 2H</th>
      <th>% Absolute Record</th>
      <th>% MCTS 3M samples</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>alien</td>
      <td>1,190</td>
      <td>19,380</td>
      <td>328.53%</td>
      <td>271.89%</td>
      <td>7.69%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>amidar</td>
      <td>1,222</td>
      <td>4,306</td>
      <td>194.40%</td>
      <td>250.35%</td>
      <td>2.77%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>assault</td>
      <td>1,317</td>
      <td>1,280</td>
      <td>17.06%</td>
      <td>85.56%</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>asteroids</td>
      <td>2,733</td>
      <td>76,270</td>
      <td>289.12%</td>
      <td>160.94%</td>
      <td>0.76%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>beam rider</td>
      <td>4,052</td>
      <td>2,160</td>
      <td>12.64%</td>
      <td>12.76%</td>
      <td>0.22%</td>
      <td>29.86%</td>
    </tr>
    <tr>
      <th>5</th>
      <td>boxing</td>
      <td>2,027</td>
      <td>100</td>
      <td>104.17%</td>
      <td>833.33%</td>
      <td>101.01%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>breakout</td>
      <td>5,309</td>
      <td>36</td>
      <td>7.98%</td>
      <td>113.21%</td>
      <td>NaN</td>
      <td>8.87%</td>
    </tr>
    <tr>
      <th>7</th>
      <td>centipede</td>
      <td>1,960</td>
      <td>1,024,000</td>
      <td>11,764.71%</td>
      <td>8,521.26%</td>
      <td>78.67%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>crazy climber</td>
      <td>1,243</td>
      <td>217,900</td>
      <td>163.75%</td>
      <td>608.17%</td>
      <td>48.75%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>double dunk</td>
      <td>5,327</td>
      <td>20</td>
      <td>400.00%</td>
      <td>-129.03%</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>enduro</td>
      <td>826</td>
      <td>476</td>
      <td>28.35%</td>
      <td>55.35%</td>
      <td>13.16%</td>
      <td>60.41%</td>
    </tr>
    <tr>
      <th>11</th>
      <td>freeway</td>
      <td>807</td>
      <td>10</td>
      <td>30.30%</td>
      <td>33.33%</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ice hockey</td>
      <td>12,158</td>
      <td>52</td>
      <td>NaN</td>
      <td>5,200.00%</td>
      <td>144.44%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>montezuma</td>
      <td>5,175</td>
      <td>2,500</td>
      <td>16,666.67%</td>
      <td>52.60%</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ms pacman</td>
      <td>5,129</td>
      <td>58,521</td>
      <td>1,390.38%</td>
      <td>372.91%</td>
      <td>20.17%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>phoenix</td>
      <td>1,289</td>
      <td>11,930</td>
      <td>32.38%</td>
      <td>164.71%</td>
      <td>0.30%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>qbert</td>
      <td>2,728</td>
      <td>35,750</td>
      <td>189.66%</td>
      <td>265.70%</td>
      <td>1.49%</td>
      <td>189.66%</td>
    </tr>
    <tr>
      <th>17</th>
      <td>seaquest</td>
      <td>6,149</td>
      <td>5,220</td>
      <td>41.65%</td>
      <td>12.41%</td>
      <td>0.99%</td>
      <td>160.27%</td>
    </tr>
    <tr>
      <th>18</th>
      <td>space invaders</td>
      <td>4,261</td>
      <td>3,605</td>
      <td>80.29%</td>
      <td>216.00%</td>
      <td>0.58%</td>
      <td>153.14%</td>
    </tr>
    <tr>
      <th>19</th>
      <td>tennis</td>
      <td>1,242</td>
      <td>24</td>
      <td>266.67%</td>
      <td>-300.00%</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>tutankham</td>
      <td>3,023</td>
      <td>223</td>
      <td>69.47%</td>
      <td>132.74%</td>
      <td>6.38%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>video pinball</td>
      <td>1,083</td>
      <td>604,043</td>
      <td>78.61%</td>
      <td>3,418.85%</td>
      <td>0.66%</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>wizard of wor</td>
      <td>2,229</td>
      <td>93,090</td>
      <td>785.44%</td>
      <td>1,957.32%</td>
      <td>39.83%</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

In the following Google Sheet we are logging the performance of our Agent relative to the current alternatives.
If you find that some benchmark is outdated, or you are not capable of replicating some of our results, please
open an issue and we will update the document.

> [Fractal AI Performance Sheet](https://docs.google.com/spreadsheets/d/1JcNw2L0YL_I2iGZPJ0bNKJshlTaqMuEl5CP2W5zie6M/edit?usp=sharing)

#### Or just run the example and check by yourself!

## Additional Resources

### Theoretical foundations
[Fractal AI: A Fragile Theory of Intelligence](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing):
This document explains the fundamental principles of the Fractal AI theory in which our Agent is based. 
We tried very hard to build our own solution, so we worked all the fundamental principles completely from scratch.
This means that it should contain anything you need to understand the theory without further reading required.
Any comment on how to explain things more clearly will be welcome.

### Blog
 [Sergio's Blog: EntropicAI](http://entropicai.blogspot.com.es/):
 Here we have documented and explained the evolution of our research process for developing this algorithm,
 and some experiments where we tried to apply our theory to other research fields.
 
### YouTube

[Fractal AI playlist](https://www.youtube.com/playlist?list=PLEXwXLT-a6beFPzal3OznPQC0pieccAle)
Here you can find some videos of what we accomplished over the years. Among other things, you can find videos 
recorded using a custom library, which can be used to create different task in continuous control environments,
  and visualizations of how the Agent samples the state space.

### Related Papers

[GAS Paper](https://arxiv.org/abs/1705.08691) **[[3](#bibliography)]**:
 We tried to publish a paper describing an application of our theory to general optimization,
but it was not published because our method "lacked scientific relevance", and there was no need for more algorithms that were not proven to work at a huge scale.
As we lack the resources to deploy our methods at a bigger scale, we were unable to meet the requirements for publishing. 

There are better ways to apply our theory to general optimization, but it is a nice example of why code is better than math to explain our theory. When you try to formalize it, 
it gets really non-intuitive.

[Causal Entropic Forces by Alexander Wissner-Gross](http://alexwg.org/publications/PhysRevLett_110-168702.pdf) **[[2](#bibliography)]**: 
The fundamental concepts behind this paper inspired our research. We develop our theory aiming to calculate future entropy faster,
 and being able to leverage the information contained in the Entropy of any state space, together with any potential function.
 
## Other tasks solved

Besides Atari games, we have also used our theory to solve different continuous control environments involving task such as:

- **Collecting rocks with a spaceship** ([Video](https://www.youtube.com/watch?v=HLbThk624jI) and [blog post](http://entropicai.blogspot.com.es/2016/04/understanding-mining-example.html)): 
    This agent can catch rocks using a hook that behaves like an elastic band. We are capable of sampling low  probability trajectories in such chaotic space state.
       
       
- **Multi Agent Environments**: It is aso possible to control multi agent environments, like [Maintaining a formation](https://www.youtube.com/watch?v=J9kW1lhT06A), [cooperating to achieve a shared goal](https://www.youtube.com/watch?v=DsvSH3cNhnE), or [fighting](http://entropicai.blogspot.com.es/2015/05/tonight-four-of-my-new-fractal-minded.html) against each other.
 A nice property of our methods is that their computational cost scales linearly with the number of agents. 
       

- **Stochastic simulations**: It can even [handle uncertainty in a continuous domain](http://entropicai.blogspot.com.es/2015/06/passing-asteroids-test.html?m=0).
You can also check this on Atari games by setting the clone_seeds parameter of the agent to False.


- **Multi objective and multi agent path finding**: We can solve multi objective path finding in nearly real time.  [Video 1](https://www.youtube.com/watch?v=AoiGseO7g1I) [Video 2](https://www.youtube.com/watch?v=R61FRUf-F6M) [Blog Post](http://entropicai.blogspot.com.es/search/label/Path%20finding)


- **General optimization**: Here you can find a [visual representation](http://entropicai.blogspot.com.es/2016/02/serious-fractal-optimizing.html?m=0) of how our algorithm explores the state space.


## Other applications

- **Physics**: Physics is basically a path finding problem, so our theory can be thought as a sort of non-equilibrium statistical mechanics. 
Given that our algorithm is surprisingly good at path finding, we wonder how well it can be applied to solve Feynman path integrals.
 Conceptually, it is relatively simple to map some properties of a Swarm, to the properties of a wave function.
 If you used something similar to our agent to move around the gaussian packets that are used when sampling integrals, 
 maybe it would be easier to focus on regions with a meaningful contribution to the sum.
 
- **Neural nets**: It is possible to use our theory to make Deep Learning more efficient, but this code release does not focus on models.
                    For now, it should be pretty clear that using FAI instead of MCTS is worth trying.
                    
- **Evolutionary strategies**: The principles of the theory also allow to design evolutionary strategies for training DNN,
 using something conceptually similar to what [Uber](https://eng.uber.com/deep-neuroevolution/) did.
  This is the way to go in case you want to solve Starcraft quickly without understanding the theory.
  Using this method, guarantees that you will end up with something you cannot control.
  If you try this in a properly scaled implementation without *perfect understanding*, a long term disaster is guaranteed.

- **Economics**: Our theory allow us to quantify and model the *personality* and *irrationality* of an agent, 
and it has non-equilibrium risk-control mechanisms, we bet someone will think of an interesting application.

- **Looks like Alien Math**: It is so weird that it almost qualifies as *"alien math"*. If you only knew this algorithm,
 you could pretty much arrive at the same conclusions as our current scientific knowledge arrives. 
 It is funny to think that Science without gradients is also possible.
 

## Conjectures on some of its properties


We cannot provide any formal proof about this algorithm, because we don't know any tool suitable for analyzing
 the far-from-equilibrium regime in which the Agent operates. These are just conjectures and they could be wrong.
 
 Any suggestion about how to prove our conjectures will be welcome. This list is non-exhaustive and it will be updated.
 
 **State Swarm**: Structure consisting of different states that interact with each other in order to build a causal cone. 
 The agent uses a Swarm to build a causal cone used to approximate the Q values of each action.
 
- ***It is possible to prove that this algorithm is unprovable with any known mathematical tools.***

  Maybe someone can proof that FAI is unprovable **[[9](#bibliography)]**.
 
![Improvable](assets/improvable.png)   
 
- ***A State Swarm can leverage efficiently both the information contained in the physical structure of a given State Space (Entropy/Exploration), and the potential field associated with each state.***

  This means that we are not taking only into account "how good an state is", but also "how different an state is with respect to the others", effectively solving the exploration vs exploitation problem.
 
- ***This algorithm tends to achieve symmetry. In the limit, a swarm of states will be distributed proportionally to the space-time reward distribution of the space state.*** 

  If we fix all the states in the Swarm to share time, states distribution in each slice of the causal cone  will be proportional to its reward density distribution in the limit.
 If we do not fix the time, a Swarm will automatically adjust to also distribute the states symmetrically with respect to the time horizon.
  
- ***Given a uniform prior, this algorithm will never perform worse than random. And it will only perform randomly when no
 information can be extracted from the different states in the Swarm. Changing the prior will allow for worse than random games, but
 it will increase the performance in other problems.*** 

    Yes, we have read about the No Free Lunch Theorem, and we think this is an exception.

- ***P vs NP is not the right question to ask***. 
 
  If we happen to be right, and complexity is better measured using our methods, there would be NP hard problems which should be possible to solve in polynomial time. 
  Our complexity measure can classify some P and NP problems in the same category.

- ***There exits an arbitrary good approximation to [Density Functional Theory](https://en.wikipedia.org/wiki/Density_functional_theory) that scales linearly with the number of particles, and which uncertainty depends on the amount of computational resources used to calculate the approximation.*** 

  If you treat electrons as agents, you can use the minimum action principle to formulate a proper approximation of the potential
   function in almost any known physical problem. Then you can move the particles around as if you were solving a multi-agent environment. 
 Our method  scales linearly with the number of particles, so it gives a new approach to complex problems.
 
- ***Is it possible to create a functional AGI using only fractal methods***.
 With proper funding, a lot of effort, and very large amounts of computer power we think we can build an AGI within 10 years.


## Cite us

    @misc{HERE-ARXIV-ID,
        Author = {Sergio Hernández Cerezo and Guillem Duran Ballester},
        Title = {Fractal AI: A fragile theory of intelligence},
        Year = {2018},
        Eprint = {arXiv:HERE-ARXIV-ID},
      }

## FAQ

If there are any questions regarding our methods, we will be answering them here.
 
## About

We have developed this theory for the pleasure of finding thing out as a hobby, while I was at college, and Sergio worked as a programmer.
 We had almost no financial support, nor access to a proper technical infrastructure, besides the 6 months [Source{d}](https://sourced.tech/) sponsored us. 
 
 We are not part of academia, we have no affiliation and no track record.
 
 We don't have the resources to carry on further our research, but we will gladly accept any contribution or
  sponsorship that allows us to continue working in our passion.


- **[Sergio Hernández Cerezo](https://twitter.com/EntropyFarmer)**: Studied mathematics, works as programmer, dreams about physics.

- **[Guillem Duran Ballester](https://twitter.com/Miau_DB)**: Rogue scientist, loves learning and teaching. Currently looking for an AI-related job.

**Special thanks**: We want to thank all the people who has believed in us during this years.
 Their patience, understanding, and support made possible this work.
 
 - Our families, HCSoft, Guillem's parents: Joan and Francisca, [Eulàlia Veny](https://twitter.com/linguistsmatter) and Fina. 
 
 - The people at sourced, specially [Eiso Kant](https://twitter.com/eisokant), [Waren Long](https://twitter.com/warenlg), [Vadim Markovtsev](https://twitter.com/tmarkhor),
  [Marcelo Novaes](https://twitter.com/marnovo), and [Egor Bulychev](https://twitter.com/egor_bu).
 
 - Everyone who believed in our Alien math since Guillem was in college, specially [Antoni Elias](https://twitter.com/eliasfuste),
 [Jose Berengueres](https://twitter.com/harriken), [Javier Ozón](https://twitter.com/fjozon), and [Samuel Graván](https://twitter.com/Samuel__GP).
 
 ## Bibliography
 
 -  **[1]**  Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba.
***OpenAI Gym*** . [arXiv:1606.01540](https://arxiv.org/pdf/1606.01540.pdf), 2016.

- **[2]**  Alexander Wissner-Gross. ***Causal entropic forces*** . [Physical Review Letters](http://alexwg.org/publications/PhysRevLett_110-168702.pdf), 2013.

- **[3]**  Sergio Hernández, Guillem Duran, José M. Amigó. ***General Algorithmic Search***. [arXiv:1705.08691](https://arxiv.org/abs/1705.08691), 2017.

- **[4]**  Volodymyr Mnih & others. ***Human-level control through deep reinforcement learning***. [doi:10.1038/nature14236](http://www.davidqiu.com:8888/research/nature14236.pdf), 2015.

- **[5]**  Guo, Xiaoxiao and Singh, Satinder and Lee, Honglak and Lewis, Richard L and Wang, Xiaoshi. 
***Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning***. [NIPS2014_5421](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), 2014.

- **[6]**  Matthias Plappert, Rein Houthooft, Prafulla Dhariwal, Szymon Sidor, Richard Y. Chen, Xi Chen, Tamim Asfour, Pieter Abbeel, Marcin Andrychowicz.
 ***Parameter Space Noise for Exploration***. [arXiv:1706.01905](https://arxiv.org/abs/1706.01905).

- **[7]**  Justin Fu and Irving Hsu. ***Model-Based Reinforcement Learning for Playing Atari Games***.
 [Stanford Report](http://cs231n.stanford.edu/reports/2016/pdfs/116_Report.pdf).
 
-  **[8]**  ***ATARI VCS/2600 Scoreboard***. [Atari compendium](http://www.ataricompendium.com/game_library/high_scores/high_scores.html), 2018.

-  **[9]** Shane Legg ***Machine Super Intelligence***. [Doctoral Dissertation ](http://www.vetta.org/documents/Machine_Super_Intelligence.pdf), 2008.