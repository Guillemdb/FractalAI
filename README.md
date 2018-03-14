# Fractal AI: A Fragile Theory of Intelligence
### Sergio Hernández Cerezo and Guillem Duran Ballester

![Boxing-v0](assets/boxing.gif "Boxing-v0 76 samples per action") ![MsPacman-v0](assets/mspacman.gif "MsPacman-v0 154 samples per action") ![Tennis-v0](assets/tennis.gif "Tennis-v0 1250 samples per action")   ![Centipede-v0](assets/centipede.png
  "Centipede-v0 1960 samples per action") ![MontezumaRevenge-v0](assets/montezuma.gif
 "MontezumaRevenge-v0 5175 samples per action")

>“Once you start doubting, just like you’re supposed to doubt, you ask me if the science is true.
 You say no, we don’t know what’s true, we’re trying to find out and everything is possibly wrong.”
>
>  **Richard P. Feynman**, *The Pleasure of Finding Things Out*: The Best Short Works of Richard P. Feynman 

## Table of Contents


**This README is still under active development.**


- [Abstract](#abstract)
- [Quick Start](quick-start)
- [Running the code](#installation)
- [Benchmarks](#benchmarks)
- [Additional Resources](#additional-resources)
  * [Theoretical foundations](#theoretical-foundations)
  * [Blog](#blog)
  * [YouTube](#youtube)
  * [Related papers](#related-papers)
- [Other tasks solved](#other-tasks-solved)
- [Cite us](#cite-us)
- [FAQ](#faq)
- [About](#about)
- [Todo](#todo)
- [Bibliography](#bibliography)


## Abstract

[Fractal Monte Carlo](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing) 
is a theory for efficiently sampling state spaces. It allows to derive new mathematical
 tools that may be useful for modelling information using cellular automaton-like structures
  instead of smooth functions.

In this repository we are presenting a new 
[Agent](https://github.com/FragileTheory/FractalAI/blob/master/fractalai/fractalai.py), derived
from the first principles of the theory,
 which is capable of solving Atari games several orders of magnitude more efficiently than other 
 similar techniques, like Monte Carlo Tree Search **[[1](#bibliography)]**. 

The code provided shows how it is now possible to beat some of the current state of the art
 benchmarks on Atari games, using less than 1000 samples to calculate each one of the actions when 
 MCTS uses 3 Million samples. Among other things, Fractal AI makes it possible to generate a huge database of
 top performing examples with very little amount of computation required, transforming 
 Reinforcement Learning into a supervised problem.
 
 The algorithm presented is capable of solving the exploration vs exploitation dilemma, while
 maintaining control over any aspect of the behavior of the Agent. 
 
 
## Quick start

You can refer to the [Example.ipynb](https://github.com/FragileTheory/FractalAI/blob/master/Example.ipynb) to 
see how the agent performs on any Atari game, either using RAM, or pixels as input. 

To get an idea on how the Agent works, you can refer to the 
[code](https://github.com/FragileTheory/FractalAI/blob/master/fractalai/fractalai.py), where every
 important aspect of the algorithm is commented. There is also a
 [blog post](https://entropicai.blogspot.com.es/2018/03/fractal-ai-recipe.html) where the
 fundamental aspects of the algorithm are explained.
  
## Installation

This code release aims for simplicity and self-explainability. 
It should be pretty straightforward to run in Python 3. Python 2 is not supported.

It only needs numpy and [gym[atari]](https://github.com/openai/gym) **[[2](#bibliography)]**, although we also recommend
 installing the Jupyter Notebook for running the [Example.ipynb](https://github.com/FragileTheory/FractalAI/blob/master/Example.ipynb).
notebook.

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

``git clone git@github.com:FragileTheory/FractalAI.git``

``cd FractalAI``

``sudo pip3 install -e .``


## Benchmarks

>It doesn't matter how beautiful your theory is, it doesn't matter how smart you are. 
>
>If it doesn't agree with experiment, it's wrong.
>
> **Richard P. Feynman**

***This section is still work in progress, and it will be updated during the day.***

The following benchmarks have been calculated on a single machine 
([Razer Blade laptop](https://www.razerzone.com/gaming-laptops/razer-blade-pro)) using the
 implementation we provide in this repository. The parameters used were chosen using first principles
 of the theory, and the performance observed corresponds with the expected values.

### FMC Performance table

In the following table we show performance with respect to benchmarks widely accepted in the reinforcement
 learning community. For each game we tried, it displays the following information:
 
- **FMC Score**: This is the maximum scored we achieved in the games we have documented. The number
 of runs for each game, and the parameters used may vary from one game to another.

- **SoTa**: It stands for "State of The Art", and it represents the maximum score achieved by any of
 the following algorithms: Random, DQN, C51 DQN, NoisyNet-DQN, Dueling, NoisyNet-Dueling, A3C,
  NoisyNet-A3C, A2C, HyperNEAT, ES FF, and  MCTS. Some scores are reported as an average across
   multiple runs (at most 100), and it is important to take that into account when interpreting the benchmark.
 **[[1](#bibliography)]**, **[[3](#bibliography)]**, **[[4](#bibliography)]**, **[[5](#bibliography)]**,
  **[[6](#bibliography)]**, **[[7](#bibliography)]**. A detailed source for each score can be 
 found in the performance sheet. 

- **Human**: According to **[[5](#bibliography)]**, this is the mean score achieved by *a
 professional human games tester playing under controlled conditions*.
  
- **Absolute record**: It is the maximum score achieved by a human player as reported in **[[8](#bibliography)]**.

- **MCTS**: Scores achieved using UCT Monte Carlo Tree Search with 3 Million samples per action **[[1](#bibliography)]**.

- **N samples**: This is the mean number of samples that have been used in calculating each action.
 This is, the number of times we called step() on the environment per action.
 
- **% vs SoTa**: Relative performance of FMC vs the State of The Art, according to the formula
 (FMC Score / SoTa) * 100.
   
![Benchmarks](assets/benchmarks.png)  

#### Detailed Google Sheet

In the following Google Sheet we are logging the performance of our Agent relative to the current alternatives.
You can also find all the parameters we used in each of the runs.

If you find that some benchmark is outdated, or you are not capable of replicating some of our results, please
open an issue and we will update the document.

> [Fractal AI performance sheet](https://docs.google.com/spreadsheets/d/1JcNw2L0YL_I2iGZPJ0bNKJshlTaqMuEl5CP2W5zie6M/edit?usp=sharing)

### Please help us running the example, and check by yourself!

Want to help? Once you have a [recorded video](https://github.com/FragileTheory/FractalAI/blob/master/Example.ipynb),
 please write down the parameters you used and the
 mean time displayed as output in the notebook, and share them with us. You can do that in two ways:

- Upload the mp4 generated to YouTube and post it on Twitter. Write the parameters you used and the mean time
 in the same tweet as the video, and use the hashtag **#Fractal_ATARI**. 
  
- Open an issue in this repository and share the data with us.
 
 Thanks!

### Benchmarking tool

We are building a tool that will allow for testing our theory at a bigger scale. We want to provide
confidence intervals using different parameter combinations, and we are working on it.

## Additional Resources

### Theoretical foundations
[Fractal AI: A Fragile Theory of Intelligence](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing):
This document explains the fundamental principles of the Fractal AI theory in which our Agent is based. 
We tried very hard to build our own solution, so we worked all the fundamental principles completely from scratch.
We try to be consistent with existing terminology, and this document should contain everything
 you need to understand the theory. Any comment on how to explain things more clearly will be welcome.

### Blog
 [Sergio's blog: EntropicAI](http://entropicai.blogspot.com.es/):
 Here we have documented and explained the evolution of our research process for developing this algorithm,
 and some experiments where we tried to apply our theory to other research fields.
 
### YouTube

[Fractal AI playlist](https://www.youtube.com/playlist?list=PLEXwXLT-a6beFPzal3OznPQC0pieccAle)
Here you can find some videos of what we accomplished over the years. Besides Atari games, you can find videos 
recorded using a custom library, which can be used to create different task in continuous control environments,
  and visualizations of how the Agent samples the state space.

### Related Papers

[GAS paper](https://arxiv.org/abs/1705.08691) **[[9](#bibliography)]**:
 A manuscript describing an application of our theory to general optimization. 

There are better ways to apply our theory to general optimization, but it is a nice example of why code is better than math to explain our theory. When you try to formalize it, 
it gets really non-intuitive.

[Causal Entropic Forces by Alexander Wissner-Gross](http://alexwg.org/publications/PhysRevLett_110-168702.pdf) **[[10](#bibliography)]**: 
The fundamental concepts behind this paper inspired our research. We develop our theory aiming to calculate future entropy faster,
 and being able to leverage the information contained in the Entropy of any state space, together with any potential function.
 
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

## Cite us

We will include a link to to arXiv in a few days, once it gets published. For now, the 
[pdf file](assets/FractalAI V1.0.pdf) is inside the assets folder.

    @misc{HERE-ARXIV-ID,
        Author = {Sergio Hernández Cerezo and Guillem Duran Ballester},
        Title = {Fractal AI: A fragile theory of intelligence},
        Year = {2018},
        Eprint = {arXiv:HERE-ARXIV-ID},
      }

## FAQ

If there are any questions regarding our methods, we will be answering them here.
 
## About

We have developed this theory for the pleasure of finding thing out as a hobby, while I was at
 college, and Sergio worked as a programmer. Besides the 6 months [Source{d}](https://sourced.tech/) kindly
 sponsored us, we dedicated our personal time to this project making the most efficient use possible
 of the resources we had available.
 
 We are not part of academia, we have no affiliation and no track record. This could not have been
possible without [HCSoft](hcsoft.net), that supported our research, and believed in
our ideas since the very beginning.
 
 We don't have the resources to carry on further our research, but we will gladly accept any contribution or
  sponsorship that allows us to continue working in our passion.

- **[Sergio Hernández Cerezo](https://twitter.com/EntropyFarmer)**: Studied mathematics, works as programmer, dreams about physics.

- **[Guillem Duran Ballester](https://twitter.com/Miau_DB)**: Rogue scientist, loves learning and teaching. Currently looking for an AI-related job.

**Special thanks**: We want to thank all the people who has believed in us during this years.
 Their patience, understanding, and support made possible this work.
 
 - Our families, [HCSoft](hcsoft.net), Guillem's parents: Joan and Francisca, [Eulàlia Veny](https://twitter.com/linguistsmatter), and Fina. 
 
 - The people at sourced, specially [Eiso Kant](https://twitter.com/eisokant), [Waren Long](https://twitter.com/warenlg), [Vadim Markovtsev](https://twitter.com/tmarkhor),
  [Marcelo Novaes](https://twitter.com/marnovo), and [Egor Bulychev](https://twitter.com/egor_bu).
 
 - Everyone who believed in our Alien math since Guillem was in college, specially [José M. Amigó](http://www.umh.es/contenido/pdi/:persona_5536/datos_es.html), [Antoni Elias](https://twitter.com/eliasfuste),
 [Jose Berengueres](https://twitter.com/harriken), [Javier Ozón](https://twitter.com/fjozon), and [Samuel Graván](https://twitter.com/Samuel__GP).
 

## TODO

We are currently working in many improvements to the project, and we will welcome any contribution.

- Improve docstrings and code clarity.

- Add command line interface.

- Upload to pip and Conda.

- Create a Docker container.

- Write an 8 pages paper that details the implementation presented in the repository.
 
## Bibliography
 - **[1]**  Guo, Xiaoxiao and Singh, Satinder and Lee, Honglak and Lewis, Richard L and Wang, Xiaoshi. 
***Deep Learning for Real-Time Atari Game Play Using Offline Monte-Carlo Tree Search Planning***. [NIPS2014_5421](http://papers.nips.cc/paper/5421-deep-learning-for-real-time-atari-game-play-using-offline-monte-carlo-tree-search-planning.pdf), 2014.

- **[2]**  Greg Brockman and Vicki Cheung and Ludwig Pettersson and Jonas Schneider and John Schulman and Jie Tang and Wojciech Zaremba.
***OpenAI Gym*** . [arXiv:1606.01540](https://arxiv.org/pdf/1606.01540.pdf), 2016.

- **[3]**  Marc G. Bellemare, Will Dabney Rémi Munos. ***A Distributional Perspective on Reinforcement Learning***. [arXiv:1707.06887](https://arxiv.org/pdf/1707.06887.pdf), 2017.

- **[4]**  Meire Fortunato, Mohammad Gheshlaghi Azar, Bilal Piot, Jacob Menick, Matteo Hessel, Ian Osband, Alex Graves, Vlad Mnih, Remi Munos, Demis Hassabis,
 Olivier Pietquin, Charles Blundell, Shane Legg. ***Noisy networks for exploration***. [arXiv:1706.10295](https://arxiv.org/pdf/1706.10295.pdf), 2018.
 
- **[5]**  Volodymyr Mnih & others. ***Human-level control through deep reinforcement learning***. [doi:10.1038/nature14236](http://www.davidqiu.com:8888/research/nature14236.pdf), 2015.
 
- **[6]**  Matthias Plappert, Rein Houthooft, Prafulla Dhariwal, Szymon Sidor, Richard Y. Chen, Xi Chen, Tamim Asfour, Pieter Abbeel, Marcin Andrychowicz.
***Parameter Space Noise for Exploration***. [arXiv:1706.01905](https://arxiv.org/abs/1706.01905), 2017.

- **[7]**  Tim Salimans, Jonathan Ho, Xi Chen, Szymon Sidor, Ilya Sutskever.
***Evolution Strategies as a Scalable Alternative to Reinforcement Learning***. [arXiv:1703.03864](https://arxiv.org/abs/1703.03864), 2017.

- **[8]**  ***ATARI VCS/2600 Scoreboard***. [Atari compendium](http://www.ataricompendium.com/game_library/high_scores/high_scores.html), 2018.

- **[9]**  Sergio Hernández, Guillem Duran, José M. Amigó. ***General Algorithmic Search***. [arXiv:1705.08691](https://arxiv.org/abs/1705.08691), 2017.

- **[10]**  Alexander Wissner-Gross. ***Causal entropic forces***. [Physical Review Letters](http://alexwg.org/publications/PhysRevLett_110-168702.pdf), 2013.

- **[11]**  Shane Legg ***Machine Super Intelligence***. [Doctoral Dissertation ](http://www.vetta.org/documents/Machine_Super_Intelligence.pdf), 2008.

