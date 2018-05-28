# Fractal AI: A Fragile Theory of Intelligence

**Please note this project is under active development and may change over time.
Treat it as an open beta.**

![Boxing-v0](assets/boxing.gif "Boxing-v0 76 samples per action")
![MsPacman-v0](assets/mspacman.gif "MsPacman-v0 154 samples per action")
![Tennis-v0](assets/tennis.gif "Tennis-v0 1250 samples per action")
![Centipede-v0](assets/centipede.png "Centipede-v0 1960 samples per action")
![MontezumaRevenge-v0](assets/montezuma.gif "MontezumaRevenge-v0 5175 samples per action")

>Once you start doubting, just like you’re supposed to doubt, you ask me if the science is true.
>You say no, we don’t know what’s true, we’re trying to find out and everything is possibly wrong.

_–Richard P. Feynman, **The Pleasure of Finding Things Out**: The Best Short Works of Richard P. Feynman._

## Table of Contents

- [Abstract](#abstract)
- [Quick Start](#quick-start)
- [Running the code](#installation)
- [Benchmarks](#benchmarks)
- [Additional Resources](#additional-resources)
  - [Theoretical foundations](#theoretical-foundations)
  - [Blog](#blog)
  - [YouTube](#youtube)
  - [Related research](#related-research)
- [Cite us](#cite-us)
- [FAQ](#faq)
- [About the Authors](#about-the-authors)
- [Todo](#todo)
- [Bibliography](#bibliography)

## Abstract

[Fractal AI](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing)
([arXiv](https://arxiv.org/abs/1803.05049)) is a theory for efficiently sampling state spaces.
It allows one to derive new mathematical tools that could be useful for modeling information
using cellular automaton-like structures instead of smooth functions.

In this repository we present a new agent called [Fractal Monte Carlo Agent](fractalai/fractalmc.py),
derived from the first principles of the theory. The agent is capable of solving Atari games
under the [OpenAI Gym](https://github.com/openai/gym) several [orders of magnitude more efficiently](#benchmarks)
than similar techniques, such as _Monte Carlo Tree Search (MCTS)_ **[[1](#bibliography)]**.

We also present the [Swarm Wave algorithm](fractalai/swarm_wave.py),
a tool derived from Fractal AI (FAI) that allows one to solve Markov decision processes under a
perfect model of the environment. Under this assumption, the algorithm is about five orders of
magnitude more efficient than MCTS, effectively "solving" a substantial number of Atari games.

The code provided under this repository exemplifies how it is now possible to beat
some of the current state-of-the-art benchmarks on Atari games and a large sample of top-performing examples
with little computation required, turning Reinforcement Learning (RL) into a supervised problem.

The algorithms propose a new approach to modeling the decision space, while maintaining
control over any aspects of the agent's behavior. The algorithms can be applied
to both sampling discrete and continuous state spaces.

## Quick Start

To quickly understand the fundamentals of Fractal AI you can refer to the [Introduction to FAI](introduction_to_fai.md).
The document provides a brief explanation of the algorithms here presented and their
potential applications on the field of Reinforcement Learning.

To proof how how the Fractal Monte Carlo Agent performs on any Atari game you can refer to the [FMC example notebook](FMC_example.ipynb).
The example allows runs either using the RAM content or pixel render as observations.

To better understand how the Swarm Wave algorithm works in practice you can refer to the [Swarm Wave example notebook](Swarm_Wave_example.ipynb).

Pleas note [the authors](#about-the-authors) are open to discuss the ideas and code here presented under the
conceptual framework of Reinforcement Learning and its standard terminology.

## Installation

The code provided aims to be both simple and self-explanatory.
Requirements and instructions to set up the environment are provided below.

### Requirements

- Python 3. Python 2 is not supported nor currently expected to be supported.
- Python [numpy library](http://docs.scipy.org/doc/numpy/reference/?v=20180402183410).
- Python [OpenAI Gym [Atari]](https://github.com/openai/gym) **[[2](#bibliography)]**.
  - OpenAI Gym dependencies.
- _(Optional)_ [Jupyter Notebook](http://www.jupyter.org) for running the example notebooks provided.

### Installing dependencies

As a first step, install the dependencies as explained on the OpenAI gym documentation:

>To install the full set of environments, you'll need to have some system
>packages installed. We'll build out the list here over time; please let us know
>what you end up installing on your platform.
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

### Cloning and Installing the FractalAI Repository

On the terminal, run:

```bash
git clone git@github.com:FragileTheory/FractalAI.git
cd FractalAI
sudo pip3 install -e .
```

## Benchmarks

>It doesn't matter how beautiful your theory is, it doesn't matter how smart you are.
>
>If it doesn't agree with experiment, it's wrong.

_–Richard P. Feynman_

The results for the benchmarks provided come from a single ([Razer Blade laptop](https://www.razerzone.com/gaming-laptops/razer-blade-pro))
using the Fractal Monte Carlo Agent (FMC) through the code from the reference implementation provided in this repository.
The parameters utilized were chosen according to the the first principles of the theory.
The performance observed corresponds with the expected values.

| Total number of games available               |  55 |  |
| :-------------------------------------------- | --: | ------: |
| FMC better than avg. human                    |  52 |   94.55% |
| FMC better than the state-of-the-art          |  48 |   87.27% |
| Games solved or defeats the human record      |  31 |   56.36% |

### Fractal Monte Carlo Agent Performance Table

The following table depicts the Fractal Monte Carlo Agent performance based on
widely accepted benchmarks within the reinforcement learning community.

For each game played by the agent, the following information is is provided:

- **Human record**: the maximum score achieved by a human player, as reported in **[[8](#bibliography)]**.
- **SOtA**: stands for "State of the Art":
  - Represents the best score achieved by any of the following algorithms:
    Random, DQN, C51 DQN, NoisyNet-DQN, Dueling, NoisyNet-Dueling, A3C,
    NoisyNet-A3C, A2C, HyperNEAT, ES FF, and MCTS.
  - Take into account when interpreting the benchmark table that some of the scores
    are reported as an average across multiple runs (at most 100).
    See **[[1](#bibliography)]**, **[[3](#bibliography)]**, **[[4](#bibliography)]**,
    **[[5](#bibliography)]**, **[[6](#bibliography)]**, **[[7](#bibliography)]**.
    A detailed source for each score can be found in the [detailed performance sheet](#detailed-performance-sheet).
- **FMC**: the maximum score achieved by Fractal Monte Carlo Agent in the games documented.
  The number of runs for each game and the parameters used may vary between each game.
- **FMC vs SOtA**: Relative performance of _FMC_ vs. the corresponding _State of the Art_.

| Game | Human Record | SOtA | FMC | FMC vs SOtA |
|:--- | :---: | :---: | :---: | ---:|
 | alien | 251916 | 5899 | ***479940*** | 8136% | 
 | amidar | 155339 | 2354 | ***5779*** | 245% | 
 | assault | 8647 | 11477 | ***14472*** | 126% | 
 | asterix | 335500 | 406211 | ***999500*** | 246% | 
 | asteroids | 10004100 | 26380 | ***12575000*** | 47669% | 
 | atlantis | 7352737 | 8782433 | ***10000100*** | 114% | 
 | bank heist | 199978 | 1611.9 | ***3139*** | 195% | 
 | battle zone (*) | 863000 | 42767 | ***999000*** | 2336% | 
 | beam rider | 999999 | 30276.5 | ***999999*** | 3303% | 
 | berzerk | 1057940 | 3409 | ***17610*** | 517% | 
 | bowling | 300 | 135.8 | ***180*** | 133% | 
 | boxing | 100 | 99.4 | ***100*** | 101% | 
 | breakout | 752 | 748 | ***864*** | 116% | 
 | centipede | 1301709 | 25275.2 | ***1351000*** | 5345% | 
 | chopper command (*) | 999900 | 15600 | ***999900*** | 6410% | 
 | crazy climber | 447000 | 179877 | ***2254100*** | 1253% | 
 | demon attack (*) | 1556345 | 130955 | ***999970*** | 764% | 
 | double dunk | 199 | 5 | ***24*** | 480% | 
 | enduro | 3617.9 | 3454 | ***5279*** | 153% | 
 | fishing derby | 71 | 49.8 | ***63*** | 127% | 
 | freeway | 34 | ***33.9*** | 33 | 97% | 
 | frostbite (*) | 552590 | 7413 | ***999960*** | 13489% | 
 | gopher (*) | 120000 | 104368.2 | ***999980*** | 958% | 
 | gravitar | 1673950 | 1693.2 | ***14050*** | 830% | 
 | hero | 1000000 | ***105929.4*** | 43255 | 41% | 
 | ice hockey | 36 | 10.6 | ***64*** | 604% | 
 | jamesbond | 45550 | 4214 | ***152950*** | 3630% | 
 | kangaroo | 1436500 | ***14854*** | 10800 | 73% | 
 | krull | 1006680 | 12601.4 | ***426534*** | 3385% | 
 | kung fu master | 1000000 | 48375 | ***172600*** | 357% | 
 | montezuma | 1219200 | 4739.6 | ***5600*** | 118% | 
 | ms pacman (*) | 290090 | 6283 | ***999990*** | 15916% | 
 | name this game | 25220 | 15572.5 | ***53010*** | 340% | 
 | phoenix | 4014440 | 70324.3 | ***250450*** | 356% | 
 | pitfall | 114000 | ***123*** | 0.001 | 0% | 
 | pong | 21 | ***21*** | ***21*** | 100% | 
 | private eye | 103100 | 40908.2 | ***41760*** | 102% | 
 | qbert (*) | 2400000 | 23784 | ***999975*** | 4204% | 
 | riverraid | 194940 | ***21162.6*** | 18510 | 87% | 
 | road runner (*) | 2038100 | 69524 | ***999900*** | 1438% | 
 | robotank | 74 | 65.3 | ***94*** | 144% | 
 | seaquest (*) | 527160 | 266434 | ***999999*** | 375% | 
 | skiing | -3272 | ***-7983.6*** | -99999 | 1253% | 
 | solaris | 281740 | 11830 | ***93520*** | 791% | 
 | space invaders | 621535 | 15311.5 | ***17970*** | 117% | 
 | star gunner (*) | 77400 | 125117 | ***999800*** | 799% | 
 | tennis | 24 | 23.1 | ***24*** | 104% | 
 | time pilot | 66500 | 11666 | ***90000*** | 771% | 
 | tutankham | 3493 | 321 | ***342*** | 107% | 
 | up n down (*) | 168830 | 145113 | ***999999*** | 689% | 
 | venture | 31900 | ***3800*** | 1500 | 39% | 
 | video pinball (*) | 91862206 | 949604 | ***999999*** | 105% | 
 | wizard of wor (*) | 233700 | 12352 | ***99900*** | 809% | 
 | yars revenge | 15000105 | 69618.1 | ***98491*** | 141% | 
 
 (*) Games with the "1 Million bug" where max. score is hard-limited
 
#### Detailed Performance Sheet

We provide a more detailed Google Docs spreadsheet where the performance of the
Fractal Monte Carlo Agent is logged relative to the current alternatives.
In the spreadsheet we also provide the parameters used in each of the runs.

If you find any outdated benchmarks or for some reaons you are unable to replicate
some of our results, please [open an issue](https://github.com/FragileTheory/FractalAI/issues)
and we will update the document accordingly.

- [Fractal AI performance sheet](https://docs.google.com/spreadsheets/d/1JcNw2L0YL_I2iGZPJ0bNKJshlTaqMuEl5CP2W5zie6M/edit?usp=sharing)

### Benchmarking tool

We are currently building a tool that will allow for testing our theory at larger scale.
We aim to provide confidence intervals for the results, by using different parameter combinations.

## Additional Resources

### Theoretical Foundations

[Fractal AI: A Fragile Theory of Intelligence](https://docs.google.com/document/d/13SFT9m0ERaDY1flVybG16oWWZS41p7oPBi3904jNBQM/edit?usp=sharing):
This document explains the fundamental principles of the Fractal AI theory in which our Agent is based.
We worked all the fundamental principles completely from scratch to build our own solution.
We try to be consistent with existing terminology, and this document should contain everything
you need to understand the theory. Comments on how to better explain the content are appreciated.

### Blog

 [EntropicAI, Sergio Hernández Cerezo's blog](http://entropicai.blogspot.com/):
 Here you can find the evolution of the research process for developing this algorithm,
 documented and explained, as well as experiments which aim to apply the theory to other fields of research.

### YouTube

[Fractal AI playlist](https://www.youtube.com/playlist?list=PLEXwXLT-a6beFPzal3OznPQC0pieccAle):
In the Youtube playlist you can find videos of the accomplishments over the years.
Besides the recordings Atari games using the Agent, you can find videos recorded using
a custom library that allows one to create different tasks in continuous control environments,
as well as visualizations of how the Agent samples the state space.

### Related Research

[GAS paper](https://arxiv.org/abs/1705.08691) **[[9](#bibliography)]**:
A manuscript describing an application of the Fractal AI theory on general optimization problems.
There are certainly better ways to apply the theory such problems, yet it illustrates why
code is better than maths to explain the theory. When trying to formalize it,
things can get really non-intuitive.

[Causal Entropic Forces by Alexander Wissner-Gross](http://alexwg.org/publications/PhysRevLett_110-168702.pdf) **[[10](#bibliography)]**:
The fundamental concepts behind this paper inspired the present research.
We develop our theory aiming to calculate future entropy more quickly and being able to
leverage the information contained in the Entropy of any state space, together with any potential function.

## Cite us

    @misc{1803.05049,
        Author = {Sergio Hernández Cerezo and Guillem Duran Ballester},
        Title = {Fractal AI: A fragile theory of intelligence},
        Year = {2018},
        Eprint = {arXiv:1803.05049},
      }

## FAQ

As questions regarding the research and methodology we will address them under the FAQ.

You can refer to the [FAQ document](FAQ.md).

## About the Authors

Authors:

- **[Sergio Hernández Cerezo](https://twitter.com/EntropyFarmer)**: Studied mathematics, works as programmer, dreams about physics.
- **[Guillem Duran Ballester](https://twitter.com/Miau_DB)**: Rogue scientist, loves learning and teaching. Currently looking for work opportunities related to AI.

The authors have developed the theory as personal side projects driven purely by intellectual curiosity.
Guillem worked on it while attending college, and Sergio while working as a programmer.
The authors are not part of academia, have no corporate affiliation and no formal track record.

All the time and resources involved came from the authors themselves, besides the support from:

- [HCSoft](http://hcsoft.net), which supported our research and believed the ideas since the very beginning.
- [source{d}](https://sourced.tech/), which kindly sponsored the project for 6 months.

We currently do not have the resources to further carry our research. We will gladly accept
contributions or sponsorships that allow us to continue working with what is our passion.

**Special thanks**: We want to thank all the people who has believed in us along the years.
Their patience, understanding and support made possible for this project to evolve to this point.

- Our families, [HCSoft](hcsoft.net), Guillem's parents: Joan and Francisca, [Eulàlia Veny](https://twitter.com/linguistsmatter), and Fina.
- The people at source{d}, specially [Eiso Kant](https://twitter.com/eisokant), [Waren Long](https://twitter.com/warenlg),
  [Vadim Markovtsev](https://twitter.com/tmarkhor), [Marcelo Novaes](https://twitter.com/marnovo),
  and [Egor Bulychev](https://twitter.com/egor_bu).
- Everyone who believed in our "alien math" since Guillem was in college, specially
  [José M. Amigó](http://www.umh.es/contenido/pdi/:persona_5536/datos_es.html), [Antoni Elias](https://twitter.com/eliasfuste),
  [Jose Berengueres](https://twitter.com/harriken), [Javier Ozón](https://twitter.com/fjozon),
  [Samuel Graván](https://twitter.com/Samuel__GP), and [Marc Garcia](https://twitter.com/datapythonista).

## TODO

We are actively working in improving this project, and we welcome all contributions.
Some of the to-dos in our roadmap:

- Making the repository more friendly to academia.
- Improving the Introduction to Fractal AI document.
- Improving code clarity and docstrings.
- Updating Benchmarks with current records.
- Providing a command line interface (CLI).
- Uploading the project to pip and Conda package managers.
- Creating a Docker container for ease of use.

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
