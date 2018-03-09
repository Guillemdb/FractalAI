# Fractalai: An approach to General Artificial Intelligence

### Sergio Hernández Cerezo and Guillem Duran Ballester
Under construction
## Abstract

Fractal AI is a theoretical framework for general artificial intelligence in any kind of Markov decision processes.
In this repository we are presenting a new Agent, which is capable of solving Atari games
several orders of magnitude more efficiently than similar techniques, like Monte Carlo Tree Search. 

The code provided shows how it is now viable to beat some of the current state of the art benchmarks on Atari games,
using a less than 1000 samples to calculate one decision. Fractal AI makes it possible to generate a huge database of
 top performing examples with very little amount of computation required, transforming Reinforcement Learning into a 
 supervised problem.
 
 The algorithm presented is capable of solving the exploration vs exploitation dilemma, while
 maintaining control over any aspect of the behavior of the Agent. From a mathematical perspective, 
 Fractal AI also offers a new way of measuring intelligence, and complexity in any kind of state space, 
 thus giving rise to a new kind of risk control theory.

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
|alien         |19380.0  |271.89%     |NaN|1190.0|5.0|25.0|2000.0|50.0|
|amidar        |4306.0   |194.40%     |NaN|1222.0|5.0|25.0|2000.0|50.0|
|assault       |1280.0   |17.06%      |NaN|1317.0|5.0|25.0|2000.0|50.0|
|asteroids     |76270.0  |160.94%     |NaN|2733.0|2.0|20.0|5000.0|NaN|
|beam rider    |2160.0   |12.64%      |29.86%|4052.0|5.0|25.0|2000.0|100.0|
|boxing        |100.0    |104.17%     |NaN|2027.0|3.0|30.0|2000.0|150.0|
|breakout      |36.0     |7.98%       |8.87%|5309.0|5.0|30.0|20000.0|150.0|
|centipede     |529355.0 |4,405.05%   |NaN|1960.0|1.0|20.0|6000.0|100.0|
|double dunk   |20.0     |400.00%     |NaN|5327.0|3.0|50.0|8000.0|NaN|
|enduro        |471.0    |28.05%      |59.77%|NaN|5.0|15.0|2000.0|50.0|
|ice hockey    |52.0     |5,200.00%   |NaN|12158.0|3.0|50.0|20000.0|250.0|
|ms pacman     |58521.0  |372.91%     |NaN|5129.0|10.0|20.0|20000.0|250.0|
|qbert         |35750.0  |189.66%|189.66%|2728.0|5.0|20.0|5000.0|NaN|
|seaquest      |3180.0   |7.56%|97.64%|6252.0|5.0|25.0|20000.0|250.0|
|space invaders|3605.0   |80.29%|153.14%|4261.0|2.0|35.0|6000.0|NaN|
|tennis        |16.0     |177.78%|NaN|2437.0|4.0|30.0|5000.0|NaN|
|video pinball |496681.0 |64.64%|NaN|1083.0|2.0|30.0|5000.0|NaN|
|wizard of wor |93090.0  |785.44%|NaN|2229.0|4.0|35.0|5000.0|NaN|

In the following Google Sheet you we are logging the performance of our Agent relative to the current alternatives.
If you find that some benchmark is outdated, or you are not capable of replicating any of our results, please
open an issue and we will update the document.



[Fractal AI Performance](https://docs.google.com/spreadsheets/d/1JcNw2L0YL_I2iGZPJ0bNKJshlTaqMuEl5CP2W5zie6M/edit?usp=sharing)



## Resources


## Installation

It only needs numpy and [gym](https://github.com/openai/gym).


## Benchmarks

[Google spreadsheet](https://docs.google.com/spreadsheets/d/1JcNw2L0YL_I2iGZPJ0bNKJshlTaqMuEl5CP2W5zie6M/edit?usp=sharing)
## Run the example notebook to check out by yourself
