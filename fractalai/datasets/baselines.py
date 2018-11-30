# This is miscellaneous stuff to hack the OpenAI baselines package.
# Using this code you can substitute the OpenAI gym environments with an environment that outputs
# games played by a Swarm. This way you don't need to care about being lucky in order to find
# good examples.

import os
import numpy as np
from gym.envs.registration import registry as gym_registry
from fractalai.datasets.mlswarm import MLWave
from fractalai.model import RandomDiscreteModel
from fractalai.datasets.data_env import DataVecEnv
from fractalai.environment import AtariFAIWrapper, AtariEnvironment

from baselines import logger
from baselines.bench import Monitor
from baselines.common import set_global_seeds
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from baselines.common.atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    ClipRewardEnv,
    ScaledFloatFrame,
    FrameStack,
    WarpFrame,
    EpisodicFrameEnv,
    NormRewardEnv,
    EpisodicRewardEnv,
)

swarm_kwargs = dict(
    dt_mean=10,  # Apply the same action n times in average
    dt_std=5,  # Repeat same action a variable number of times
    min_dt=5,  # Minimum number of consecutive steps to be taken
    samples_limit=300000,  # 200000  # Maximum number of samples allowed
    reward_limit=5000,  # Stop the sampling when this score is reached
    n_walkers=50,  # Maximum width of the tree containing al the trajectories
    render_every=100,  # print statistics every n iterations.
    balance=2,  # Balance exploration vs exploitation
    save_data=True,  # Save the data generated
    accumulate_rewards=True,
    prune_tree=True,
)

generator_kwargs = {}

frame_skip = 4


def make_atari(env_id):
    spec = gym_registry.spec(env_id)
    # not actually needed, but we feel safer
    spec.max_episode_steps = None
    spec.max_episode_time = None
    env = spec.make()
    assert "NoFrameskip" in env.spec.id
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    return env


def wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):
    """Configure environment for DeepMind-style Atari.
    """
    if episode_life:
        env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def make_env(env_id, wrapper_kwargs):
    env = make_atari(env_id)
    return wrap_deepmind(env, **wrapper_kwargs)


def make_atari_env(
    env_id,
    num_env,
    seed,
    wrapper_kwargs=None,
    start_index=0,
    n_actors: int = 8,
    folder=None,
    mode="online",
    swarm_kwargs=swarm_kwargs,
    generator_kwargs=generator_kwargs,
):
    wrapper_kwargs = wrapper_kwargs if wrapper_kwargs is not None else {}

    def env_callable():
        return AtariEnvironment(name=env_id, min_dt=frame_skip, obs_ram=True)

    def data_env_callable():
        env = AtariFAIWrapper(make_env(env_id, wrapper_kwargs))
        env.reset()
        return env

    def model_callable():
        return RandomDiscreteModel(n_actions=int(make_env(env_id, wrapper_kwargs).action_space.n))

    denv = DataVecEnv(
        num_envs=num_env,
        n_actors=n_actors,
        swarm_class=MLWave,
        env_callable=env_callable,
        model_callable=model_callable,
        swarm_kwargs=swarm_kwargs,
        generator_kwargs=generator_kwargs,
        data_env_callable=data_env_callable,
        seed=seed,
        folder=folder,
        mode=mode,
    )
    return denv


def wrap_modified_rr(
    env,
    episode_life=True,
    episode_reward=False,
    episode_frame=False,
    norm_rewards=True,
    frame_stack=False,
    scale=False,
):
    """Configure environment for DeepMind-style Atari modified as described in RUDDER paper;
    """
    if episode_life:
        print("Episode Life")
        env = EpisodicLifeEnv(env)
    if episode_reward:
        print("Episode Reward")
        env = EpisodicRewardEnv(env)
    if episode_frame:
        print("Episode Frame")
        env = EpisodicFrameEnv(env)
    _ori_r_games = [
        "DoubleDunk",
        "Boxing",
        "Freeway",
        "Pong",
        "Bowling",
        "Skiing",
        "IceHockey",
        "Enduro",
    ]
    original_reward = any([game in env.spec.id for game in _ori_r_games])

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if norm_rewards and not original_reward:
        print("Normalizing reward....")
        env = NormRewardEnv(env, 100.0)
    else:
        print("Normal reward")
    if frame_stack:
        env = FrameStack(env, 4)
    return env


def make_rude(env_id, wrapper_kwargs):
    env = make_atari(env_id)
    return wrap_modified_rr(env, **wrapper_kwargs)


def make_rude_env(
    env_id, num_env, seed: int = 1, wrapper_kwargs=None, start_index=0, n_actors: int = 8
):
    wrapper_kwargs = wrapper_kwargs if wrapper_kwargs is not None else {}

    def env_callable():
        return AtariEnvironment(name=env_id, min_dt=frame_skip, obs_ram=True)

    def data_env_callable():
        env = AtariFAIWrapper(make_rude(env_id, wrapper_kwargs))
        env.reset()
        return env

    def model_callable():
        return RandomDiscreteModel(n_actions=int(make_rude(env_id, wrapper_kwargs).action_space.n))

    denv = DataVecEnv(
        num_envs=num_env,
        n_actors=n_actors,
        swarm_class=MLWave,
        env_callable=env_callable,
        model_callable=model_callable,
        swarm_kwargs=swarm_kwargs,
        generator_kwargs=generator_kwargs,
        data_env_callable=data_env_callable,
        seed=seed,
    )
    return denv


def _make_atari_env(env_id, num_env, seed, wrapper_kwargs=None, start_index=0):
    """
    Create a wrapped, monitored SubprocVecEnv for Atari. This is the same as the one used
     in OpenAI Baselines.
    """
    from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):  # pylint: disable=C0111
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            env = Monitor(
                env,
                logger.get_dir() and os.path.join(logger.get_dir(), str(rank)),
                allow_early_resets=True,
            )
            return wrap_deepmind(env, **wrapper_kwargs)

        return _thunk

    set_global_seeds(seed)
    return SubprocVecEnv([make_env(i + start_index) for i in range(num_env)])


class A2CTester:
    """This class tests the network used in an A2C agent to see how it performs in the environment.
    """

    def __init__(
        self,
        model,
        env_id,
        num_env: int = 4,
        seed: int = 1,
        wrapper_kwargs=None,
        start_index=0,
        stack_frames: int = 4,
    ):
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        wrapper_kwargs["episode_life"] = False
        self.env = VecFrameStack(
            _make_atari_env(env_id, num_env, seed, wrapper_kwargs, start_index), stack_frames
        )
        self.model = model
        self.end_ix = np.zeros(num_env, dtype=bool)
        self.states = model.initial_state
        self.obs = None
        self.dones = None

    def end_condition(self):
        self.end_ix = np.logical_or(self.end_ix, self.dones)
        return self.end_ix.all()

    def play_game(self):

        self.obs = self.env.reset()
        self.end_ix = np.zeros(len(self.obs), dtype=bool)
        self.dones = np.zeros(len(self.obs), dtype=bool)
        total_rewards = []
        total_len = []
        current_rewards = np.zeros(len(self.obs))
        game_len = 1
        while not self.end_condition():
            try:
                actions, values, states, _ = self.model.step(self.obs, self.states, self.dones)

                obs, rewards, dones, infos = self.env.step(actions)
                game_len += 1
                current_rewards += rewards
                self.states = states
                self.dones = dones
                for n, done in enumerate(dones):
                    if done:
                        self.obs[n] = self.obs[n] * 0

                        total_rewards.append(float(current_rewards[n]))
                        total_len.append(game_len)
                self.obs = obs
            except RuntimeError:
                break
        return total_rewards, total_len
