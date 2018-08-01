# This is an interface that mimics the OpenaAI baselines vectorized envs, and OpenAI environments.
# The cool thing about it is that you now have an environment that automagically plays good games,
# so you you have full control over how you explore the state space of any problem.

from typing import Callable
import copy
import numpy as np
from gym import Env
from baselines.common.vec_env import VecEnv
from fractalai.datasets.ray import ParallelDataGenerator


class DataEnv(Env):

    def __init__(self, n_actors: int, swarm_class: Callable,
                 env_callable: Callable, model_callable: Callable,
                 swarm_kwargs: dict, generator_kwargs: dict, data_env_callable: Callable=None,
                 seed: int=1):
        self.use_data_env = data_env_callable is not None
        self.generator = ParallelDataGenerator(n_actors=n_actors, swarm_class=swarm_class,
                                               data_env_callable=data_env_callable,
                                               swarm_kwargs=swarm_kwargs,
                                               env_callable=env_callable,
                                               model_callable=model_callable,
                                               generator_kwargs=generator_kwargs, seed=seed)
        self.env = env_callable() if data_env_callable is None else data_env_callable()
        self._ix = 0
        self.game_stream = self.generator.game_stream(examples=False, full_game=self.use_data_env)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.current_game = None
        self.observs = None
        self.actions = None
        self.rewards = None
        self.ends = None
        self.new_obs = None
        self.states = None
        self.infos = None

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self):
        return self.env.env.unwrapped

    def step(self, action):
        if self.use_data_env:
            obs, reward, end, info, action = self.get_example()
            info = copy.copy(info)
            info["action"] = copy.copy(action)
            return obs, reward, end, info
        else:
            obs, action, reward, new_ob, end = self.get_example()
            info = {"action": copy.copy(action)}
            return new_ob, reward, end, info

    def load_new_game(self):
        self._ix = -1
        if self.use_data_env:
            self.states, self.observs, self.rewards, \
              self.ends, self.infos, self.actions = next(self.game_stream)
        else:
            self.states, self.observs, self.actions, self.rewards,\
                self.new_obs, self.ends = next(self.game_stream)

    def get_example(self):
        if self._ix < len(self.rewards) - 1:
            self._ix += 1
            end = self.ends[self._ix]
        else:
            end = True
        i = self._ix
        if self.use_data_env:
            return self.observs[i], self.rewards[i], end, self.infos[i], self.actions[i]
        else:
            return self.observs[i], self.actions[i], self.rewards[i], self.new_obs[i], end

    def reset(self):
        self.load_new_game()
        if self.use_data_env:
            obs, reward, end, info, action = self.get_example()
            # info = copy.copy(info)
            # info["action"] = copy.copy(action)
            return obs  # , reward, end, info
        else:
            obs, action, reward, new_ob, end = self.get_example()
            # info = {"action": action}
            return new_ob  # , reward, end, info

    def render(self, mode='human'):
        self.env.set_state(self.states[self._ix].copy())
        self.env.step(self.actions[self._ix])
        self.env.render()

    def seed(self, seed=None):
        self.env.env.seed(seed)


class DataVecEnv(VecEnv):

    def __init__(self, num_envs: int, n_actors: int, swarm_class: Callable,
                 env_callable: Callable, model_callable: Callable,
                 swarm_kwargs: dict, generator_kwargs: dict, data_env_callable: Callable=None,
                 seed: int=1, folder=None, mode="online"):
        self.use_data_env = data_env_callable is not None
        self.generator = ParallelDataGenerator(n_actors=n_actors, swarm_class=swarm_class,
                                               data_env_callable=data_env_callable,
                                               swarm_kwargs=swarm_kwargs,
                                               env_callable=env_callable,
                                               model_callable=model_callable,
                                               generator_kwargs=generator_kwargs, seed=seed,
                                               folder=folder, mode=mode)
        self.env = env_callable() if data_env_callable is None else data_env_callable()
        self._ix = 0
        self.game_stream = self.generator.game_stream(examples=False, full_game=self.use_data_env)
        super(DataVecEnv, self).__init__(observation_space=self.env.observation_space,
                                         action_space=self.env.action_space, num_envs=num_envs)
        self.games = {i: None for i in range(self.num_envs)}
        self._indexes = np.zeros(self.num_envs, dtype=int)

    def __getattr__(self, item):
        return getattr(self.env, item)

    def get_example(self, ix):
        if self.use_data_env:
            _, observs, rewards, ends, infos, actions = self.games[ix]
        else:
            _, observs, actions, rewards, new_obs, ends = self.games[ix]
        if self._indexes[ix] < len(rewards) - 1:
            self._indexes[ix] = self._indexes[ix] + 1
            end = ends[self._indexes[ix]]
        else:
            end = True
        i = self._indexes[ix]
        if self.use_data_env:
            return observs[i], rewards[i], end, infos[i], actions[i]
        else:
            return observs[i], actions[i], rewards[i], new_obs[i], end

    def get_examples(self):
        observs = []
        rewards = []
        ends = []
        infos = []
        for i, game in self.games.items():
            if self.use_data_env:
                obs, reward, end, info, act = self.get_example(i)
                info["action"] = act
                observs.append(obs)
                rewards.append(reward)
                ends.append(end)
                infos.append(info)
            else:
                obs, act, reward, new_obs, end = self.get_example(i)
                info = {"action": act}
                observs.append(new_obs)
                rewards.append(reward)
                ends.append(end)
                infos.append(info)
        return np.array(observs), np.array(rewards), np.array(ends), infos

    def reset(self):
        self.games = {i: next(self.game_stream) for i in range(self.num_envs)}
        obs, rewards, ends, infos = self.get_examples()
        return obs

    def autoreset(self):
        for key, val in self.games.items():
            if self._indexes[key] >= len(self.games[key][3]) - 1:
                self._indexes[key] = 0
                self.games[key] = next(self.game_stream)

    def step_async(self, actions):
        pass

    def step_wait(self):
        self.autoreset()
        return self.get_examples()

    def close(self):
        pass

    def render(self, mode='human'):
        self.env.set_state(self.games[0][0][self._indexes[0]].copy())
        if self.use_data_env:
            self.env.step(self.games[0][-1][self._indexes[0]])
        else:
            self.env.step(self.games[0][2][self._indexes[0]])
        self.env.render()

