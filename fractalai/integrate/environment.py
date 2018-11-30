from typing import Callable
import numpy as np
from gym import spaces
from fractalai.environment import Environment


class FunctionEnvironment(Environment):
    def __init__(self, target_func: Callable, n_dims, max_x: int = 100, min_x: int = -100):
        super(FunctionEnvironment, self).__init__(name=target_func.__name__)
        self.function = target_func
        self.state = np.zeros(self.n_dims)
        self.min_x = min_x
        self.max_x = max_x
        self.n_dims = n_dims
        self.action_space = spaces.Box(low=-1, high=1, shape=(n_dims,))

    def step(self, action, state=None, n_repeat_action: int = 1):
        if state is not None:
            self.set_state(state)
        reward = 0
        old_reward = 0
        end = False
        for i in range(max(1, n_repeat_action)):
            new_state = self.state + action
            new_reward = self.function(self.state[:-1])
            reward += new_reward - old_reward
            old_reward = new_reward
            _end = np.logical_not(self.min_x < new_state < self.max_x)
            end = end or _end
        return new_state, reward, end, {}

    def step_batch(self, actions, states=None, n_repeat_action: [int, np.ndarray] = None) -> tuple:
        """
        Take a step on a batch of states.
        :param actions: Chosen action applied to the environment.
        :param states: Set the environment to the given state before stepping it.
        :param n_repeat_action: Consecutive number of times that the action will be applied.
        :return:
        """
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        n_repeat_action = (
            n_repeat_action
            if isinstance(n_repeat_action, np.ndarray)
            else np.ones(len(states)) * n_repeat_action
        )
        data = [
            self.step(action, state, n_repeat_action=dt)
            for action, state, dt in zip(actions, states, n_repeat_action)
        ]
        new_states, observs, rewards, terminals, lives = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            lives.append(info)
        if states is None:
            return observs, rewards, terminals, lives
        else:
            return new_states, observs, rewards, terminals, lives

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state

    def reset(self):
        self.state = np.zeros(self.n_dims)


class MapFunctionEnvironment(Environment):
    def __init__(
        self, target_func: Callable, init_state, max_x: int = 100, min_x: int = -100, stepsize=0.1
    ):
        super(MapFunctionEnvironment, self).__init__(name=target_func.__name__)
        self.function = target_func
        self.state = init_state
        self.min_x = min_x
        self.max_x = max_x
        self.n_dims = len(init_state)
        self.action_space = spaces.Box(low=-stepsize, high=stepsize, shape=(len(init_state),))

    def step(self, action, state=None, n_repeat_action: int = 1):
        if state is not None:
            self.set_state(state)
        end = False
        mask = np.zeros(action.shape, dtype=int)
        indexes = np.random.choice(np.arange(len(action)), size=n_repeat_action)
        mask[indexes] = 1
        new_state = np.clip(self.state + action * mask, self.min_x, self.max_x)
        reward = self.function(np.array([self.state]))
        if state is None:
            return new_state, reward, end, {"terminal": False, "lives": 0}
        else:
            return new_state, new_state, reward, end, {"terminal": False, "lives": 0}

    def step_batch(self, actions, states=None, n_repeat_action: [int, np.ndarray] = None) -> tuple:
        """
        Take a step on a batch of states.
        :param actions: Chosen action applied to the environment.
        :param states: Set the environment to the given state before stepping it.
        :param n_repeat_action: Consecutive number of times that the action will be applied.
        :return:
        """
        if states is None:
            raise ValueError
        mask = np.zeros(actions.shape, dtype=int)
        if isinstance(n_repeat_action, int):
            n_repeat_action = np.ones(states.shape[0]) * n_repeat_action

        for i, n_acts in enumerate(n_repeat_action):
            indexes = np.random.choice(np.arange(actions.shape[1]), size=n_acts)
            mask[i, indexes] = 1
        new_states = np.clip(self.state + actions * mask, self.min_x, self.max_x)
        rewards = self.function(new_states).flatten()
        ends = np.logical_not(
            [(self.min_x < new_states[i] < self.max_x).any() for i in range(actions.shape[0])]
        )
        infos = [{"terminal": False, "lives": 0} for _ in range(actions.shape[0])]
        return new_states.copy(), new_states.copy(), rewards, ends, infos

    def get_state(self):
        return self.state.copy()

    def set_state(self, state):
        self.state = state.copy()

    def reset(self, return_state: bool = True):
        self.state = np.random.uniform(self.min_x, self.max_x, size=self.n_dims)
        if return_state:
            return self.state.copy(), self.state.copy()
        return self.state.copy()
