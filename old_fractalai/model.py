import numpy as np
from typing import Iterable
from fractalai.state import State


class BaseModel:
    """This is the model of the simulation that will be part of the policy.

     Its function is sampling an action from a given state.
     """

    def __init__(self, action_space, action_shape=(None,)):
        self.action_space = action_space
        self._action_shape = action_shape

    @property
    def action_shape(self):
        return self._action_shape

    def _predict(self, state: [State, list, np.ndarray] = None) -> [np.ndarray, float, int]:
        """
        Returns one action available from a given state or a vector of swarm.

        :param state: **State** or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        raise NotImplementedError

    def predict(self, state: [State, Iterable] = None) -> np.ndarray:
        """
        Returns one action chosen at random from the policy's action space
        :param state: State or list of swarm to sample a random action from. It handles both a
        single state and a vector of swarm.
        :return: int representing the action to be taken in case there is a single state. If swarm
        is an iterable returns a numpy array containing the action for each state.
        """
        if not isinstance(state, Iterable):
            return self._predict(state)
        return np.array([self._predict(s) for s in state])


class DiscreteModel(BaseModel):
    """This is the model of the simulation that will be part of the policy.

     Its function is sampling a discrete from a given state. It also calculates the probabilities
     over each available actions.
     """

    def __init__(self, n_actions: int, action_space=None):
        super(DiscreteModel, self).__init__(action_space=action_space)
        self._n_actions = n_actions

    @property
    def n_actions(self):
        return self._n_actions

    @property
    def action_shape(self):
        return tuple([self.n_actions])

    def _predict(self, state: [State, list, np.ndarray] = None) -> int:
        """
        Returns one action available from a given state or a vector of swarm.
        :param state: State or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        raise NotImplementedError


class RandomDiscreteModel(DiscreteModel):
    """Discrete Model that samples a discrete action randomly from a set of ``n_actions`` possible
    actions."""

    def __init__(self, n_actions: int, action_space=None):
        """Policy that performs random actions on an Atari game"""
        super(RandomDiscreteModel, self).__init__(n_actions=n_actions, action_space=action_space)

    def _predict(self, state: State) -> [int, np.ndarray]:
        """
        Returns one action chosen at random from the policy's action space
        :param state: State or list of swarm to sample a random action from. It handles both a
        single state and a vector of swarm.
        :return: int representing the action to be taken in case there is a single state. If swarm
        is an iterable returns a numpy array containing the action for each state.
        """
        pred = np.random.randint(0, self.n_actions)
        vector = np.zeros(self.n_actions)
        vector[pred] = 1
        return vector


class RandomPongModel(DiscreteModel):
    def __init__(self, action_space=None):
        """Policy that performs random actions on an Pong game"""
        super(RandomPongModel, self).__init__(n_actions=6, action_space=action_space)

    def _predict(self, state: State) -> [int, np.ndarray]:
        """
        Returns one action chosen at random from the policy's action space
        :param state: State or list of swarm to sample a random action from. It handles both a
        single state and a vector of swarm.
        :return: int representing the action to be taken in case there is a single state. If swarm
        is an iterable returns a numpy array containing the action for each state.
        """
        pred = np.random.choice([3, 4])
        vector = np.zeros(self.n_actions)
        vector[pred] = 1
        return vector


class ContinuousModel(BaseModel):
    """This is the Model class meant to work with ``dm_control`` environments.

     Its function is sampling a random action from a given state. It uses a uniform prior.
     """

    def __init__(self, action_space):
        from dm_control.rl.specs import BoundedArraySpec

        assert isinstance(action_space, BoundedArraySpec), "Please use a dm_control action_spec"
        super(ContinuousModel, self).__init__(action_space=action_space)

    @property
    def action_spec(self):
        return self.action_space

    @property
    def shape(self):
        return self.action_space.shape

    @property
    def dtype(self):
        return self.action_space.dtype

    @property
    def name(self):
        return self.action_space.name

    @property
    def minimum(self):
        return self.action_space.minimum

    @property
    def maximum(self):
        return self.action_space.maximum

    def _predict(self, state: State = None) -> np.ndarray:
        """
        Returns one action available from a given state or a vector of swarm.
        :param state: State or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        raise NotImplementedError


class RandomContinuousModel(ContinuousModel):
    def __init__(self, action_space):
        super(RandomContinuousModel, self).__init__(action_space=action_space)

    def _predict(self, state: State = None) -> np.ndarray:
        """
        Returns one action available from a given state or a vector of swarm.
        :param state: State or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        return np.random.uniform(self.minimum, self.maximum, size=self.shape)


class RandomMomentumModel(ContinuousModel):
    def __init__(self, action_space, dt=1, uniform=True):
        super(RandomMomentumModel, self).__init__(action_space=action_space)
        self.dt = dt
        self.uniform = uniform

    def _predict(self, state: State = None) -> np.ndarray:
        """
        Returns one action available from a given state or a vector of swarm.
        :param state: State or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        old_action = state.model_data if state.model_data is not None else np.zeros(self.shape)
        # Assumes min negative and max positive, be careful.
        # This * 2 gives maximum variability, so any time it can change between max and min if dt=1
        if self.uniform:
            vel = np.random.uniform(self.minimum * 2, self.maximum * 2, size=self.shape)
        else:
            vel = np.random.standard_normal(self.shape)
        action = old_action + vel * self.dt
        action = np.clip(action, self.minimum, self.maximum)
        state.update_model_data(action)
        return action


class ContinuousDiscretizedModel(RandomContinuousModel):
    def __init__(self, action_space, n_act_dof=7):
        super(ContinuousDiscretizedModel, self).__init__(action_space=action_space)
        self.n_act_dof = n_act_dof

    def _normalize_vector(self, values: np.ndarray) -> np.ndarray:
        """
        Normalizes the values of a vector to be in range [0, 1].
        :param values: Values that will be normalized.
        :return: Normalized values.
        """
        max_r, min_r = values.max(), values.min()
        if min_r == max_r:
            values = np.ones(len(values), dtype=np.float32)
        else:
            values = (values - min_r) / (max_r - min_r)
        normed = values / values.sum()
        return normed

    def _predict(self, state: State = None) -> np.ndarray:
        """
        Returns one action available from a given state or a vector of swarm.
        :param state: State or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        perturbation = np.random.uniform(self.minimum, self.maximum, size=self.shape)

        jump = (self.maximum - self.minimum) / self.n_act_dof
        rounded = (perturbation / jump).astype(int) * jump
        return rounded
