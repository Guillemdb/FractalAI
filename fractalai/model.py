import numpy as np
from typing import Iterable, Sized


class BaseModel:
    """This is the model of the simulation that will be part of the policy.

     Its function is sampling an action from a given observation.
     """

    def __init__(self, action_space, action_shape=(None,)):
        self.action_space = action_space
        self._action_shape = action_shape

    @property
    def action_shape(self):
        return self._action_shape

    def predict(self, observation: np.ndarray=None) -> [np.ndarray, float, int]:
        """
        Returns one action available from a given state or a vector of swarm.

        :param observation: **State** or vector of swarm.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        raise NotImplementedError

    def predict_batch(self,  observations: Iterable) -> np.ndarray:
        """
        Returns a set of actions corresponding to a vector of observations.
        :param observations: vector of observations to sample a random action from.

        :return: Returns a numpy array containing the action corresponding to each observation.
        """

        return np.array([self.predict(obs) for obs in observations])


class DiscreteModel(BaseModel):
    """This is the base class for discrete models.

     Its function is sampling a discrete action given an observation.
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

    def predict(self, observation: np.ndarray=None) -> int:
        """
        Returns one action available from a given state or a vector of swarm.
        :param observation: observation representing the state to be modeled.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        raise NotImplementedError


class RandomDiscreteModel(DiscreteModel):
    """Discrete Model that samples an action randomly from a set of ``n_actions`` possible
    actions."""

    def __init__(self, n_actions: int, action_space=None,
                 max_wakers: int=100, samples: int=100000, use_block: bool=False):
        """

        :param n_actions:
        :param action_space:
        :param max_wakers:
        :param samples:
        """
        super(RandomDiscreteModel, self).__init__(n_actions=n_actions, action_space=action_space)
        self.max_walkers = max_wakers
        self.samples = samples
        self.use_block = use_block
        if use_block:
            self.noise = np.random.randint(0, high=int(self.n_actions), size=(int(max_wakers),
                                                                              int(samples)))
            self._i = 0

    def predict(self, observation: np.ndarray=None) -> [int, np.ndarray]:
        """
        Returns one action chosen at random.
        :param observation: Will be ignored. Observation representing the current state.
        :return: int representing the action to be taken.
        """
        if not self.use_block:
            return np.random.randint(0, high=int(self.n_actions))
        self._i += 1
        return self.noise[0, self._i % self.samples]

    def predict_batch(self, observations: [Sized, Iterable]) -> np.ndarray:
        """
        Returns a vector of actions chosen at random.
        :param observations: Represents a vector of observations. Only used in determining the size
        of the returned array.
        :return: Numpy array containing the action chosen for each observation.
        """

        if not self.use_block:
            return np.random.randint(0, high=int(self.n_actions), size=(len(observations),))
        self._i += 1
        return self.noise[:len(observations), self._i % self.samples]


class ESModel(BaseModel):

    def __init__(self, weights_shapes: [Iterable], sigma: float=0.01):
        super(ESModel, self).__init__(action_space=weights_shapes)
        self.weigths_shapes = weights_shapes
        self.sigma = sigma

    def predict(self, observation: np.ndarray=None) -> [int, np.ndarray]:
        """
        Returns one action chosen at random.
        :param observation: Will be ignored. Observation representing the current state.
        :return: int representing the action to be taken.
        """
        return [np.random.randn(*shape) * self.sigma for shape in self.weigths_shapes]

    def predict_batch(self, observations: [Sized, Iterable]) -> np.ndarray:
        """
        Returns a vector of actions chosen at random.
        :param observations: Represents a vector of observations. Only used in determining the size
        of the returned array.
        :return: Numpy array containing the action chosen for each observation.
        """

        perturbations = []
        for i in range(len(observations)):
            x = [np.random.randn(*shape) * self.sigma for shape in self.weigths_shapes]
            perturbations.append(x)
        return np.array(perturbations)



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

    def predict(self, observation: np.ndarray=None) -> np.ndarray:
        """
        Returns one action sampled from a continuous domain.
        :param observation: observation corresponding to a given state.
        :return: An scalar in case ``state`` represents a single state, or a numpy array otherwise.
        """
        raise NotImplementedError


class RandomContinuousModel(ContinuousModel):

    def __init__(self, action_space):
        super(RandomContinuousModel, self).__init__(action_space=action_space)

    def predict(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Returns one action sampled from a continuous domain. It uses a uniform prior.
        :param observation: observation corresponding to a given state.
        :return: Numpy array representing the randomly chosen action.
        """
        return np.random.uniform(self.minimum,
                                 self.maximum,
                                 size=self.shape)

    def predict_batch(self, observations: np.ndarray = None) -> np.ndarray:
        """
        Returns a vector of actions sampled from a continuous domain. It ses a uniform prior.
        :param observations: Array of observations corresponding to a vector of states.
        :return: Numpy array representing a vector of randomly chosen actions.
        """
        return np.random.uniform(self.minimum,
                                 self.maximum,
                                 size=(len(observations),)+self.shape)


class ContinuousDiscretizedModel(RandomContinuousModel):

    def __init__(self, action_space, n_act_dof=7):
        super(ContinuousDiscretizedModel, self).__init__(action_space=action_space)
        self.n_act_dof = n_act_dof

    def predict(self, observation: np.ndarray=None) -> np.ndarray:
        """
        Returns one action sampled at random from a continuous domain after discretizing it.
        :param observation: observation corresponding to a given state.
        :return: Numpy array representing a vector of randomly chosen actions.
        """
        perturbation = np.random.uniform(self.minimum, self.maximum, size=self.shape)

        jump = (self.maximum - self.minimum) / self.n_act_dof
        rounded = (perturbation / jump).astype(int) * jump
        return rounded
