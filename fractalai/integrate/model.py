import numpy as np
from gym import spaces
from fractalai.model import BaseModel


class NormalContinuousModel(BaseModel):
    def __init__(self, stepsize: float = 0.01, n_dims: int = 1):
        self.stepsize = stepsize
        action_space = spaces.Box(low=-10 * stepsize, high=10 * self.stepsize, shape=(n_dims,))

        super(NormalContinuousModel, self).__init__(action_space=action_space)

        self.shape = (n_dims,)

    def predict(self, observation: np.ndarray = None) -> np.ndarray:
        """
        Returns one action sampled from a continuous domain. It uses a uniform prior.
        :param observation: observation corresponding to a given state.
        :return: Numpy array representing the randomly chosen action.
        """
        return np.random.standard_normal(self.shape) * self.stepsize

    def predict_batch(self, observations: np.ndarray = None) -> np.ndarray:
        """
        Returns a vector of actions sampled from a continuous domain. It ses a uniform prior.
        :param observations: Array of observations corresponding to a vector of states.
        :return: Numpy array representing a vector of randomly chosen actions.
        """
        vals = np.random.standard_normal(size=(len(observations),) + self.shape) * self.stepsize
        return vals.reshape(-1, observations.shape[1])
