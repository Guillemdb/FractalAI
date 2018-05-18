import copy
import numpy as np
from typing import Iterable


class AbstractState:
    """This class is meant to contain all the information needed about the environment at a given
    moment.
    """

    @property
    def observed(self) -> np.ndarray:
        """Observation output by the environment"""
        raise NotImplementedError

    @property
    def microstate(self) -> [int, float, np.ndarray, "Microstate"]:
        """Information about the full state of the environment that allows to clone it."""
        raise NotImplementedError

    @property
    def dead(self) -> bool:
        """Describes if the episode has terminated."""
        raise NotImplementedError

    @property
    def terminal(self) -> bool:
        """Describes if the episode has terminated."""
        raise NotImplementedError

    @property
    def reward(self) -> bool:
        """Contains the reward output by thhe environment at a given moment."""
        raise NotImplementedError

    @property
    def model_data(self) -> Iterable:
        """Additional data output by the model of the environment."""
        raise NotImplementedError

    @property
    def model_action(self) -> np.ndarray:
        """Probability distribution of choosing a given action output by the model."""
        raise NotImplementedError

    @property
    def policy_data(self) -> Iterable:
        """Additional data output by the policy of the environment."""
        raise NotImplementedError

    @property
    def policy_action(self) -> np.ndarray:
        """Probability distribution of choosing a given action output by the policy."""
        raise NotImplementedError

    def update_state(self,
                     observed: np.ndarray=None,
                     microstate: [int, float, np.ndarray, "Microstate"]=None,
                     reward: [int, float]=None,
                     end: bool=None):
        """
        Update the information of the state with the raw information that is output by the model.
        :param observed: np.ndarray; Observation of the environment.
        :param microstate: Information needed to clone the state of the environment.
        :param reward: float; Reward obtained from the environment.
        :param end: bool; Indicates if the episode has terminated.
        :return: None
        """
        raise NotImplementedError

    def create_clone(self) -> "State":
        """Returns an exact copy of the current state of the environment."""
        raise NotImplementedError


class State(AbstractState):
    """This class represent the notion of State in the FAI algorithm. This is a representation of
    the state of the system in the framework of statistical mechanics, so it completely
    characterises the environment in a given instant.

    Parameters
    ----------
    :param observed: np.ndarray;
               Observation corresponding to the the represented state.

    :param microstate: int;
               Description of the microstate corresponding to the the represented state.

    :param model_data: Additional data output by the model of the environment.

    :param model_action: Probability distribution of choosing a given action output by
                        the model.

    :param policy_action: Probability distribution of choosing a given action output
                          by the policy.

    :param policy_data: Additional data output by the policy of the environment.

    Attributes
    ----------
    :param reward: float;
               Reward corresponding to the the represented state.

    :param end: bool;
         Indicates if the current state meets the dead condition.
    """

    def __init__(self,
                 observed: np.ndarray=None,
                 microstate: [int, float, np.ndarray, "Microstate"]=None,
                 reward: float=0.,
                 end: bool=False,
                 model_data: Iterable=None,
                 model_action: np.ndarray=None,
                 policy_action: np.ndarray=None,
                 policy_data: Iterable=None):
        """
        Initializes an State class meant to define the state of an Environment.
        :param observed: np.ndarray.
        :param microstate: Information about the full state of the environment that allows
                           to clone it.
        :param reward: float; Contains the reward output by the environment at a given moment.
        :param end: bool; Describes if the episode has terminated.
        :param model_data: Additional data output by the model of the environment.
        :param model_action: Probability distribution of choosing a given action output by
                            the model.
        :param policy_action: Probability distribution of choosing a given action output
                              by the policy.
        :param policy_data: Additional data output by the policy of the environment.
        """
        self._observed = observed
        self._microstate = microstate
        self._dead = end
        self._terminal = end
        self._reward = reward
        self._model_data = model_data
        self._model_action = model_action
        self._policy_data = policy_data
        self._policy_action = policy_action

    @property
    def observed(self) -> np.ndarray:
        """Observation output by the environment"""
        return self._observed

    @property
    def microstate(self) -> [int, float, np.ndarray, "Microstate"]:
        """Information about the full state of the environment that allows to clone it."""
        return self._microstate

    @observed.setter
    def observed(self, val: np.ndarray):
        """Observation output by the environment"""
        self._observed = val

    @microstate.setter
    def microstate(self, val: [int, float, np.ndarray, "Microstate"]):
        """Information about the full state of the environment that allows to clone it."""
        self._microstate = val

    @property
    def dead(self) -> bool:
        return self._dead

    @property
    def terminal(self) -> bool:
        return self._terminal

    @terminal.setter
    def terminal(self, val: bool):
        self._terminal = val

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def model_data(self) -> Iterable:
        return self._model_data

    @property
    def model_action(self) -> np.ndarray:
        return self._model_action

    @property
    def policy_data(self) -> Iterable:
        return self._policy_data

    @property
    def policy_action(self) -> [int, float, np.ndarray, None]:
        return self._policy_action

    def update_policy_data(self, val):
        self._policy_data = val

    def update_policy_action(self, val):
        self._policy_action = val

    def update_end(self, val: bool):
        self._dead = val
        self.terminal = val

    def update_reward(self, val: [int, float]):
        self._reward = val

    def reset_state(self):
        self._reward = 0.
        self._dead = False
        self._terminal = False

    def update_model_data(self, val: Iterable):
        self._model_data = val

    def update_model_action(self, val: Iterable):
        self._model_action = val

    def update_state(self,
                     observed: np.ndarray=None,
                     microstate: [int, float, np.ndarray, "Microstate"]=None,
                     reward: [int, float]=None,
                     end: bool=None,
                     policy_action: np.ndarray=None,
                     model_action: np.ndarray = None,
                     policy_data: Iterable=None,
                     model_data: Iterable=None):
        """
        Update the information of the state with the raw information that is output by the model.
        :param observed: np.ndarray; Observation of the environment.
        :param microstate: Information needed to clone the state of the environment.
        :param reward: float; Reward obtained from the environment.
        :param end: bool; Indicates if the episode has terminated.
        :param model_data: Additional data output by the model of the environment.
        :param model_action: Probability distribution of choosing a given action output by
                            the model.
        :param policy_action: Probability distribution of choosing a given action output
                              by the policy.
        :param policy_data: Additional data output by the policy of the environment.
        :return: None.
        """
        if observed is not None:
            self.observed = observed
        if microstate is not None:
            self.microstate = microstate
        if reward is not None:
            self.update_reward(reward)
        if end is not None:
            self.update_end(end)
        if policy_action is not None:
            self._policy_action = policy_action
        if model_action is not None:
            self._model_action = model_action
        if policy_data is not None:
            self._policy_data = policy_data
        if model_data is not None:
            self._model_data = model_data

    def create_clone(self) -> "State":
        """Returns an exact copy of the current state of the environment."""
        return copy.deepcopy(self)


class MicrostateCore:
    """This is for avoiding memory leaks when cloning Atari ale environments."""
    def __init__(self, env, ptr):
        self._env = env
        self._value = ptr

    @property
    def value(self):
        return self._value

    def __del__(self):
        self._env.unwrapped.ale.deleteState(self._value)
        self._value = None


class Microstate:
    """This is for avoiding memory leaks when cloning  Atari ale environments."""
    def __init__(self, env, ptr):
        self._core = MicrostateCore(env, ptr)

    @property
    def value(self):
        return self._core.value

    def __deepcopy__(self, memo):
        ms = Microstate.__new__(Microstate)
        ms._core = self._core
        return ms


class AtariState(State):

    """Creates an State class that is meant to be used in Atari games.  Adds tracking lives to
    the base State class.

    Parameters
    ----------

    :param observed: np.ndarray;
               Observation corresponding to the the represented state.

    :param microstate: int;
               Description of the microstate corresponding to the the represented state.

    :param reward: float;
               Reward corresponding to the the represented state.

    :param end: bool;
         Indicates if the current state meets the dead condition.

    Attributes
    ----------
    dead: bool;
             Returns true if the current sate is considered dead by the FAI.

    lives: int;
           Number of lives left in the game in the current game.

    reward: float;
               Reward corresponding to the the represented state.

    dead: bool;
         Indicates if the current state meets the dead condition.
    """

    def __init__(self, observed: np.ndarray=None,
                 microstate: "Microstate"=None,
                 reward: float = 0, end: bool = False, lives: int = -1000, model_data=None,
                 model_action=None, policy_data=None, policy_action=None):
        """
        Initializes an State class meant to define the state of an Atari game.
        :param observed: np.ndarray.
        :param microstate: Information about the full state of the environment that allows
                           to clone it.
        :param reward: float; Contains the reward output by thhe environment at a given moment.
        :param end: bool; Describes if the episode has terminated.
        :param model_data: Additional data output by the model of the environment.
        :param model_action: Probability distribution of choosing a given action output by
                            the model.
        :param policy_action: Probability distribution of choosing a given action output
                              by the policy.
        :param policy_data: Additional data output by the policy of the environment.
        :param lives: Number of lives the agent has in the current environment.

        """
        super(AtariState, self).__init__(observed=observed, microstate=microstate,
                                         reward=reward, end=end, model_data=model_data,
                                         policy_data=policy_data, model_action=model_action,
                                         policy_action=policy_action)
        self._old_lives = lives
        self._lives = lives

    @property
    def dead(self) -> bool:
        """Returns a bool indicating if the current state meets the death condition"""
        return self._dead or self.lives < self._old_lives

    @property
    def lives(self) -> int:
        """
        Number of lives in the current game.
        :return: The number of lives the agent has in the current state.
        """
        return self._lives

    def update_lives(self, val: [int, float, dict]):
        """Updates the number of lives in the current game and keeps track of the last value"""
        self._old_lives = int(self._lives)

        if isinstance(val, dict):
            self._lives = int(val['ale.lives'])
        elif isinstance(val, (int, float)):
            self._lives = int(val)
        else:
            raise ValueError(
                "Lives is not a number nor a dict: found type {}".format(type(val)))
        if self.lives < self._old_lives:
            self._dead = True

    def update_state(self,
                     observed: np.ndarray=None,
                     microstate: [int, float, np.ndarray, "Microstate"]=None,
                     reward: [int, float]=None,
                     end: bool=None,
                     lives: int=None,
                     policy_action: np.ndarray=None,
                     model_action: np.ndarray = None,
                     policy_data: Iterable=None,
                     model_data: Iterable=None):
        """
        Update the information of the state with the raw information that is output by the model.
        :param observed: np.ndarray; Observation of the environment.
        :param microstate: Information needed to clone the state of the environment.
        :param reward: float; Reward obtained from the environment.
        :param end: bool; Indicates if the episode has terminated.
        :param model_data: Additional data output by the model of the environment.
        :param model_action: Probability distribution of choosing a given action output by
                            the model.
        :param policy_action: Probability distribution of choosing a given action output
                              by the policy.
        :param policy_data: Additional data output by the policy of the environment.
        :return: None.
        """
        super().update_state(observed=observed, microstate=microstate, reward=reward, end=end,
                             policy_action=policy_action, model_action=model_action,
                             policy_data=policy_data, model_data=model_data)
        if lives is not None:
            self.update_lives(lives)
