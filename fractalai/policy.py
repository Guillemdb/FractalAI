import numpy as np
from typing import Iterable
from fractalai.state import State
from fractalai.environment import Environment
from fractalai.model import BaseModel, DiscreteModel, RandomContinuousModel, RandomDiscreteModel


class Policy:
    """
    The policy is in charge of given a state, returning the next state of the system, and
    processing the raw state information provided by the environment. The provided model will
    be used to sample an action.
    """

    def __init__(self, model: BaseModel, env: Environment):
        """
        The policy is in charge of given a state, returning the next state of the system, and
        processing the raw state information provided by the environment. The provided model will
        be used to sample an action.
        :param model: Model which given a state returns a prob dist for taking
                      each one of the available actions.
        :param env: Environment where the chosen actions are applied.
        """
        self._is_discrete = isinstance(model, DiscreteModel)
        self._model = model
        self._env = env
        self._last_pred = None
        self._model_pred = None

    @property
    def name(self) -> str:
        """Name of the current environment."""
        return self.env.name

    @property
    def is_discrete(self) -> bool:
        """The model is meant to be used in discrete environments."""
        return self._is_discrete

    @property
    def env(self):
        return self._env

    @property
    def model(self):
        return self._model

    @property
    def last_pred(self):
        return self._last_pred

    @property
    def model_pred(self):
        return self._model_pred

    @property
    def n_actions(self) -> int:
        """Number of available actions that can be taken."""
        if self.is_discrete:
            return self.model.n_actions
        else:
            raise ValueError("The current model is not discrete")

    def _predict(self, state: State, *args, **kwargs) -> np.ndarray:
        """
        Samples the action that will be taken at a given state.
        :param state: Current State of the system.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: The more probable action of the action probability distribution.
        """
        raise NotImplementedError

    def predict(self, state: [State, Iterable], *args, **kwargs) -> [State, list]:
        """
        Samples the action that will be taken at a given state.
        :param state: Current State of the system.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: The more probable action of the action probability distribution.
        """
        if not isinstance(state, Iterable):
            return self._predict(state, *args, **kwargs)
        return [self._predict(s, *args, **kwargs) for s in state]

    def _act(self, state: State, render: bool = False, *args, **kwargs) -> State:
        """ Given an arbitrary state, acts on the environment and return the
        processed new state where the system ends up.
        :param state: State that represents the current state of the environment.
        :param render: If true, the environment will be displayed.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: Next state of the system.
        """
        model_action = self.model.predict(state)
        self._model_pred = model_action
        action = self.predict(state, *args, **kwargs)
        self._last_pred = action
        new_state = self.env.step(state, action)
        if render:
            self.env.render()
        new_state.update_state(policy_action=action, model_action=model_action)
        return new_state

    def act(self, state: [State, Iterable], render: bool = False,
            *args, **kwargs) -> [State, list]:
        """ Given an arbitrary state, acts on the environment and return the
        processed new state where the system ends up.
        :param state: State that represents the current state of the environment.
        :param render: If true, the environment will be displayed.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: Next state of the system.
        """
        if not isinstance(state, Iterable):
            return self._act(state, render=render, *args, **kwargs)
        return [self._act(s, render=render, *args, **kwargs) for s in state]

    def _step(self, state: State, action: [np.ndarray, int, float], fixed_steps: int=1) -> State:
        """
        Steps one state and returns the processed version of the next state.
        :param state: Current state of the environment.
        :param action: Action or probability distribution of the action to be taken. If its a prob
                       distribution the most likely value will be used as the action.
        :return: Processed version of the next state the simulation will be in.
        """
        return self.env.step(state, action, fixed_steps=fixed_steps)

    def step(self, state: [State, Iterable], action: [int, np.ndarray, list, tuple],
             fixed_steps: int=1) -> [State, list]:
        """
        Step a vector of state.
        :param state:
        :param action: Vector of actions or probability distributions of the actions to be taken.
                        If it is a vector of prob distribution the most likely value will be used
                        as the action.
        :return: Processed vector of state.
        """

        if not isinstance(state, Iterable):
            return self._step(state=state, action=action, fixed_steps=fixed_steps)
        return [self._step(state=si, action=ai, fixed_steps=fixed_steps)
                for si, ai in zip(state, action)]

    def skip_frames(self, n_frames: int=0, render=False) -> State:
        """
        Skip the next n states. Play actions at random for `n_frames` frames.
        It is useful at the beginning of the environment to save some time.
        :param n_frames: Number of frames to be skipped by taking random actions.
        :param render: Render the environment while taking random actions.
        :return: The state of the environment after taking random actions for n_frames.
        """
        new_state = self.env.state.create_clone()
        if isinstance(self.model, DiscreteModel):
            rnd_model = RandomDiscreteModel(n_actions=self.n_actions)
        else:
            rnd_model = RandomContinuousModel(action_space=self.env.env.action_spec())
        action = rnd_model.predict(new_state)
        for i in range(n_frames):
            new_state = self.step(state=new_state, action=action)
            if render:
                self.env.render()
        self.env.set_simulation_state(new_state)
        return new_state

    def reset(self) -> [State]:
        """Reset environment and return initial state"""
        return self.env.reset()

    def evaluate(self, render=False, max_steps=None) -> tuple:
        """
        Evaluate the current model by playing one episode.
        :param render:
        :param max_steps:
        :return:
        """
        state = self.env.reset()
        episode_len = 0
        step = 0
        stop_step = False
        while not (state.terminal or state.dead) and not stop_step:
            state = self.act(state)
            episode_len += 1
            stop_step = False if max_steps is None else step > max_steps
            step += 1
            if render:
                self.env.render()

        return state.reward, episode_len


class GreedyPolicy(Policy):
    def __init__(self, model: BaseModel, env: Environment):
        super(GreedyPolicy, self).__init__(model=model, env=env)

    def _predict(self, state: State, *args, **kwargs) -> np.ndarray:
        """
        Samples the action that will be taken at a given state.
        :param state: Current State of the system.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: The action probability distribution.
        """
        return self.model.predict(state, *args, **kwargs)


class PolicyWrapper(Policy):

    def __init__(self, policy: Policy):
        super(PolicyWrapper, self).__init__(model=policy.model, env=policy.env)
        self.policy = policy

    def __getattr__(self, item):
        """Forward unimplemented attributes to the underlying policy."""
        return getattr(self.policy, item)

    def _predict(self, state: State, *args, **kwargs) -> np.ndarray:
        """
        Samples the action that will be taken at a given state.
        :param state: Current State of the system.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: The more probable action of the action probability distribution.
        """
        return self.policy.predict(state, *args, **kwargs)

    def _act(self, state: [State, Iterable], render: bool = False,
             *args, **kwargs) -> [State, list]:
        """ Given an arbitrary state, acts on the environment and return the
        processed new state where the system ends up.
        :param state: State that represents the current state of the environment.
        :param render: If true, the environment will be displayed.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: Next state of the system.
        """
        return self.policy.act(state=state, render=render, *args, **kwargs)

    def _step(self, state: [State, np.ndarray, list, tuple],
              action: [np.ndarray, list, tuple], fixed_steps: int=1) -> [State, list]:
        """

        :param state:
        :param action:
        :param fixed_steps:
        :return:
        """
        return self.policy.step(state=state, action=action, fixed_steps=fixed_steps)
