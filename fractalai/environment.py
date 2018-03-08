import copy
import numpy as np
import gym
from typing import Iterable
from gym.envs.registration import registry as gym_registry
from gym.envs.atari import AtariEnv
from fractalai.state import Microstate, State, AtariState


class Environment:
    """This class handles the simulation of the environment in the FAI algorithm. It act as a
    transfer function between states. Given a pair of State, Action it should be able to compute
    the next state of the environment.

    We have not implemented the get_environment_state because we only need to read its initial
     condition. So the initial state will be provided when reset() is called.

    Parameters
    ----------
    name: str;
          Name of the  environment that the Simulator represents.
    """

    def __init__(self, name: str, state: State=None, fixed_steps: int=1):
        """This class handles the simulation of the environment in the FAI algorithm. It acts as a
        transfer function between states. Given a pair of State, Action it should compute the next
         state of the environment.

        Parameters
        ----------
        :param name: str; Name of the  environment that the Simulator represents.
        :param state: State; State object that will be used to store information about
                                 the state of the environment.
        :param fixed_steps: The number of consecutive times that the action will be applied. This
                            allows us to set the frequency at which the policy will play.
        """
        self.fixed_steps = fixed_steps
        self._name = name
        self._state = state
        self._cum_reward = 0

    @property
    def state(self):
        return self._state

    @property
    def name(self):
        return self._name

    def set_seed(self, seed):
        pass

    def _step(self, state: State, action: [int, np.ndarray, list], fixed_steps: int=1) -> State:
        """
        Steps one state once. This way we can make the step() function work on batches also. If
        you are too lazy to add vectorized support, override this function when inheriting.
        :param state: Current State of the environment.
        :param action: Scalar. Action that will be taken in the environment.
        :param fixed_steps: The number of consecutive times that the action will be applied. This
                            allows us to set the frequency at which the policy will play.
        :return: The next unprocessed state of the environment.
        """
        self.set_simulation_state(state)
        return self.step_simulation(action, fixed_steps=fixed_steps)

    def step(self, state: [State, Iterable], action: np.ndarray, fixed_steps: int=1) -> [State,
                                                                                         list]:
        """
        Step either a State or a batch of States.
        :param state: State or vector of states to be stepped.
        :param action: Action that will be taken in each one of the states.
        :param fixed_steps: The number of consecutive times that the action will be applied. This
                            allows us to set the frequency at which the policy will play.
        :return: The State of the environment after taking the desired number of steps.
                 If the input provided was a batch of states, return the stepped batch.
        """
        if not isinstance(state, Iterable):
            return self._step(state, action, fixed_steps=fixed_steps)
        return [self._step(st, ac, fixed_steps=fixed_steps) for (st, ac) in zip(state, action)]

    def reset(self) -> State:
        """Resets the simulator and returns initial state of the environment.
         """
        raise NotImplementedError

    def render(self):
        """Render shows on the screen the current state of the environment.
        Please, try to make it look cool."""
        raise NotImplementedError

    def set_simulation_state(self, state: State):
        """
        Sets the microstate of the environment to the microstate of the target State.
        :param state: State object that will be used to set the new state of the environment.
        :return: None
        """
        raise NotImplementedError

    def step_simulation(self, action: np.ndarray, fixed_steps: int=1) -> State:
        """Perturbs the simulator with an arbitrary action.
        :param action: action that will be selected in the environment.
        :param fixed_steps: The number of consecutive times that the action will be applied. This
                            allows us to set the frequency at which the policy will play.
        :return: State representing the new state the environment is in.
        """
        raise NotImplementedError


class OpenAIEnvironment(Environment):
    """Simulator meant to deal with OpenAI environments. We use this to test stuff
             on a cartpole environment.
    Parameters
    ----------
    name: str;
          Name of the atari environment to be created. See: https://gym.openai.com/envs

    Attributes
    ----------
        env: openai AtariEnv;
             Environment object where the simulation takes place.
    """
    def __init__(self, name: str="CartPole-v0", env: gym.Env=None):
        """This initializes the state of the environment to match the desired environment.
        It should deal with the undocumented wrappers that gym has so we avoid random resets when
        simulating.
        """
        super(OpenAIEnvironment, self).__init__(name=name)
        if env is None and name:
            spec = gym_registry.spec(name)
            # not actually needed, but we feel safer
            spec.max_episode_steps = None
            spec.max_episode_time = None
            self._env = spec.make()
            self._name = name
        elif env is not None:
            self._env = env
            self._name = env.spec.id
        else:
            raise ValueError("An env or an env name must be specified")
        self._state = self.reset()

    @property
    def env(self):
        """Access to the openai environment."""
        return self._env

    def set_seed(self, seed):
        np.random.seed(seed)
        self.env.seed(seed)

    def render(self, *args, **kwargs):
        """Shows the current screen of the video game in a separate window"""
        self.env.render(*args, **kwargs)

    def reset(self) -> State:
        """Resets the environment and returns the first observation"""
        if self._state is None:
            self._state = State()
        obs = self.env.reset()
        if hasattr(self.env.unwrapped, "state"):
            microstate = self.env.unwrapped.state
        else:
            microstate = 0  # State will not be stored. Implement a subclass if you need it.
        self.state.reset_state()
        self.state.update_state(observed=obs, microstate=microstate, end=False, reward=0.)
        return self.state.create_clone()

    def set_simulation_state(self, state: State):
        """Sets the microstate of the simulator to the microstate of the target State"""
        self._state = state
        self._cum_reward = state.reward
        self.env.unwrapped.state = state.microstate

    def step_simulation(self, action: np.array, fixed_steps=1) -> State:
        """Perturb the simulator with an arbitrary action"""
        end = False
        for i in range(fixed_steps):
            observed, reward, _end, info = self.env.step(action.argmax())
            self._cum_reward += reward
            end = end or _end
            if end:
                break
        if hasattr(self.env.unwrapped, "state"):
            microstate = copy.deepcopy(self.env.unwrapped.state)
        else:
            microstate = 0  # State will not be stored. Implement a subclass if you need it.
        self.state.reset_state()
        self.state.update_state(observed=observed, microstate=microstate, reward=self._cum_reward,
                                end=end,  # model_action=action, policy_action=action,
                                model_data=[info])
        if end:
            self.env.reset()
        return self.state


class AtariEnvironment(Environment):
    """Environment class used for managing Atari games. It can be used as a perfect simulation, or
    as an imperfect one. It can work using rgb images, or ram as obs_0.

    Parameters
    ----------
    name: str;
          Name of the atari environment to be created. See: https://gym.openai.com/envs#atari
          works also with "GameName-ram-v0" like environments.

    clone_seeds: bool;
                 If true, clone the pseudo random number generators of the environment for a
                 perfect simulation. False provides an stochastic Simulator.

    Attributes
    ----------
        env: openai AtariEnv;
             Environment object where the simulation takes place.
    """

    def __init__(self, state: AtariState=None, name: str="MsPacman-v0",
                 clone_seeds: bool=True, env: AtariEnv=None,
                 fixed_steps: int=1):
        """
        Environment class used for managing Atari games. It can be used as a perfect simulation, or
        as an imperfect one. It can handle rgb images, or ram as observations.
        :param name: Name of the atari environment to be created.
                     See: https://gym.openai.com/envs#atari works also with "GameName-ram-v0" like
                     environments.
        :param clone_seeds:  bool;
                 If true, clone the pseudo random number generators of the emulator for a
                 perfect simulation. False provides an stochastic simulation.
        :param env: Openai AtariEnv, optional; Use an already existing env instead of creating one.
        :param fixed_steps: The number of consecutive times that the action will be applied. This
                            allows us to set the frequency at which the policy will play.
        """
        self._clone_seeds = clone_seeds
        self._cum_reward = 0
        if env is None and name:
            spec = gym_registry.spec(name)
            # not actually needed, but we feel safer
            spec.max_episode_steps = None
            spec.max_episode_time = None

            self._env = spec.make()
            self._name = name
        elif env is not None:
            self._env = env
            self._name = env.spec.id
        else:
            raise ValueError("An env or an env name must be specified")
        self._state = AtariState() if state is None else state
        if state is None:
            self._state = self.reset()

        super(AtariEnvironment, self).__init__(name=name, state=self.state,
                                               fixed_steps=fixed_steps)

    @property
    def env(self):
        return self._env

    @property
    def gym_env(self):
        return self._env

    @property
    def ram(self) -> np.ndarray:
        """Decode the ram values of the current state of the environment."""
        ram_size = self.gym_env.unwrapped.ale.getRAMSize()
        ram = np.zeros(ram_size, dtype=np.uint8)
        return self.gym_env.unwrapped.ale.getRAM(ram)

    @property
    def clone_seeds(self):
        """The full state of the system is being cloned, including random seeds."""
        return self._clone_seeds

    def get_microstate(self):
        if self.clone_seeds:
            microstate = self.env.unwrapped.ale.cloneSystemState()
        else:
            microstate = self.env.unwrapped.ale.cloneState()
        return Microstate(self.env, microstate)

    def set_seed(self, seed):
        np.random.seed(seed)
        self.env.seed(seed)

    def render(self, *args, **kwargs):
        """Shows the current screen of the video game in a separate window"""
        self.env.render(*args, **kwargs)

    def reset(self) -> AtariState:
        """
        Resets the environment and returns the first observation
        :return: AtariState representing the initial state of the game.
        """
        obs = self.env.reset()
        if self.clone_seeds:
            microstate = self.env.unwrapped.ale.cloneSystemState()
        else:
            microstate = self.env.unwrapped.ale.cloneState()
        self.state.reset_state()
        self.state.update_state(observed=obs, microstate=Microstate(self.env, microstate),
                                end=False, reward=0)
        return self.state.create_clone()

    def set_simulation_state(self, state: AtariState):
        """
        Sets the microstate of the environment to the microstate of the target State
        :param state: State that will be set on the environment.
        :return: None.
        """
        # Set the internal state of the atari emulator
        self._cum_reward = state.reward
        self._state = state
        if self.clone_seeds:
            self.env.unwrapped.ale.restoreSystemState(state.microstate.value)
        else:
            self.env.unwrapped.ale.restoreState(state.microstate.value)

    def step_simulation(self, action: np.ndarray, fixed_steps: int=1) -> AtariState:
        """
        Perturb the simulator with an arbitrary action.
        :param action: int representing the action to be taken.
        :param fixed_steps: The number of consecutive times that the action will be applied. This
                            allows us to set the frequency at which the policy will play.
        :return: State representing the state of the environment after taking the desired number of
                steps.
        """
        end = False
        _dead = False
        for i in range(fixed_steps):
            observed, reward, _end, lives = self.env.step(action.argmax())
            end = end or _end
            _dead = _dead or reward < 0
            self._cum_reward += reward
            if end:
                break

        if self.clone_seeds:
            microstate = self.env.unwrapped.ale.cloneSystemState()

        else:
            microstate = self.env.unwrapped.ale.cloneState()

        self.state.update_state(observed=observed, reward=self._cum_reward,
                                end=end, lives=lives, microstate=Microstate(self.env, microstate))
        if end:
            self.env.reset()
        return self.state


class DMControlEnv(Environment):
        """I am offering this just to show that it can also work with any kind of problem, but I will
        not be offering support for the dm_control package. It relies on Mujoco, and I don't want
        to pollute this publication with proprietary code. Unfortunately, Mujoco is the only
         library that allows to easily set and clone the state of the environment.

         If anyone knows how to make it work with OpenAI Roboschool(PyBullet), I will release a
          distributed version of the algorithm that allows to scale easily to thousands of walkers,
        and run simulations in a cluster using ray.
        """

        def __init__(self, domain_name="cartpole", task_name="balance",
                     visualize_reward: bool=True, fixed_steps: int=1,
                     custom_death: "CustomDeath"=None):
            """
            Creates DMControlEnv and initializes the environment.

            :param domain_name: match dm_control interface.
            :param task_name: match dm_control interface.
            :param visualize_reward: match dm_control interface.
            :param fixed_steps: The number of consecutive times that an action will be applied.
                            This allows us to set the frequency at which the policy will play.
            :param custom_death: Pro hack to beat the shit out of DeepMind even further.
            """
            from dm_control import suite
            name = str(domain_name) + ":" + str(task_name)
            super(DMControlEnv, self).__init__(name=name, state=None)
            self.fixed_steps = fixed_steps
            self._render_i = 0
            self._env = suite.load(domain_name=domain_name, task_name=task_name,
                                   visualize_reward=visualize_reward)
            self._name = name
            self.viewer = []
            self._last_time_step = None

            self._custom_death = custom_death
            self.reset()

        def action_spec(self):
            return self.env.action_spec()

        @property
        def physics(self):
            return self.env.physics

        @property
        def env(self):
            """Access to the environment."""
            return self._env

        def set_seed(self, seed):
            np.random.seed(seed)
            self.env.seed(seed)

        def render(self, mode='human'):
            img = self.env.physics.render(camera_id=0)
            if mode == 'rgb_array':
                return img
            elif mode == 'human':
                self.viewer.append(img)
            return True

        def reset(self) -> State:
            """Resets the environment and returns the first observation"""
            if self._state is None:
                self._state = State()
            time_step = self.env.reset()
            observed = self._time_step_to_obs(time_step)

            microstate = (np.array(self.env.physics.data.qpos),
                          np.array(self.env.physics.data.qvel),
                          np.array(self.env.physics.data.ctrl))

            self._render_i = 0
            self.state.reset_state()
            self.state.update_state(observed=observed, microstate=microstate,
                                    end=False, reward=time_step.reward)
            return self.state.create_clone()

        def set_simulation_state(self, state: State):
            """
            Sets the microstate of the simulator to the microstate of the target State.
            I will be super grateful if someone shows me how to do this using Open Source code.

            :param state:
            :return:
            """
            self._state = state
            self._cum_reward = state.reward
            with self.env.physics.reset_context():
                # mj_reset () is  called  upon  entering  the  context.
                self.env.physics.data.qpos[:] = state.microstate[0]  # Set  position ,
                self.env.physics.data.qvel[:] = state.microstate[1]  # velocity
                self.env.physics.data.ctrl[:] = state.microstate[2]  # and  control.

        def step_simulation(self, action: np.array, fixed_steps: int=None) -> State:
            """
            Perturb the simulator with an arbitrary action.
            :param action: int representing the action to be taken.
            :param fixed_steps: The number of consecutive times that the action will be applied.
                        This allows us to set the frequency at which the policy will play.
            :return: State representing the state of the environment after taking the desired
                    number of steps.
            """

            if fixed_steps is not None:
                self.fixed_steps = fixed_steps
            custom_death = False
            end = False

            for i in range(self.fixed_steps):
                time_step = self.env.step(action)
                end = end or time_step.last()
                self._cum_reward += time_step.reward
                # The death condition is a super efficient way to discard huge chunks of the
                # state space at discretion of the programmer. I am not uploading the code,
                # because it is really easy to screw up if you provide the wrong function.
                # In the end, its just a way to leverage human knowledge, and that should only
                # be attempted if you have a profound understanding of how everything works.
                # If done well, the speedup is a few orders of magnitude.
                if self._custom_death is not None:
                    custom_death = custom_death or \
                                   self._custom_death.calculate(self,
                                                                time_step,
                                                                self._last_time_step)
                self._last_time_step = time_step
                if end:
                    break
            # Here we save the state of the simulation inside the microstate attribute.
            observed = self._time_step_to_obs(time_step)
            microstate = (np.array(self.env.physics.data.qpos),
                          np.array(self.env.physics.data.qvel),
                          np.array(self.env.physics.data.ctrl))

            # self.state.reset_state() If commented, policy and model data is not set to None.
            self.state.update_state(observed=observed, microstate=microstate,
                                    reward=self._cum_reward,
                                    end=end)
            # This is written as a hack because using custom deaths should be a hack.
            if self._custom_death is not None:
                self.state._dead = self.state._dead or custom_death

            if end:
                self.env.reset()
            return self.state

        @staticmethod
        def _time_step_to_obs(time_step) -> tuple:
            # Concat observations in a single, so it is easier to calculate distances
            obs_array = np.hstack([np.array([time_step.observation[x]]).flatten()
                                   for x in time_step.observation])
            return obs_array
