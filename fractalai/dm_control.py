import sys
import traceback
import numpy as np
from fractalai.environment import Environment, ExternalProcess, BatchEnv
from gym.envs.classic_control import rendering


class DMControlEnv(Environment):
        """I am offering this just to show that it can also work with any kind of problem, but I will
        not be offering support for the dm_control package. It relies on Mujoco, and I don't want
        to pollute this publication with proprietary code. Unfortunately, Mujoco is the only
         library that allows to easily set and clone the state of the environment.
         If anyone knows how to make it work with OpenAI Roboschool(PyBullet), I will release a
          distributed version of the algorithm that allows to scale easily to thousands of walkers,
        and run simulations in a cluster using ray.
        """

        def __init__(self, name: str = "cartpole-balance",
                     visualize_reward: bool = True, n_repeat_action: int = 1,
                     custom_death: "CustomDeath" = None):
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
            domain_name, task_name = name.split("-")
            super(DMControlEnv, self).__init__(name=name, n_repeat_action=n_repeat_action)
            self._render_i = 0
            self._env = suite.load(domain_name=domain_name, task_name=task_name,
                                   visualize_reward=visualize_reward)
            self._name = name
            self.viewer = []
            self._last_time_step = None
            self._viewer = rendering.SimpleImageViewer()

            self._custom_death = custom_death

            self.reset()

        def __getattr__(self, item):
            return getattr(self._env, item)

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

        def show_game(self, sleep: float=0.05):
            import time
            for img in self.viewer:
                self._viewer.imshow(img)
                time.sleep(sleep)

        def reset(self, return_state: bool = False) -> [np.ndarray, tuple]:
            """Resets the environment and returns the first observation"""

            time_step = self._env.reset()
            observed = self._time_step_to_obs(time_step)
            self._render_i = 0
            if not return_state:
                return observed
            else:
                return self.get_state(), observed

        def set_state(self, state: np.ndarray):
            """
            Sets the microstate of the simulator to the microstate of the target State.
            I will be super grateful if someone shows me how to do this using Open Source code.
            :param state:
            :return:
            """
            with self.env.physics.reset_context():
                # mj_reset () is  called  upon  entering  the  context.
                self.env.physics.data.qpos[:] = state[0]  # Set  position ,
                self.env.physics.data.qvel[:] = state[1]  # velocity
                self.env.physics.data.ctrl[:] = state[2]  # and  control.

        def get_state(self) -> tuple:
            state = (np.array(self.env.physics.data.qpos),
                     np.array(self.env.physics.data.qvel),
                     np.array(self.env.physics.data.ctrl))
            return state

        def step(self, action: np.ndarray, state: np.ndarray = None,
                 n_repeat_action: int = None) -> tuple:
            n_repeat_action = n_repeat_action if n_repeat_action is not None\
                else self.n_repeat_action

            custom_death = False
            end = False
            cum_reward = 0
            if state is not None:
                self.set_state(state)
            for i in range(n_repeat_action):
                time_step = self.env.step(action)
                end = end or time_step.last()
                cum_reward += time_step.reward
                # The death condition is a super efficient way to discard huge chunks of the
                # state space at discretion of the programmer.
                if self._custom_death is not None:
                    custom_death = custom_death or \
                                   self._custom_death.calculate(self,
                                                                time_step,
                                                                self._last_time_step)
                self._last_time_step = time_step
                if end:
                    break
            observed = self._time_step_to_obs(time_step)
            # This is written as a hack because using custom deaths should be a hack.
            if self._custom_death is not None:
                end = end or custom_death

            if state is not None:
                new_state = self.get_state()
                return new_state, observed, cum_reward, end, {"lives": 0, "dt": n_repeat_action}
            return observed, cum_reward, end, {"lives": 0, "dt": n_repeat_action}

        def step_batch(self, actions, states=None,
                       n_repeat_action: [int, np.ndarray] = None) -> tuple:
            """

            :param actions:
            :param states:
            :param n_repeat_action:
            :return:
            """
            n_repeat_action = n_repeat_action if n_repeat_action is not None \
                else self.n_repeat_action
            n_repeat_action = n_repeat_action if isinstance(n_repeat_action, np.ndarray) \
                else np.ones(len(states)) * n_repeat_action
            data = [self.step(action, state, n_repeat_action=dt)
                    for action, state, dt in zip(actions, states, n_repeat_action)]
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

        @staticmethod
        def _time_step_to_obs(time_step) -> np.ndarray:
            # Concat observations in a single array, so it is easier to calculate distances
            obs_array = np.hstack([np.array([time_step.observation[x]]).flatten()
                                   for x in time_step.observation])
            return obs_array


class ExternalDMControl(ExternalProcess):

    def __init__(self, name, wrappers=None, n_repeat_action: int=1, **kwargs):
        """Step environment in a separate process for lock free paralellism.
        The environment will be created in the external process by calling the
        specified callable. This can be an environment class, or a function
        creating the environment and potentially wrapping it. The returned
        environment should not access global variables.
        Args:
          constructor: Callable that creates and returns an OpenAI gym environment.
        Attributes:
          observation_space: The cached observation space of the environment.
          action_space: The cached action space of the environment.
        """
        self.name = name
        super(ExternalDMControl, self).__init__(constructor=(name, wrappers,
                                                             n_repeat_action, kwargs))

    def _worker(self, data, conn):
        """The process waits for actions and sends back environment results.
        Args:
          data: tuple containing the necessari parameters.
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            name, wrappers, n_repeat_action, kwargs = data

            env = DMControlEnv(name, n_repeat_action=n_repeat_action,
                               **kwargs)
            # dom_name, task_name = name.split("-")
            # custom_death = CustomDeath(domain_name=dom_name,
            #                             task_name=task_name)
            env.reset()
            while True:
                try:
                    # Only block for short times to have keyboard exceptions be raised.
                    if not conn.poll(0.1):
                        continue
                    message, payload = conn.recv()
                except (EOFError, KeyboardInterrupt):
                    break
                if message == self._ACCESS:
                    name = payload
                    result = getattr(env, name)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CALL:
                    name, args, kwargs = payload
                    result = getattr(env, name)(*args, **kwargs)
                    conn.send((self._RESULT, result))
                    continue
                if message == self._CLOSE:
                    assert payload is None
                    break
                raise KeyError('Received message of unknown type {}'.format(message))
        except Exception:  # pylint: disable=broad-except
            import tensorflow as tf
            stacktrace = ''.join(traceback.format_exception(*sys.exc_info()))
            tf.logging.error('Error in environment process: {}'.format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class ParallelDMControl(Environment):
    """Wrap a dm_control environment to be stepped in parallel."""

    def __init__(self, name: str, n_repeat_action: int=1, n_workers: int=8,
                 blocking: bool=True, **kwargs):
        """

        :param name: Name of the Environment
        :param env_class: Class of the environment to be wrapped.
        :param n_workers: number of workers that will be used.
        :param blocking: step the environments asynchronously.
        :param args: args of the environment that will be parallelized.
        :param kwargs: kwargs of the environment that will be parallelized.
        """

        super(ParallelDMControl, self).__init__(name=name)

        envs = [ExternalDMControl(name=name, n_repeat_action=n_repeat_action, **kwargs)
                for _ in range(n_workers)]
        self._batch_env = BatchEnv(envs, blocking)
        self._env = DMControlEnv(name, n_repeat_action=n_repeat_action, **kwargs)

        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step_batch(self, actions: np.ndarray, states: np.ndarray=None,
                   n_repeat_action: [np.ndarray, int]=None):
        return self._batch_env.step_batch(actions=actions, states=states,
                                          n_repeat_action=n_repeat_action)

    def step(self, action: np.ndarray, state: np.ndarray=None, n_repeat_action: int=None):
        return self._env.step(action=action, state=state, n_repeat_action=n_repeat_action)

    def reset(self, return_state: bool = True, blocking: bool=True):
        state, obs = self._env.reset(return_state=True)
        self.sync_states()
        return state, obs if return_state else obs

    def get_state(self):
        return self._env.get_state()

    def set_state(self, state):
        self._env.set_state(state)
        self.sync_states()

    def sync_states(self):
        self._batch_env.sync_states(self.get_state())


class CustomDeath:

    def __init__(self, domain_name="cartpole", task_name="balance"):
        self._domain_name = domain_name
        self._task_name = task_name

    @property
    def task_name(self):
        return self._task_name

    @property
    def domain_name(self):
        return self._domain_name

    def calculate(self, env: DMControlEnv, time_step, last_time_step):

        if self._domain_name == "cartpole" and self.task_name == "balance":
            return self._cartpole_balance_death(env=env, time_step=time_step)
        elif self._domain_name == "hopper":
            return self._hopper_death(env=env, time_step=time_step, last_time_step=last_time_step)
        elif self._domain_name == "hopper":
            return self._hopper_death(env=env, time_step=time_step, last_time_step=last_time_step)
        elif self._domain_name == "walker":
            return self._walker_death(env=env, time_step=time_step, last_time_step=last_time_step)
        else:
            return self._default_death(time_step, last_time_step)

    @staticmethod
    def _default_death(time_step, last_time_step) -> bool:
        last_rew = last_time_step.reward if last_time_step is not None else 0
        return time_step.reward <= 0 and last_rew > 0

    @staticmethod
    def _cartpole_balance_death(env, time_step, ) -> bool:
        """If the reward is less than 0.7 consider a state dead. This threshold is because rewards
        lesser than 0.7 involve positions where the cartpole is not balanced.
        """
        return time_step.reward < 0.75 or abs(env.physics.cart_position()) > 0.5

    @staticmethod
    def _hopper_death(env, time_step, last_time_step) -> bool:
        min_torso_height = 0.1
        max_reward_drop = 0.3

        torso_touches_ground = env.physics.height() < min_torso_height
        # reward_change = time_step.reward - (last_time_step.reward if
        # last_time_step is not None else 0)
        # reward_drops = reward_change < -max_reward_drop * env.n_repeat_action
        return False

    @staticmethod
    def _walker_death(env, time_step, last_time_step) -> bool:
        min_torso_height = 0.1
        max_reward_drop_pct = 0.5
        # max_tilt = 0
        min_reward = 0.1

        torso_touches_ground = env.physics.torso_height() < min_torso_height
        last_reward = last_time_step.reward if last_time_step is not None else 0.00001
        reward_change = (time_step.reward / last_reward)
        reward_drops = reward_change < max_reward_drop_pct
        torso_very_tilted = False  # abs(env.physics.torso_upright())
        # < max_tilt and reward_change < 0
        # torso_very_tilted = torso_very_tilted if not env.state.dead else False

        crappy_reward = time_step.reward < min_reward if not env.state.dead else False

        return reward_drops or torso_touches_ground or torso_very_tilted or crappy_reward
