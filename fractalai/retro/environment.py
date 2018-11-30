import sys
import traceback
import numpy as np
from fractalai.environment import Environment, ExternalProcess, BatchEnv, resize_frame
from gym import spaces
import retro


class RetroEnvironment(Environment):
    """Environment for playing Atari games."""

    def __init__(
        self,
        name: str,
        n_repeat_action: int = 1,
        height: float = 100,
        width: float = 100,
        wrappers=None,
        **kwargs
    ):
        self._env_kwargs = kwargs
        self.height = height
        self.width = width
        super(RetroEnvironment, self).__init__(name=name, n_repeat_action=n_repeat_action)
        self._env = None
        if height is not None and width is not None:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8
            )
        self.wrappers = wrappers

    def init_env(self):
        env = retro.make(self.name, **self._env_kwargs).unwrapped
        if self.wrappers is not None:
            for wrap in self.wrappers:
                env = wrap(env)
        self._env = env
        self.action_space = self._env.action_space
        self.observation_space = (
            self._env.observation_space
            if self.observation_space is None
            else self.observation_space
        )

    def __getattr__(self, item):
        return getattr(self._env, item)

    def get_state(self) -> np.ndarray:
        state = self._env.em.get_state()
        return np.frombuffer(state, dtype=np.int32)

    def set_state(self, state: np.ndarray):
        raw_state = state.tobytes()
        self._env.em.set_state(raw_state)
        return state

    def step(
        self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None
    ) -> tuple:
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        if state is not None:
            self.set_state(state)
        reward = 0
        for i in range(n_repeat_action):
            obs, _reward, _, info = self._env.step(action)
            reward += _reward
            end_screen = info.get("screen_x", 0) >= info.get("screen_x_end", 1e6)
            terminal = info.get("x", 0) >= info.get("screen_x_end", 1e6) or end_screen
        obs = (
            resize_frame(obs, self.height, self.width)
            if self.width is not None and self.height is not None
            else obs
        )
        if state is not None:
            new_state = self.get_state()
            return new_state, obs, reward, terminal, info
        return obs, reward, terminal, info

    def step_batch(self, actions, states=None, n_repeat_action: [int, np.ndarray] = None) -> tuple:
        """

        :param actions:
        :param states:
        :param n_repeat_action:
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
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        if states is None:
            return observs, rewards, terminals, infos
        else:
            return new_states, observs, rewards, terminals, infos

    def reset(self, return_state: bool = True):
        obs = self._env.reset()
        obs = (
            resize_frame(obs, self.height, self.width)
            if self.width is not None and self.height is not None
            else obs
        )
        if not return_state:
            return obs
        else:
            return self.get_state(), obs


class ExternalRetro(ExternalProcess):
    def __init__(
        self,
        name,
        wrappers=None,
        n_repeat_action: int = 1,
        height: float = 100,
        width: float = 100,
        **kwargs
    ):
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
        super(ExternalRetro, self).__init__(
            constructor=(name, wrappers, n_repeat_action, height, width, kwargs)
        )

    def _worker(self, data, conn):
        """The process waits for actions and sends back environment results.
        Args:
          constructor: Constructor for the OpenAI Gym environment.
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            name, wrappers, n_repeat_action, height, width, kwargs = data
            env = RetroEnvironment(
                name,
                wrappers=wrappers,
                n_repeat_action=n_repeat_action,
                height=height,
                width=width,
                **kwargs
            )
            env.init_env()
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
                raise KeyError("Received message of unknown type {}".format(message))
        except Exception:  # pylint: disable=broad-except
            import tensorflow as tf

            stacktrace = "".join(traceback.format_exception(*sys.exc_info()))
            tf.logging.error("Error in environment process: {}".format(stacktrace))
            conn.send((self._EXCEPTION, stacktrace))
            conn.close()


class ParallelRetro(Environment):
    """Wrap any environment to be stepped in parallel."""

    def __init__(
        self,
        name: str,
        n_repeat_action: int = 1,
        height: float = 100,
        width: float = 100,
        wrappers=None,
        n_workers: int = 8,
        blocking: bool = True,
        **kwargs
    ):
        """

        :param name: Name of the Environment
        :param env_class: Class of the environment to be wrapped.
        :param n_workers: number of workers that will be used.
        :param blocking: step the environments asynchronously.
        :param args: args of the environment that will be parallelized.
        :param kwargs: kwargs of the environment that will be parallelized.
        """

        super(ParallelRetro, self).__init__(name=name)

        envs = [
            ExternalRetro(
                name=name,
                n_repeat_action=n_repeat_action,
                height=height,
                width=width,
                wrappers=wrappers,
                **kwargs
            )
            for _ in range(n_workers)
        ]
        self._batch_env = BatchEnv(envs, blocking)
        self._env = RetroEnvironment(
            name,
            n_repeat_action=n_repeat_action,
            height=height,
            width=width,
            wrappers=wrappers,
            **kwargs
        )
        self._env.init_env()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step_batch(
        self,
        actions: np.ndarray,
        states: np.ndarray = None,
        n_repeat_action: [np.ndarray, int] = None,
    ):
        return self._batch_env.step_batch(
            actions=actions, states=states, n_repeat_action=n_repeat_action
        )

    def step(self, action: np.ndarray, state: np.ndarray = None, n_repeat_action: int = None):
        return self._env.step(action=action, state=state, n_repeat_action=n_repeat_action)

    def reset(self, return_state: bool = True, blocking: bool = True):
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
