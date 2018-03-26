import atexit
import multiprocessing
import sys
import traceback
from gym.envs.registration import registry as gym_registry
import numpy as np


class Environment:
    """Inherit from this class to test the Swarm on a different problem."""
    def __init__(self, name, n_repeat_action: int=1):
        self._name = name
        self.n_repeat_action = n_repeat_action

    @property
    def name(self):
        return self._name

    def step(self, action, state=None, n_repeat_action: int=1) -> tuple:
        raise NotImplementedError

    def step_batch(self, actions, states=None, n_repeat_action: int=1) -> tuple:
        raise NotImplementedError

    def reset(self) -> tuple:
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state):
        raise NotImplementedError


class AtariEnvironment(Environment):
    """Environment for playing Atari games."""

    def __init__(self, name: str, clone_seeds: bool=True, n_repeat_action: int=1):
        super(AtariEnvironment, self).__init__(name=name, n_repeat_action=n_repeat_action)
        self.clone_seeds = clone_seeds
        spec = gym_registry.spec(name)
        # not actually needed, but we feel safer
        spec.max_episode_steps = None
        spec.max_episode_time = None
        self._env = spec.make()

    def __getattr__(self, item):
        return getattr(self._env, item)

    @property
    def n_actions(self):
        return self._env.action_space.n

    def get_state(self) -> np.ndarray:
        if self.clone_seeds:
            return self._env.unwrapped.clone_full_state()
        else:
            return self._env.unwrapped.clone_state()

    def set_state(self, state: np.ndarray):
        if self.clone_seeds:
            self._env.unwrapped.restore_full_state(state)
        else:
            self._env.unwrapped.restore_state(state)
        return state

    def step(self, action: np.ndarray, state: np.ndarray = None,
             n_repeat_action: int = 1) -> tuple:
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        if state is not None:
            self.set_state(state)

        terminal = False
        old_lives = -np.inf
        reward = 0
        for i in range(n_repeat_action):

            obs, _reward, end, info = self._env.step(action)
            lives = info["ale.lives"]
            reward += _reward
            terminal = terminal or end or lives < old_lives
            old_lives = lives
        if state is not None:
            new_state = self.get_state()
            return new_state, obs, reward, terminal, lives
        return obs, reward, terminal, lives

    def step_batch(self, actions, states=None, n_repeat_action: int=None) -> tuple:
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        data = [self.step(action, state, n_repeat_action=n_repeat_action)
                for action, state in zip(actions, states)]
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

    def reset(self, return_state: bool=True):
        if not return_state:
            return self._env.reset()
        else:
            obs = self._env.reset()
            return self.get_state(), obs


def split_similar_chunks(vector: list, n_chunks: int):
    chunk_size = int(np.ceil(len(vector) / n_chunks))
    for i in range(0, len(vector), chunk_size):
        yield vector[i:i + chunk_size]


class ExternalProcess(object):
    """Step environment in a separate process for lock free parallelism.
    It is mostly a copy paste from
    https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
    """

    # Message types for communication via the pipe.
    _ACCESS = 1
    _CALL = 2
    _RESULT = 3
    _EXCEPTION = 4
    _CLOSE = 5

    def __init__(self, constructor):
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
        self._conn, conn = multiprocessing.Pipe()
        self._process = multiprocessing.Process(
            target=self._worker, args=(constructor, conn))
        atexit.register(self.close)
        self._process.start()
        self._observ_space = None
        self._action_space = None

    @property
    def observation_space(self):
        if not self._observ_space:
            self._observ_space = self.__getattr__('observation_space')
        return self._observ_space

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = self.__getattr__('action_space')
        return self._action_space

    def __getattr__(self, name):
        """Request an attribute from the environment.
        Note that this involves communication with the external process, so it can
        be slow.
        Args:
          name: Attribute to access.
        Returns:
          Value of the attribute.
        """
        self._conn.send((self._ACCESS, name))
        return self._receive()

    def call(self, name, *args, **kwargs):
        """Asynchronously call a method of the external environment.
        Args:
          name: Name of the method to call.
          *args: Positional arguments to forward to the method.
          **kwargs: Keyword arguments to forward to the method.
        Returns:
          Promise object that blocks and provides the return value when called.
        """
        payload = name, args, kwargs
        self._conn.send((self._CALL, payload))
        return self._receive

    def close(self):
        """Send a close message to the external process and join it."""
        try:
          self._conn.send((self._CLOSE, None))
          self._conn.close()
        except IOError:
          # The connection was already closed.
          pass
        self._process.join()

    def set_state(self, state, blocking=True):
        promise = self.call('set_state', state)
        if blocking:
            return promise()
        else:
            return promise

    def step_batch(self, actions, states=None, n_repeat_action: int=1, blocking=True):
        promise = self.call('step_batch', actions, states, n_repeat_action)
        if blocking:
            return promise()
        else:
            return promise

    def step(self, action, state=None, n_repeat_action: int=1, blocking=True):
        """Step the environment.
        Args:
          action: The action to apply to the environment.
          blocking: Whether to wait for the result.
        Returns:
          Transition tuple when blocking, otherwise callable that returns the
          transition tuple.
        """

        promise = self.call('step', action, state, n_repeat_action)
        if blocking:
            return promise()
        else:
            return promise

    def reset(self, blocking=True, return_states: bool=False):
        """Reset the environment.
        Args:
          blocking: Whether to wait for the result.
        Returns:
          New observation when blocking, otherwise callable that returns the new
          observation.
        """
        promise = self.call('reset', return_states=return_states)
        if blocking:
            return promise()
        else:
            return promise

    def _receive(self):
        """Wait for a message from the worker process and return its payload.
        Raises:
          Exception: An exception was raised inside the worker process.
          KeyError: The reveived message is of an unknown type.
        Returns:
          Payload object of the message.
        """
        message, payload = self._conn.recv()
        # Re-raise exceptions in the main process.
        if message == self._EXCEPTION:
            stacktrace = payload
            raise Exception(stacktrace)
        if message == self._RESULT:
            return payload
        raise KeyError('Received message of unexpected type {}'.format(message))

    def _worker(self, constructor, conn):
        """The process waits for actions and sends back environment results.
        Args:
          constructor: Constructor for the OpenAI Gym environment.
          conn: Connection for communication to the main process.
        Raises:
          KeyError: When receiving a message of unknown type.
        """
        try:
            env = constructor()
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


class BatchEnv(object):
    """Combine multiple environments to step them in batch.
    It is mostly a copy paste from
    https://github.com/tensorflow/agents/blob/master/agents/tools/wrappers.py
    """

    def __init__(self, envs, blocking):
        """Combine multiple environments to step them in batch.
        To step environments in parallel, environments must support a
        `blocking=False` argument to their step and reset functions that makes them
        return callables instead to receive the result at a later time.
        Args:
          envs: List of environments.
          blocking: Step environments after another rather than in parallel.
        Raises:
          ValueError: Environments have different observation or action spaces.
        """
        self._envs = envs
        self._blocking = blocking
        observ_space = self._envs[0].observation_space
        if not all(env.observation_space == observ_space for env in self._envs):
            raise ValueError('All environments must use the same observation space.')
        action_space = self._envs[0].action_space
        if not all(env.action_space == action_space for env in self._envs):
            raise ValueError('All environments must use the same observation space.')

    def __len__(self):
        """Number of combined environments."""
        return len(self._envs)

    def __getitem__(self, index):
        """Access an underlying environment by index."""
        return self._envs[index]

    def __getattr__(self, name):
        """Forward unimplemented attributes to one of the original environments.
        Args:
          name: Attribute that was accessed.
        Returns:
          Value behind the attribute name one of the wrapped environments.
        """
        return getattr(self._envs[0], name)

    def _make_transitions(self, actions, states=None, n_repeat_action: int=1):
        states = states if states is not None else [None] * len(actions)
        chunks = len(self._envs)
        states_chunk = split_similar_chunks(states, n_chunks=chunks)
        actions_chunk = split_similar_chunks(actions, n_chunks=chunks)
        results = []
        for env, states_batch, actions_batch in zip(self._envs, states_chunk, actions_chunk):
                result = env.step_batch(actions=actions_batch, states=states_batch,
                                        n_repeat_action=n_repeat_action, blocking=self._blocking)
                results.append(result)

        _states = []
        observs = []
        rewards = []
        terminals = []
        infos = []
        for result in results:
            if self._blocking:
                if states is None:
                    obs, rew, ends, info = result
                else:
                    _sts, obs, rew, ends, info = result
                    _states += _sts
            else:
                if states is None:
                    obs, rew, ends, info = result()
                else:
                    _sts, obs, rew, ends, info = result()
                    _states += _sts
            observs += obs
            rewards += rew
            terminals += ends
            infos += info
        if states is None:
            transitions = observs, rewards, terminals, infos
        else:
            transitions = _states, observs, rewards, terminals, infos
        return transitions

    def step_batch(self, actions, states=None, n_repeat_action: int=1):
        """Forward a batch of actions to the wrapped environments.
        Args:
          actions: Batched action to apply to the environment.
          states: States to be stepped. If None, act on current state.
          n_repeat_action: Number of consecutive times the action will be applied.
        Raises:
          ValueError: Invalid actions.
        Returns:
          Batch of observations, rewards, and done flags.
        """
        for index, (env, action) in enumerate(zip(self._envs, actions)):
            if not env.action_space.contains(action):
                message = 'Invalid action at index {}: {}'
                raise ValueError(message.format(index, action))

        if states is None:
            observs, rewards, dones, lives = self._make_transitions(actions, None, n_repeat_action)
        else:
            states, observs, rewards, dones, lives = self._make_transitions(actions, states,
                                                                            n_repeat_action)
        observ = np.stack(observs)
        reward = np.stack(rewards)
        done = np.stack(dones)
        lives = np.stack(lives)
        if states is None:
            return observ, reward, done, lives
        else:
            return states, observs, rewards, dones, lives

    def sync_states(self, state, blocking: bool=True):
        for env in self._envs:
            try:
                env.set_state(state, blocking=blocking)
            except EOFError:
                continue

    def reset(self, indices=None, return_states: bool=True):
        """Reset the environment and convert the resulting observation.
        Args:
          indices: The batch indices of environments to reset; defaults to all.
          return_states: return the corresponding states after reset.
        Returns:
          Batch of observations.
        """
        if indices is None:
            indices = np.arange(len(self._envs))
        if self._blocking:
            observs = [self._envs[index].reset(return_states=return_states) for index in indices]
        else:
            transitions = [self._envs[index].reset(blocking=False,
                                                   return_states=return_states)
                           for index in indices]
            transitions = [trans() for trans in transitions]
            states, observs = zip(*transitions)

        observ = np.stack(observs)
        if return_states:
            return np.array(states), observ
        return observ

    def close(self):
        """Send close messages to the external process and join them."""
        for env in self._envs:
            if hasattr(env, 'close'):
                env.close()


def env_callable(name, env_class, *args, **kwargs):
    return lambda: env_class(name, *args, **kwargs)


class ParallelEnvironment(Environment):
    """Wrap any environment to be stepped in parallel."""

    def __init__(self, name, env_class, n_workers: int=8, blocking: bool=True, *args, **kwargs):
        """

        :param name: Name of the Environment
        :param env_class: Class of the environment to be wrapped.
        :param n_workers: number of workers that will be used.
        :param blocking: step the environments asynchronously.
        :param args: args of the environment that will be parallelized.
        :param kwargs: kwargs of the environment that will be parallelized.
        """
        super(ParallelEnvironment, self).__init__(name=name)
        self._env = env_callable(name, env_class, *args, **kwargs)()
        envs = [ExternalProcess(constructor=env_callable(name, env_class, *args, **kwargs))
                for _ in range(n_workers)]
        self._batch_env = BatchEnv(envs, blocking)

    def __getattr__(self, item):
        return getattr(self._env, item)

    def step_batch(self, actions: np.ndarray, states: np.ndarray=None, n_repeat_action: int=1):
        return self._batch_env.step_batch(actions=actions, states=states,
                                          n_repeat_action=n_repeat_action)

    def step(self, action: np.ndarray, state: np.ndarray=None, n_repeat_action: int=1):
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