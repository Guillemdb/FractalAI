import numpy as np
import gym
from gym.wrappers import Monitor
from fractalai.policy import PolicyWrapper
from fractalai.state import AtariState, Microstate


class AtariMonitor:

    def __init__(self, name: str, directory: str, clone_seeds: bool=True,
                 fixed_steps: int=1, force: bool=False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):
        self.name = name
        self.state = AtariState()
        self.directory = directory
        self.force = force
        self.resume = resume
        self.write_upon_reset = write_upon_reset
        self.uid = uid
        self.mode = mode

        self.fixed_steps = fixed_steps
        self.clone_seeds = clone_seeds
        self._cum_reward = None
        self._env = self._init_monitor()

    @property
    def env(self) -> Monitor:
        return self._env

    def _init_monitor(self) -> Monitor:
        env = gym.make(self.name)
        monitor = Monitor(env, directory=self.directory, force=self.force, resume=self.resume,
                          write_upon_reset=self.write_upon_reset, uid=self.uid, mode=self.mode)
        return monitor

    def _get_microstate(self) -> Microstate:
        if self.clone_seeds:
            microstate = self.env.unwrapped.ale.cloneSystemState()

        else:
            microstate = self.env.unwrapped.ale.cloneState()

        return Microstate(self.env, microstate)

    def step(self, action: np.ndarray, fixed_steps: int=None) -> AtariState:
        fixed_steps = self.fixed_steps if fixed_steps is None else fixed_steps
        end = False
        _dead = False
        for i in range(fixed_steps):
            observed, reward, _end, lives = self.env.step(action.argmax())
            end = end or _end
            _dead = _dead or reward < 0
            self._cum_reward += reward
            self.env.render()
            if end:
                break

        microstate = self._get_microstate()
        self.state.update_state(observed=observed, reward=self._cum_reward,
                                end=end, lives=lives, microstate=microstate)
        return self.state.create_clone()

    def reset(self) -> AtariState:
        observed = self.env.reset()
        microstate = self._get_microstate()
        self._cum_reward = 0
        self.state.update_state(observed=observed, reward=self._cum_reward,
                                end=False, lives=-1000, microstate=microstate)
        return self.state.create_clone()

    def skip_frames(self, n_frames: int=0) -> AtariState:
        action = np.zeros(self.env.action_space.n)
        ix = self.env.action_space.sample()
        action[ix] = 1
        for i in range(n_frames):
            self.step(action=action)
            if self.state.terminal:
                break
        return self.state.create_clone()


class AtariMonitorPolicy(PolicyWrapper):

    def __init__(self, policy, directory: str, clone_seeds: bool = True,
                 fixed_steps: int = 1, force: bool = False, resume=False,
                 write_upon_reset=False, uid=None, mode=None):

        super(AtariMonitorPolicy, self).__init__(policy=policy)
        self.monitor = AtariMonitor(name=self.policy.name, directory=directory,
                                    clone_seeds=clone_seeds, fixed_steps=fixed_steps, force=force,
                                    write_upon_reset=write_upon_reset, uid=uid, resume=resume,
                                    mode=mode)

    def record_video(self, skip_frames: int=0):
        from IPython.core.display import clear_output
        self.monitor.reset()
        moni_state = self.monitor.skip_frames(n_frames=skip_frames)
        episode_len = skip_frames
        while not moni_state.terminal:
            action = self.policy.predict(moni_state)
            moni_state = self.monitor.step(action=action)
            episode_len += self.monitor.fixed_steps
            print("Step: {} Score: {}\n {}".format(episode_len, moni_state.reward, self.policy))
            clear_output(True)
            if moni_state.terminal:
                break
        self.monitor.env.close()

    def run(self, skip_frames: int=0):
        try:
            self.record_video(skip_frames)
        except KeyboardInterrupt:
            self.monitor.env.close()
