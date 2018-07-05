import numpy as np
import copy
import time
from fractalai.datasets.data_generator import Swarm, DLTree
from fractalai.swarm_wave import SwarmWave
from fractalai.fractalmc import FractalMC


class MLWave(SwarmWave):

    def __init__(self, *args, **kwargs):

        super(MLWave, self).__init__(*args, **kwargs)
        self.old_obs = None
        self.tree = DLTree()
        self.env.reset()

    def __str__(self):
        text = super(SwarmWave, self).__str__()
        sam_step = self._n_samples_done / len(self.tree.data.nodes)
        samples = len(self.tree.data.nodes)

        new_text = "{}\n"\
                   "Generated {} Examples |" \
                   " {:.2f} samples per example.\n".format(text, samples, sam_step)
        return new_text

    def collect_data(self, *args, **kwargs):
        self.run_swarm(*args, **kwargs)

    def get_best_id(self):
        return self.walkers_id[self.rewards.argmax()]

    def step_walkers(self):
        old_ids = self.walkers_id.copy()
        self.old_obs = self.observations.copy()
        Swarm.step_walkers(self)
        if self.save_data:
            for i, idx in enumerate(self.walkers_id):
                #print(self.data.get_infos([idx])[0])
                self.tree.append_leaf(int(idx), parent_id=int(old_ids[i]),
                                      state=self.data.get_states([idx])[0],
                                      action=self.data.get_actions([idx])[0],
                                      dt=self.dt[i],
                                      reward=float(self.rewards[i]),
                                      terminal=bool(self.data.get_infos([idx])[0]["terminal"]),
                                      obs=self.observations[i])

    def recover_game(self, index=None) -> tuple:
        """
        By default, returns the game sampled with the highest score.
        :param index: id of the leaf where the returned game will finish.
        :return: a list containing the observations of the target sampled game.
        """
        if index is None:
            index = self.walkers_id[self.rewards.argmax()]
        return self.tree.get_branch(index)

    def render_game(self, index=None, sleep: float=0.02):
        """Renders the game stored in the tree that ends in the node labeled as index."""
        states, actions, dts, ends, observs, olds = self.recover_game(index)
        _obs = self._env._env.reset()
        petar = False
        self._env.set_state(states[0])
        for state, action, dt, end, obs, old_obs in zip(states, actions, dts, ends, observs, olds):
            _old_obs = _obs
            _, _obs, _, _end, _ = self._env._env.step(action, state=state, n_repeat_action=1)
            self._env._env.render()
            time.sleep(sleep)
            if petar:
                raise ValueError("Rendering a dead state")
            if _end:
                petar = True

            if not np.allclose(obs, _obs):
                print(obs - _obs, old_obs - _old_obs)
                #raise ValueError("obs petan")
            #for i in range(max(0, dt - 1)):
            #    self._env.step(action, n_repeat_action=1)
            #    self._env.render()
            #    time.sleep(sleep)


class MLFMC(FractalMC):

    def __init__(self, data_env, true_min_dt,  *args, **kwargs):

        super(MLFMC, self).__init__(*args, **kwargs)
        self.old_obs = None
        self.data_env = data_env
        self.true_min_dt = true_min_dt
        self.tree = DLTree()
        self._best_id = 0
        self.env.reset()

    def get_best_id(self):
        return self._best_id

    def recover_game(self, index=None) -> tuple:
        """
        By default, returns the game sampled with the highest score.
        :param index: id of the leaf where the returned game will finish.
        :return: a list containing the observations of the target sampled game.
        """
        if index is None:
            index = self.walkers_id[self.rewards.argmax()]
        return self.tree.get_branch(index)

    def render_game(self, index=None, sleep: float=0.02):
        """Renders the game stored in the tree that ends in the node labeled as index."""
        states, actions, dts, ends, observs, olds = self.recover_game(index)
        _obs = self._env._env.reset()
        petar = False
        self._env.set_state(states[0])
        for state, action, dt, end, obs, old_obs in zip(states, actions, dts, ends, observs, olds):
            _old_obs = _obs
            _, _obs, _, _end, _ = self._env._env.step(action, state=state, n_repeat_action=1)
            self._env._env.render()
            time.sleep(sleep)
            if petar:
                raise ValueError("Rendering a dead state")
            if _end:
                petar = True

            if not np.allclose(obs, _obs):
                print(obs - _obs, old_obs - _old_obs)

    def collect_data(self, render: bool = False, print_swarm: bool=False):
        """

        :param render:
        :param print_swarm:
        :return:
        """

        self.tree.reset()
        self.env.reset()
        state, obs = self.data_env.reset(return_state=True)
        i_step, self._agent_reward, end = 0, 0, False
        _end = False
        for i in range(self.skip_initial_frames):
            i_step += 1
            action = 0

            state, obs, _reward, _end, info = self.data_env.step(state=state, action=action,
                                                                 n_repeat_action=1)
            self.tree.append_leaf(i_step, parent_id=i_step - 1,
                                  state=state, action=np.ones(self.env.n_actions),
                                  dt=1,
                                  reward=np.ones(self.env.n_actions),
                                  terminal=bool(info["terminal"]),
                                  obs=obs)

            self._agent_reward += _reward
            self._last_action = action
            end = info.get("terminal", _end)
            if end:
                break
        self._save_steps = []

        while not end and self._agent_reward < self.reward_limit:
            i_step += 1
            self.run_swarm(state=copy.deepcopy(state), obs=obs)
            action_dist, reward_dist = self.estimate_distributions(state=state, obs=obs)
            action = (action_dist + reward_dist).argmax()
            state, obs, _reward, _end, info = self.data_env.step(state=state,
                                                                 action=action,
                                                                 n_repeat_action=1)
            self.tree.append_leaf(i_step, parent_id=i_step - 1,
                                  state=state, action=action_dist,
                                  dt=1,
                                  reward=reward_dist,
                                  terminal=bool(_end),
                                  obs=obs)

            self._agent_reward += _reward
            self._last_action = action
            end = info.get("terminal", _end)
            self._best_id = i_step

            if render:
                self.data_env.render()
            if print_swarm:
                from IPython.core.display import clear_output
                print(self)
                clear_output(True)
            if self._update_parameters:
                self.update_parameters()
