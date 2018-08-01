import time
import copy
import numpy as np
from typing import Callable
from fractalai.swarm import Swarm, DynamicTree


class SwarmWave(Swarm):

    tree = DynamicTree()

    def __init__(self, env, model, n_walkers: int=100, balance: float=1.,
                 reward_limit: float=None, samples_limit: int=None, render_every: int=1e10,
                 save_data: bool=True, accumulate_rewards: bool=True, dt_mean: float=None,
                 dt_std: float=None, custom_reward: Callable=None, custom_end: Callable=None,
                 keep_best: bool=False, min_dt: int=1, prune_tree: bool=True,
                 process_obs: Callable=None, can_win: bool=False):
        """
        :param env: Environment that will be sampled.
        :param model: Model used for sampling actions from observations.
        :param n_walkers: Number of walkers that the swarm will use
        :param balance: Balance coefficient for the virtual reward formula.
        :param reward_limit: Maximum reward that can be reached before stopping the swarm.
        :param samples_limit: Maximum number of time the Swarm can sample the environment
         before stopping.
        :param render_every: Number of iterations that will be performed before printing the Swarm
         status.
        """
        super(SwarmWave, self).__init__(env=env, model=model, n_walkers=n_walkers,
                                        balance=balance, reward_limit=reward_limit,
                                        samples_limit=samples_limit, render_every=render_every,
                                        accumulate_rewards=accumulate_rewards, dt_mean=dt_mean,
                                        dt_std=dt_std, custom_end=custom_end,
                                        custom_reward=custom_reward, keep_best=keep_best,
                                        min_dt=min_dt, process_obs=process_obs, can_win=can_win)
        self.save_data = save_data
        self.prune_tree = prune_tree
        self.old_ids = np.zeros(self.n_walkers)
        self._current_index = None
        self._curr_states = []
        self._curr_actions = []
        self._curr_dts = []
        self._current_ix = -1

    def __str__(self):
        text = super(SwarmWave, self).__str__()
        if self.save_data:
            efi = (len(self.tree.data.nodes) / self._n_samples_done) * 100
            sam_step = self._n_samples_done / len(self.tree.data.nodes)
            samples = len(self.tree.data.nodes)
        else:
            efi, samples, sam_step = 0, 0, 0
        new_text = "{}\n"\
                   "Efficiency {:.2f}%\n" \
                   "Generated {} Examples |" \
                   " {:.2f} samples per example.\n".format(text, efi, samples, sam_step)
        return new_text

    def init_swarm(self, state: np.ndarray=None, obs: np.ndarray=None):
        super(SwarmWave, self).init_swarm(state=state, obs=obs)
        self.tree.data.nodes[0]["obs"] = obs if obs is not None else self.env.reset()[1]
        self.tree.data.nodes[0]["terminal"] = False

    def step_walkers(self):
        old_ids = self.walkers_id.copy()
        super(SwarmWave, self).step_walkers()
        if self.save_data:
            for i, idx in enumerate(self.walkers_id):
                self.tree.append_leaf(int(idx), parent_id=int(old_ids[i]),
                                      state=self.data.get_states([idx]).copy()[0],
                                      action=self.data.get_actions([idx]).copy()[0],
                                      dt=copy.deepcopy(self.dt[i]))

    def clone(self):

        super(SwarmWave, self).clone()
        # Prune tree to save memory
        if self.save_data and self.prune_tree:
            dead_leafs = list(set(self._pre_clone_ids) - set(self._post_clone_ids))
            self.tree.prune_tree(dead_leafs, self._post_clone_ids)

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
        states, actions, dts = self.recover_game(index)
        for state, action, dt in zip(states, actions, dts):
            _, _, _, end, _ = self._env.step(action, state=state, n_repeat_action=1)
            self._env.render()
            time.sleep(sleep)
            for i in range(max(0, dt - 1)):
                self._env.step(action, n_repeat_action=1)
                self._env.render()
                time.sleep(sleep)
