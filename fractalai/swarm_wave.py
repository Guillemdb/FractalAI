import time
import numpy as np
from fractalai.swarm import Swarm, DynamicTree


class SwarmWave(Swarm):

    def __init__(self, env, model, n_walkers: int=100, balance: float=1.,
                 reward_limit: float=None, samples_limit: int=None, render_every: int=1e10,
                 save_data: bool=True):
        """
        :param env: Environment that will be sampled.
        :param model: Model used for sampling actions from observations.
        :param n_walkers: Number of walkers that the swarm will use
        :param balance: Balance coefficient for the virtual reward formula.
        :param reward_limit: Maximum reward that can be reached before stopping the swarm.
        :param samples_limit: Maximum number of time the Swarm can sample the environment
         befors stopping.
        :param render_every: Number of iterations that will be performed before printing the Swarm
         status.
        """
        super(SwarmWave, self).__init__(env=env, model=model, n_walkers=n_walkers,
                                        balance=balance, reward_limit=reward_limit,
                                        samples_limit=samples_limit, render_every=render_every)
        self.save_data = save_data
        self.old_ids = np.zeros(self.n_walkers)
        self.tree = DynamicTree() if save_data else None

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

    def step_walkers(self):
        old_ids = self.walkers_id.copy()
        super(SwarmWave, self).step_walkers()
        if self.save_data:
            for idx in np.arange(self.n_walkers)[self._will_step]:
                self.tree.append_leaf(int(self.walkers_id[idx]), parent_id=int(old_ids[idx]),
                                      state=self.data.get_states([self.walkers_id[idx]]).copy()[0],
                                      action=self.data.get_actions([self.walkers_id[idx]]
                                                                   ).copy()[0])

    def clone(self):
        pre_clone_ids = set(self.walkers_id.astype(int))
        super(SwarmWave, self).clone()
        # Prune tree to save memory
        if self.save_data:
            post_clone_ids = set(self.walkers_id.astype(int))
            dead_leafs = pre_clone_ids - post_clone_ids
            self.tree.prune_tree(dead_leafs, post_clone_ids)

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
        states, actions = self.recover_game(index)
        for state, action in zip(states, actions):
            self._env.step(action, state=state)
            self._env.render()
            time.sleep(sleep)



