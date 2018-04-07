import time
import copy
import gym
import numpy as np
from typing import Callable
from fractalai.swarm import Swarm, DynamicTree


class SwarmWave(Swarm):

    def __init__(self, env, model, n_walkers: int=100, balance: float=1.,
                 reward_limit: float=None, samples_limit: int=None, render_every: int=1e10,
                 save_data: bool=True, accumulate_rewards: bool=True, dt_mean: float=None,
                 dt_std: float=None,custom_reward: Callable=None, custom_end: Callable=None):
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
                                        samples_limit=samples_limit, render_every=render_every,
                                        accumulate_rewards=accumulate_rewards, dt_mean=dt_mean,
                                        dt_std=dt_std, custom_end=custom_end,
                                        custom_reward=custom_reward)
        self.save_data = save_data
        self.old_ids = np.zeros(self.n_walkers)
        self.tree = DynamicTree() if save_data else None
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

    def step_walkers(self):
        old_ids = self.walkers_id.copy()
        super(SwarmWave, self).step_walkers()
        if self.save_data:
            for i, idx in enumerate(self.walkers_id[self._will_step]):
                self.tree.append_leaf(int(idx), parent_id=int(old_ids[self._will_step][i]),
                                      state=self.data.get_states([idx]).copy()[0],
                                      action=self.data.get_actions([idx]).copy()[0],
                                      dt=copy.deepcopy(self.dt[i]))

    def clone(self):

        super(SwarmWave, self).clone()
        # Prune tree to save memory
        if self.save_data:
            dead_leafs = self._pre_clone_ids - self._post_clone_ids
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
            self._env.step(action, state=state, n_repeat_action=1)
            self._env.render()
            time.sleep(sleep)
            for i in range(max(0, dt - 1)):
                self._env.step(action, n_repeat_action=1)
                self._env.render()
                time.sleep(sleep)


class AtariDataProvider(SwarmWave):

    def __init__(self, rgb=True,  *args, **kwargs):
        super(AtariDataProvider, self).__init__(*args, **kwargs)
        self.rgb = rgb
        data_env_name = self._env.name.replace("-ram", "") if rgb else self._env.name
        self._played_games = []
        self.data_env = gym.make(data_env_name)
        self.data_env.reset()

    def __getattr__(self, item):
        return getattr(self.data_env, item)

    def run_swarm(self, state: np.ndarray=None, obs: np.ndarray=None, print_swarm: bool=False):
        super(AtariDataProvider, self).run_swarm(state=state, obs=obs, print_swarm=print_swarm)
        self._played_games = list(set(self._post_clone_ids))

    @property
    def name(self):
        return self._env.name

    def process_game(self, index):
        self.data_env.reset()
        states, actions, dts = self.recover_game(index)
        observs, data_actions, rewards, ends = [], [], [], []
        for i, (state, action, dt) in enumerate(zip(states, actions, dts)):
            self.data_env.unwrapped.restore_full_state(state)
            obs, reward, end, info = self.data_env.step(action)
            observs.append(copy.copy(obs))
            rewards.append(copy.copy(reward))
            ends.append(copy.copy(end))
            # this syncs the actions to the obs where they are taken
            _oh_action = np.zeros(self.data_env.action_space.n)
            _oh_action[action] = 1
            if i > 0:
                data_actions.append(copy.copy(_oh_action))
            for j in range(max(0, dt - 1)):
                obs, reward, end, info = self.data_env.step(action)
                observs.append(copy.copy(obs))
                rewards.append(copy.copy(reward))
                ends.append(copy.copy(end))
                data_actions.append(copy.copy(_oh_action))
        return observs[:-1], rewards[:-1], ends[:-1], data_actions

    def _recover_random_game(self):
        if len(self._played_games) == 0:
            self.run_swarm()
        index = np.random.choice(self._played_games)
        game = self.recover_game(index)
        self._played_games.remove(index)
        return game

    def step_generator(self):
        self.data_env.reset()
        states, actions, dts = self._recover_random_game()
        for i, (state, action, dt) in enumerate(zip(states, actions, dts)):
            self.data_env.unwrapped.restore_full_state(state)
            for j in range(dt):
                try:
                    obs, reward, end, info = self.data_env.step(action)
                except RuntimeError:
                    self.data_env.reset()
                    break
                _oh_action = np.zeros(self.data_env.action_space.n)
                _oh_action[action] = 1

                if i == len(actions) - 1 and j == dt - 1:
                    end = True
                yield obs, reward, end, _oh_action
                if end:
                    self.data_env.reset()

    def game_generator(self):
        for game in set(self.walkers_id):
            yield self.process_game(game)


