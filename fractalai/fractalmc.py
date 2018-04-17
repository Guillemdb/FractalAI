import time
import numpy as np
from typing import Callable
from IPython.core.display import clear_output
from fractalai.model import DiscreteModel
from fractalai.swarm import Swarm, DynamicTree


class FractalMC(Swarm):

    def __init__(self, env, model, max_walkers: int=100, balance: float=1.,
                 time_horizon: int=15,
                 reward_limit: float=None, max_samples: int=None, render_every: int=1e10,
                 custom_reward: Callable=None, custom_end: Callable=None, dt_mean: float=None,
                 dt_std: float=None):
        """
        :param env: Environment that will be sampled.
        :param model: Model used for sampling actions from observations.
        :param max_walkers: Number of walkers that the swarm will use
        :param balance: Balance coefficient for the virtual reward formula.
        :param reward_limit: Maximum reward that can be reached before stopping the swarm.
        :param max_samples: Maximum number of time the Swarm can sample the environment
         befors stopping.
        :param render_every: Number of iterations that will be performed before printing the Swarm
         status.
        """
        self.max_walkers = max_walkers
        self.time_horizon = time_horizon
        self.max_samples = max_samples

        _max_samples = max_samples if max_samples is not None else 1e10
        self._max_samples_step = min(_max_samples, max_walkers * time_horizon)

        super(FractalMC, self).__init__(env=env, model=model, n_walkers=self.max_walkers,
                                        balance=balance, reward_limit=reward_limit,
                                        samples_limit=self._max_samples_step,
                                        render_every=render_every, custom_reward=custom_reward,
                                        custom_end=custom_end, dt_mean=dt_mean, dt_std=dt_std)
        self.init_ids = np.zeros(self.n_walkers).astype(int)

        self._save_steps = []
        self._agent_reward = 0
        self._last_action = None

        self.tree = DynamicTree()

    @property
    def init_actions(self):
        return self.data.get_actions(self.init_ids)

    def init_swarm(self, state: np.ndarray=None, obs: np.ndarray=None):
        self.init_ids = np.zeros(self.n_walkers).astype(int)
        super(FractalMC, self).init_swarm(state=state, obs=obs)

    def clone(self):
        super(FractalMC, self).clone()
        self.init_ids = np.where(self._will_clone, self.init_ids[self._clone_idx], self.init_ids)

    def weight_actions(self) -> np.ndarray:
        """Gets an approximation of the Q value function for a given state.

        It weights the number of times a given initial action appears in each state of the swarm.
        The the proportion of times each initial action appears on the swarm, is proportional to
        the Q value of that action.
        """
        if isinstance(self._model, DiscreteModel):
            counts = np.bincount(self.init_actions)
            return np.argmax(counts)
        vals = self.init_actions.sum(axis=0)
        return vals / self.n_walkers

    def update_data(self):
        init_actions = list(set(np.array(self.init_ids).astype(int)))
        walker_data = list(set(np.array(self.walkers_id).astype(int)))
        self.data.update_values(set(walker_data + init_actions))

    def run_swarm(self, state: np.ndarray=None, obs: np.ndarray=None, print_swarm: bool=False):
        """
        Iterate the swarm by either evolving or cloning each walker until a certain condition
        is met.
        :return:
        """
        self.init_swarm(state=state, obs=obs)
        while not self.stop_condition():
            try:
                # We calculate the clone condition, and then perturb the walkers before cloning
                # This allows the deaths to recycle faster, and the Swarm becomes more flexible
                if self._i_simulation > 1:
                    self.clone_condition()
                self.step_walkers()
                if self._i_simulation > 1:
                    self.clone()
                elif self._i_simulation == 0:
                    self.init_ids = self.walkers_id.copy()
                self._i_simulation += 1
                if self._i_simulation % self.render_every == 0 and print_swarm:
                    print(self)
                    clear_output(True)
            except KeyboardInterrupt:
                break
        if print_swarm:
            print(self)

    def _update_n_samples(self):
        """This will adjust the number of samples we make for calculating an state swarm. In case
        we are doing poorly the number of samples will increase, and it will decrease if we are
        sampling further than the minimum mean time desired.
        """
        limit_samples = self._max_samples_step / self.balance
        # Round and clip
        limit_clean = int(np.clip(np.ceil(limit_samples), 2, self.max_samples))
        self._max_samples_step = limit_clean

    def _update_n_walkers(self):
        """The number of parallel trajectories used changes every step. It tries to use enough
         swarm to make the mean time of the swarm tend to the minimum mean time selected.
         """
        new_n = self.n_walkers * self.balance
        new_n = int(np.clip(np.ceil(new_n), 2, self.max_walkers))
        self.n_walkers = new_n

    def _update_balance(self):
        """The balance parameter regulates the balance between how much you weight the distance of
        each state (exploration) with respect to its score (exploitation).

        A balance of 1 would mean that the computational resources assigned to a given decision
        have been just enough to reach the time horizon. This means that we can assign the same
        importance to exploration and exploitation.

        A balance lower than 1 means that we are not reaching the desired time horizon. This
        means that the algorithm is struggling to find a valid solution. In this case exploration
        should have more importance than exploitation. It also shows that we need to increase the
        computational resources.

        A balance higher than 1 means that we have surpassed the time horizon. This
        means that we are doing so well that we could use less computational resources and still
        meet the time horizon. This also means that we can give exploitation more importance,
        because we are exploring the state space well.
        """
        self.balance = self.times.mean() / self.time_horizon

    def update_parameters(self):
        """Here we update the parameters of the algorithm in order to maintain the average time of
        the state swarm the closest to the minimum time horizon possible.
        """
        self._save_steps.append(int(self._n_samples_done))  # Save for showing while printing.
        self._update_balance()
        if self.balance >= 1:  # We are doing great
            if self.n_walkers == self.max_walkers:
                self._update_n_samples()  # Decrease the samples so we can be faster.
            else:
                self._update_n_walkers()  # Thi will increase the number of swarm

        else:  # We are not arriving at the desired time horizon.
            if self._max_samples_step == self.max_samples:
                self._update_n_walkers()  # Reduce the number of swarm to avoid useless clones.
            else:
                self._update_n_samples()  # Increase the amount of computation.

    def stop_condition(self) -> bool:
        """This sets a hard limit on maximum samples. It also Finishes if all the walkers are dead,
         or the target score reached.
         """
        stop_hard = self._n_samples_done > self._max_samples_step
        stop_score = False if self.reward_limit is None else \
            self.rewards.max() >= self.reward_limit
        stop_terminal = np.logical_or(self._end_cond, self._terminals).all()
        # Define game status so the user knows why game stopped. Only used when printing the Swarm
        if stop_hard:
            self._game_status = "Sample limit reached."
        elif stop_score:
            self._game_status = "Score limit reached."
        elif stop_terminal:
            self._game_status = "All the walkers died."
        else:
            self._game_status = "Playing..."
        return stop_hard or stop_score or stop_terminal

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
        idx = max(list(self.tree.data.nodes)) if index is None else index
        states, actions, dts = self.recover_game(idx)
        for state, action, dt in zip(states, actions, dts):
            self._env.step(action, state=state, n_repeat_action=dt)
            self._env.render()
            time.sleep(sleep)

    def run_agent(self, render: bool = False, print_swarm: bool=False):
        """

        :param render:
        :return:
        """

        self.tree.reset()
        i_step, self._agent_reward, end = 0, 0, False
        self._save_steps = []
        state, obs = self._env.reset(return_state=True)
        self.tree.append_leaf(i_step, parent_id=i_step - 1,
                              state=state, action=0, dt=1)
        while not end and self._agent_reward < self.reward_limit:
            i_step += 1
            self.run_swarm(state=state.copy(), obs=obs)
            action = self.weight_actions()

            state, obs, _reward, _end, info = self._env.step(state=state, action=action)
            self.tree.append_leaf(i_step, parent_id=i_step - 1,
                                  state=state, action=action, dt=self._env.n_repeat_action)
            self._agent_reward += _reward
            self._last_action = action
            end = end or _end

            if render:
                self._env.render()
            if print_swarm:
                print(self)
                clear_output(True)
            self.update_parameters()
