import numpy as np
from fractalai.state import State
from fractalai.policy import PolicyWrapper, Policy


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Returns normalized values that sum 1.
    :param vector: array to be normalized.
    :return: Normalized vector.
    """
    reward = vector
    max_r, min_r = reward.max(), reward.min()
    if min_r == max_r:
        reward = np.ones(len(vector), dtype=np.float32)
        return reward
    reward = (reward - min_r) / (max_r - min_r)
    return reward


class FractalAI(PolicyWrapper):

    def __init__(self,
                 policy: Policy,
                 max_samples: int = 1500,
                 max_states: int=1000,
                 time_horizon: int = 25,
                 n_fixed_steps: int=1):
        """

        :param policy: Policy that will be used as a prior when calculating an action.
        :param max_samples: Max number of times that we can step the environment to calculate an
         action for a given state.
        :param max_states: Maximum number of states that will be used to build a swarm.
        :param time_horizon: Desired time horizon of the swarm
        :param n_fixed_steps:
        """
        # policy._model = policy.model
        super(FractalAI, self).__init__(policy=policy)

        self.n_fixed_steps = n_fixed_steps

        self.time_horizon = time_horizon
        self.max_states = max_states
        self.max_samples = max_samples

        self.n_states = max_states
        self.balance = 1
        self._i_simulation = 0
        self._n_samples_done = 0
        # The ideal number of samples is max_state * time_horizon
        init_samples = min(max_samples, time_horizon * max_states)
        self._n_limit_samples = init_samples

        self._root_state = policy.reset()
        self.root_state.update_policy_data(0)  # Init time step w.r. to the present state
        # This is the states pool where max_states states are stored.
        self._states = np.array([self.root_state.create_clone()
                                 for _ in range(self.max_states)], dtype=type(self.root_state))

        self._will_clone = np.zeros(self.n_states, dtype=np.bool)
        self._dead_mask = np.zeros(self.n_states, dtype=np.bool)
        self._will_step = np.zeros(self.n_states, dtype=np.bool)
        self._state_times = np.ones(self.n_states)
        self._save_steps = []

    def __str__(self):
        """Print stats about the last step performed by the agent."""

        # Time stats
        times = self.state_times
        mean_time = times.mean()
        std_time = times.std()

        # Death stats
        dead = np.array([s.dead for s in self.states]).sum()
        death_pct = dead / len(self.states) * 100

        # Worker stats
        cloned = self._will_clone.sum() / len(self.states) * 100
        step = self._will_step.sum() / len(self.states) * 100

        text = "Fractal with {} states.\n" \
               "Balance Coef {:.4f} | Mean Samples {:.2f} | Current sample limit {}\n" \
               "Algorithm last iter:\n" \
               "- Deaths: \n" \
               "    {} states: {:.2f}%\n" \
               "- Time: \n" \
               "    Mean: {:.2f} | Std: {:.2f}\n" \
               "- Worker: \n" \
               "    Step: {:.2f}% {} steps | Clone: {:.2f}% {} clones\n"

        return text.format(len(self.states),
                           self.balance, np.mean(self._save_steps), self._n_limit_samples,
                           dead, death_pct,
                           mean_time, std_time,
                           step, self._will_step.sum(), cloned, self._will_clone.sum())

    @property
    def root_state(self) -> State:
        """We are calculating the Q values for each action with respect to the root state."""
        return self._root_state

    @property
    def states(self) -> np.ndarray:
        """Return the states that are currently used."""
        return self._states[:self.n_states]

    @property
    def state_times(self) -> np.ndarray:
        """Array containing the time of each state with respect to the root state."""
        return self._state_times

    @property
    def deaths(self) -> np.ndarray:
        """Contains booleans indicating whether an state meets the death condition or not."""
        return np.array([s.dead for s in self.states])

    @property
    def init_actions(self):
        """The actions taken at the root state. They are used for approximating the Q values."""
        return np.array([s.policy_action for s in self.states])

    def _action_probabilities(self) -> np.ndarray:
        """Gets an approximation of the Q value function for a given state.

        It weights the number of times a given initial action appears in each state of the swarm.
        The the proportion of times each initial action appears on the swarm, is proportional to
        the Q value of that action.
        """
        vals = self.init_actions.sum(axis=0)
        return vals / self.n_states

    def _init_step(self):
        """Sync all the states so they become a copy of the root state"""
        self.states[:] = [self.root_state.create_clone() for _ in range(self.n_states)]
        self._will_clone = np.zeros(self.n_states, dtype=np.bool)
        self._state_times = np.ones(self.n_states)

    def _step_states(self, init_step: bool = False, n_fixed_steps: int = 1):
        """Sample an action for each walker, and act on the system.
        Keep track of the action chosen if init_step
        :param init_step: If it is the first step, keep track of the initial actions taken.
        :param n_fixed_steps: Number of consecutive steps the same action will be applied.
        :return: None.
        """
        # Only step an state if it has not cloned
        step_ix = np.logical_not(self._will_clone) if not init_step else np.ones(len(self.states),
                                                                                 dtype=bool)

        actions = self.policy.predict(self.states[step_ix])
        self.states[step_ix] = self.step(self.states[step_ix], actions,
                                         fixed_steps=n_fixed_steps)
        # Save action taken from root state inside
        if init_step:
            for ix, state in enumerate(self.states[step_ix]):
                state.update_policy_action(actions[ix])  # Save init action
                state.update_policy_data(0)  # Set simulation time to zero.
        self._n_samples_done += step_ix.sum()
        self._will_step = step_ix

    def _evaluate_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different images
        on a vector of states, and normalizes the relative distances between 1 and 2.
        In a more general scenario, any function that quantifies the notion of "how different two
        states are" could work, even though it is not a proper distance.
        """
        # Get random companion
        idx = np.random.permutation(np.arange(len(self.states)))
        obs = np.array([state.observed for state in self.states])
        # Euclidean distance between pixels
        dist = np.sqrt(np.sum((obs[idx] - obs) ** 2, axis=tuple(range(1, len(obs.shape)))))
        return normalize_vector(dist)

    def _entropic_reward(self) -> np.ndarray:
        """Calculate the entropic reward of the walkers. This quantity is used for determining
        the chance a given state has of cloning. This scalar gives us a measure of how well an
        state is solving the exploration vs exploitation problem.
        """
        dist = self._evaluate_distance()  # goes between 0 and 1
        rewards = np.array([state.reward for state in self.states])
        scores = normalize_vector(rewards) + 1  # Goes between 1 and 2
        # The balance sets how much preference we are giving exploitation over exploration.
        ent_reward = dist * scores ** self.balance
        return ent_reward

    def _update_time_step(self):
        """Keep track of the time of each state in the cone, with respect to the present state.
        In case the state has been stepped, increase time by 1.
        """
        for state in self.states[self._will_step]:
            # The current simulation time is stored inside the policy_data attribute of each state.
            current_ts = state.policy_data if state.policy_data is not None else 0
            state.update_policy_data(current_ts + 1)
        self._state_times = np.array([s.policy_data for s in self.states])

    def _clone(self):
        """The clone phase aims to change the distribution of walkers in the state space, by
         cloning some states to a randomly chosen companion. After cloning, the distribution of
         walkers will be closer to the reward distribution of the state space.
        1 - Choose a random companion who is alive.
        2 - Calculate the probability of cloning.
        3 - Clone if True or the walker is dead.
        """
        ent_rew = self._entropic_reward()

        # Besides the death condition, zero entropic reward also counts as death when cloning
        deaths = np.array([s.dead or ent_rew[i] == 0 for i, s in enumerate(self.states)])
        # The following line is tailored to atari games. Beware when using in a general case.
        deaths = np.logical_or(deaths, [s.reward < self.root_state.reward for s in self.states])
        self._dead_mask = deaths

        idx = self._get_companions()

        # The probability of cloning depends on the relationship of entropic rewards
        # with respect to a randomly chosen walker.
        value = (ent_rew[idx] - ent_rew) / np.where(ent_rew > 0, ent_rew, 1e-8)
        clone = (value >= np.random.random()).astype(bool)

        clone[deaths] = True
        clone[deaths[idx]] = False  # Do not clone to a dead companion
        # Actual cloning
        self._will_clone = clone
        for i, clo in enumerate(clone):
            if clo:
                self.states[i] = self.states[idx][i].create_clone()

    def _act(self, state: State=None, render: bool = False, *args, **kwargs) -> State:
        """ Given an arbitrary state, acts on the environment and return the
        processed new state where the system ends up.
        :param state: State that represents the current state of the environment.
        :param render: If true, the environment will be displayed.
        :param args: Will be passed to predict_proba.
        :param kwargs: Will be passed to predict_proba.
        :return: Next state of the system.
        """
        # If state not provided we will act on the root_state
        if state is not None:
            self.set_root_state(state)
        model_action = self.model.predict(self.root_state)
        self._model_pred = model_action
        action = self._predict(self.root_state, *args, **kwargs)
        self._last_pred = action
        for _ in range(self.n_fixed_steps):
            new_state = self.step(self.root_state, action)
            if render:
                self.env.render()
        new_state.update_policy_action(action)
        new_state.update_model_action(model_action)
        return new_state

    def _get_companions(self) -> np.ndarray:
        """
        Return an array of indices corresponding to a walker chosen at random.
        Self selection can happen.
        :return: np.ndarray containing the indices of the companions to be chosen for each walker.
        """
        return np.random.permutation(np.arange(self.n_states))

    def _update_parameters(self):
        """Here we update the parameters of the algorithm in order to maintain the average time of
        the state swarm the closest to the minimum time horizon possible.
        """
        self._save_steps.append(int(self._n_samples_done))  # Save for showing while printing.
        self._update_balance()
        if self.balance >= 1:  # We are doing great
            if self.n_states == self.max_states:
                self._update_n_samples()  # Decrease the samples so we can be faster.
            else:
                self._update_n_states()  # Thi will increase the number of states

        else:  # We are not arriving at the desired time horizon.
            if self._n_limit_samples == self.max_samples:
                self._update_n_states()  # Reduce the number of states to avoid useless clones.
            else:
                self._update_n_samples()  # Increase the amount of computation.

    def _update_n_samples(self):
        """This will adjust the number of samples we make for calculating an state swarm. In case
        we are doing poorly the number of samples will increase, and it will decrease if we are
        sampling further than the minimum mean time desired.
        """
        limit_samples = self._n_limit_samples / self.balance
        # Round and clip
        limit_clean = int(np.clip(np.ceil(limit_samples), 2, self.max_samples))
        self._n_limit_samples = limit_clean

    def _update_n_states(self):
        """The number of parallel trajectories used changes every step. It tries to use enough
         states to make the mean time of the swarm tend to the minimum mean time selected.
         """
        old_n = int(self.n_states)
        new_n = self.n_states * self.balance
        new_n = int(np.clip(np.ceil(new_n), 2, self.max_states))
        self.n_states = new_n
        # New states start as a copy of already existing states.
        if new_n > old_n:
            self._states[old_n:new_n] = [x.create_clone() for x
                                         in np.random.choice(self.states, int(new_n - old_n))]
        self.n_states = int(new_n)

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
        self.balance = self.state_times.mean() / self.time_horizon

    def _predict(self, state: State,
                 n_fixed_steps: int = None) -> np.ndarray:
        """
        Probability distribution over each action for the current state.
        This is done by propagating and cloning the walkers to build a causal cone.
        The cone is n_states wide and depth long.
        :param state: State from you want the actions to be predicted.
        :param n_fixed_steps: Number of steps that the same action will be applied.
        :return: Array containing the probability distribution over each action.
        """
        fixed_steps = self.n_fixed_steps if n_fixed_steps is None else n_fixed_steps
        self.set_root_state(state)
        self._n_samples_done = 0
        self._i_simulation = 1

        while self._n_samples_done <= self._n_limit_samples:
            if self._i_simulation > 1:
                self._clone()
            reset_walkers = self._i_simulation == 1 or (self.deaths.all() and
                                                        not self.root_state.dead)

            if reset_walkers:
                self._init_step()
            self._step_states(init_step=reset_walkers, n_fixed_steps=fixed_steps)
            self._i_simulation += 1
            self._update_time_step()

        self._update_parameters()
        action_probs = self._action_probabilities()
        return action_probs

    def set_root_state(self, state: [State, list, np.ndarray]):
        """Set the present state of the environment that the fractal is in externally."""
        if isinstance(state, (list, np.ndarray)):
            if len(state) == 1:
                state = state[0]
            else:
                raise ValueError("State should be a State or a vector of length 1, got {} instead"
                                 .format(state))
        self._root_state = state.create_clone()
        self._root_state.update_policy_data(0)  # Init time step w.r. to the present state

    def evaluate(self, skip_frames: int = 0, render=False,
                 max_steps=100000, *args, **kwargs) -> tuple:
        """Evaluate the current model by playing one episode."""
        self.balance = 1
        self._n_limit_samples = self.max_samples
        episode_len = 0

        self.env.reset()
        state = self.skip_frames(n_frames=skip_frames)
        while not state.terminal and episode_len < max_steps:
            state = self.act(state, render=render, *args, **kwargs)
            episode_len += 1
            if render:
                # This will also print the internal stats of the fractal after each step.
                from IPython.core.display import clear_output
                clear_output(True)
                print("Step: {} Score: {}\n {}".format(episode_len, state.reward, self))
        return state.reward, episode_len
