import numpy as np
import networkx as nx
import copy
from IPython.core.display import clear_output


def normalize_vector(vector):
    """
    Returns normalized values where min = 0 and max = 1.
    :param vector: array to be normalized.
    :return: Normalized vector.
    """
    max_r, min_r = np.max(vector), np.min(vector)
    if min_r == max_r:
        _reward = np.ones(len(vector), dtype=np.float32)
    else:
        _reward = (vector - min_r) / (max_r - min_r)
    return _reward


class DataStorage:

    def __init__(self):
        self.states = {}
        self.actions = {}

    def __getitem__(self, item):
        states = self.get_states(item)
        actions = self.get_actions(item)
        return states, actions

    def reset(self):
        self.states = {}
        self.actions = {}

    def get_states(self, labels):
        return [self.states[label] for label in labels]

    def get_actions(self, labels):
        return [self.actions[label] for label in labels]

    def append(self, walker_ids, states, actions=None):

        for w_id, state, action in zip(walker_ids, states, actions):
            self.states[w_id] = copy.deepcopy(state)
            if actions is not None:
                self.actions[w_id] = copy.deepcopy(action)

    def delete(self, walker_ids):
        new_states = {}
        new_actions = {}
        for w_id in walker_ids:

            new_states[w_id] = self.states[w_id]
            new_actions[w_id] = self.states[w_id]
        self.states = new_states
        self.actions = new_actions


class DynamicTree:
    """This is a tree data structure that stores the paths followed by the walkers. It can be
    pruned to delete paths that will no longer be needed. It uses a networkx Graph. If someone
    wants to spend time building a proper data structure, please make a PR, and I will be super
    happy!
    """

    def __init__(self):
        self.data = nx.DiGraph()
        self.data.add_node(0)
        self.root_id = 0

    def reset(self):
        self.data = nx.DiGraph()
        self.data.add_node(0)
        self.root_id = 0

    def append_leaf(self, leaf_id: int, parent_id: int, state, action):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :return:
        """
        self.data.add_node(int(leaf_id), state=state)
        self.data.add_edge(int(parent_id), int(leaf_id), action=action)

    def prune_branch(self, leaf, alive_leafs):
        """This recursively prunes a branch that only leads to an orphan leaf."""
        parent_id = self.data.in_edges([leaf])
        parent_ids = list(parent_id)
        for parent, _ in parent_ids:

            if parent == 0 or leaf == 0:
                return
            if len(self.data.out_edges([leaf])) == 0:
                self.data.remove_edge(parent, leaf)
                self.data.remove_node(leaf)

                if len(self.data.out_edges([parent])) == 0 and parent not in alive_leafs:
                    return self.prune_branch(parent, alive_leafs)
        return

    def prune_tree(self, dead_leafs, alive_leafs):
        """This prunes the orphan leaves that will no longer be used to save memory."""
        for leaf in dead_leafs:
            self.prune_branch(leaf, alive_leafs)
        return

    def get_branch(self, leaf_id) -> tuple:
        """
        Get the observation from the game ended at leaf_id
        :param leaf_id: id of the leaf node belonging to the branch that will be recovered.
        :return: Sequence of observations belonging to a given branch of the tree.
        """
        nodes = nx.shortest_path(self.data, 0, leaf_id)
        states = [self.data.node[n]["state"] for n in nodes]
        actions = [self.data.edges[(n, nodes[i+1])]["action"] for i, n in enumerate(nodes[:-1])]
        return states, actions


class Swarm:

    """This is the most basic mathematical entity that can be derived from Fractal AI theory.
    It represents a cloud of points that propagates through an state space. Each walker of the
    swarm evolves by either cloning to another walker or perturbing the environment.
    """

    def __init__(self, env, model, n_walkers: int=100, balance: float=1.,
                 reward_limit: float=None, samples_limit: int=None, render_every: int=1e10,
                 accumulate_rewards: bool=True):
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
        # Parameters of the algorithm
        self.samples_limit = samples_limit if samples_limit is not None else np.inf
        self.reward_limit = reward_limit if reward_limit is not None else np.inf
        self._model = model
        self._env = env
        self.n_walkers = n_walkers
        self.balance = balance
        self.render_every = render_every
        self.accumulate_rewards = accumulate_rewards
        # Environment information sources
        self.observations = None
        self.rewards = None
        self.infos = None

        # Internal masks
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        self._will_step = np.ones(self.n_walkers, dtype=bool)
        self._not_frozen = np.ones(self.n_walkers, dtype=bool)
        self._clone_idx = None
        # Processed information sources
        self.virtual_rewards = np.zeros(self.n_walkers)
        self.distances = np.zeros(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self._n_samples_done = 0
        self._i_simulation = 0
        self._game_status = ""
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        self.data = DataStorage()
        self._pre_clone_ids = [0]
        self._post_clone_ids = [0]
        self._remove_id = [None]

    def __str__(self):
        """Print information about the internal state of the swarm."""

        progress = 0 if self.samples_limit is None \
            else (self._n_samples_done / self.samples_limit) * 100

        if self.reward_limit is not None:
            score_prog = (self.rewards.max() / self.reward_limit) * 100
            progress = max(progress, score_prog)

        text = "Environment: {} | Walkers: {} | Deaths: {} | data_size {}\n" \
               "Total samples: {} Progress: {:.2f}%\n" \
               "Reward: mean {:.2f} | Dispersion: {:.2f} | max {:.2f} | min {:.2f} | std {:.2f}\n"\
               "Episode length: mean {:.2f} | Dispersion {:.2f} | max {:.2f} | min {:.2f} " \
               "| std {:.2f}\n" \
               "Status: {}".format(self._env.name, self.n_walkers, self._end_cond.sum(), len(self.data.states.keys()),
                                   self._n_samples_done, progress,
                                   self.rewards.mean(), self.rewards.max() - self.rewards.min(),
                                   self.rewards.max(), self.rewards.min(), self.rewards.std(),
                                   self.times.mean(), self.times.max() - self.times.min(),
                                   self.times.max(), self.times.min(),
                                   self.times.std(), self._game_status)
        return text

    @property
    def actions(self):
        return self.data.get_actions(self.walkers_id)

    @property
    def states(self):
        return self.data.get_states(self.walkers_id)

    def init_swarm(self, state: np.ndarray=None, obs: np.ndarray=None):
        """
        Synchronize all the walkers to a given state, and clear all the internal data of the swarm.
        :param state: State that all the walkers will copy. If None, a new game is started.
        :param obs: Observation corresponding to state. If None, a new game is started.
        :return:
        """
        self._game_status = ""
        if state is None or obs is None:
            state, obs = self._env.reset(return_state=True)
            obs = obs.astype(np.float32)
        # Environment Information sources

        self.observations = np.array([obs.copy() for _ in range(self.n_walkers)])
        self.rewards = np.zeros(self.n_walkers, dtype=np.float32)
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        self.infos = -1 * np.ones(self.n_walkers, dtype=bool)

        # Internal masks
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        self._will_step = np.ones(self.n_walkers, dtype=bool)
        # Processed information sources
        self.virtual_rewards = np.zeros(self.n_walkers)
        self.distances = np.zeros(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self._n_samples_done = 0
        self._i_simulation = 0
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        self.data.reset()
        actions = self._model.predict_batch(self.observations)
        states = np.array([copy.deepcopy(state) for _ in range(self.n_walkers)])
        self.data.append(walker_ids=self.walkers_id, states=states, actions=actions)

    def step_walkers(self):
        """Sample an action for each walker, and act on the environment. This is how the Swarm
        evolves.
        :return: None.
        """
        # Only step an state if it has not cloned and is not frozen

        actions = self._model.predict_batch(self.observations[self._will_step])
        states = self.data.get_states(self.walkers_id[self._will_step])
        new_state, observs, rewards, ends, infos = self._env.step_batch(actions, states=states)

        # Save data and update sample count
        steps_done = self._will_step.sum().astype(np.int32)
        new_ids = self._n_samples_done + np.arange(steps_done).astype(int)
        self.walkers_id[self._will_step] = new_ids
        self.data.append(walker_ids=new_ids, states=new_state, actions=actions)
        self.observations[self._will_step] = np.array(observs).astype(np.float32)
        if self.accumulate_rewards:
            self.rewards[self._will_step] = self.rewards[self._will_step] + np.array(rewards)
        else:
            self.rewards[self._will_step] = np.array(rewards)
        self._end_cond[self._will_step] = ends
        self.infos[self._will_step] = np.array(infos).astype(np.float32)
        self.times[self._will_step] = self.times[self._will_step].astype(np.int32) + 1
        self._n_samples_done += steps_done

    def evaluate_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the relative distances between 0 and 1.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even though it is not a proper distance.
        """
        # Get random companion
        idx = np.random.permutation(np.arange(self.n_walkers, dtype=int))
        # Euclidean distance between states (pixels / RAM)
        obs = self.observations.astype(np.float32).reshape((self.n_walkers, -1))
        dist = np.sqrt(np.sum((obs[idx] - obs) ** 2, axis=tuple(range(1, len(obs.shape)))))
        return normalize_vector(dist)

    def virtual_reward(self) -> np.ndarray:
        """Calculate the virtual reward of the walkers. This quantity is used for determining
        the chance a given walker has of cloning. This scalar gives us a measure of how well a
        walker is solving the exploration vs exploitation problem, with respect to the other
        walkers of the Swarm.
        """
        dist = self.evaluate_distance()  # goes between 0 and 1
        rewards = np.array(self.rewards).astype(np.float32)
        scores = normalize_vector(rewards) + 1  # Goes between 1 and 2
        # The balance sets how much preference we are giving exploitation over exploration
        vir_reward = dist * scores ** self.balance
        return vir_reward

    def clone_condition(self):
        """Calculates the walkers that will cone depending on their virtual rewards. Returns the
        index of the random companion chosen for comparing virtual rewards.
        """
        self._pre_clone_ids = set(self.walkers_id.astype(int))
        # Calculate virtual rewards and choose another walker at random
        vir_rew = self.virtual_reward()
        alive_walkers = np.arange(self.n_walkers, dtype=int)[np.logical_not(self._end_cond)]
        alive_walkers = alive_walkers if len(alive_walkers) > 0 else np.arange(self.n_walkers)
        self._clone_idx = idx = np.random.choice(alive_walkers, self.n_walkers)

        # The probability of cloning depends on the relationship of virtual rewards
        # with respect to a randomly chosen walker.
        value = (vir_rew[idx] - vir_rew) / np.where(vir_rew > 0, vir_rew, 1e-8)
        clone = (value >= np.random.random()).astype(bool)
        # Boundary conditions(_end_cond) modify the cloning probability.
        clone[self._end_cond] = True
        #clone[self._end_cond[idx]] = False  # Do not clone to a dead companion

        # Walkers reaching the score limit do freeze (do not clone nor step)
        self._not_frozen = (self.rewards < self.reward_limit)
        self._will_clone = np.logical_and(clone, self._not_frozen)
        self._will_step = np.logical_and(np.logical_not(self._will_clone), self._not_frozen)
        if self._will_step.sum() == 0:
            raise ValueError("Everyone is cloning")


    def perform_clone(self):
        idx = self._clone_idx
        # This is a hack to make it work on n dimensional arrays
        obs_ix = self._will_clone[(...,) + tuple(np.newaxis for _ in
                                                 range(len(self.observations.shape) - 1))]
        self.observations = np.where(obs_ix,
                                     self.observations[idx], self.observations)
        # Using np.where seems to be faster than using a for loop
        self.rewards = np.where(self._will_clone, self.rewards[idx], self.rewards)
        self._end_cond = np.where(self._end_cond, self._end_cond[idx], self._end_cond)
        self.infos = np.where(self._will_clone, self.infos[idx], self.infos)
        self.times = np.where(self._will_clone, self.times[idx], self.times)
        self.walkers_id = np.where(self._will_clone, self.walkers_id[idx],
                                   self.walkers_id).astype(int)

    def clone(self):
        """The clone operator aims to change the distribution of walkers in the state space, by
         cloning some walkers to a randomly chosen companion. After cloning, the distribution of
         walkers will be closer to the reward distribution of the state space.
        1 - Choose a random companion who is alive.
        2 - Calculate the probability of cloning based on their virtual reward relationship.
        3 - Clone if p > random[0,1] or the walker is dead.
        """
        self._will_clone[self._end_cond] = True
        self._will_clone[self._end_cond[self._clone_idx]] = False
        self.perform_clone()
        self._post_clone_ids = set(self.walkers_id.astype(int))
        self._remove_id = self._pre_clone_ids - self._post_clone_ids
        self.data.delete(self._post_clone_ids)

    def stop_condition(self) -> bool:
        """This sets a hard limit on maximum samples. It also Finishes if all the walkers are dead,
         or the target score reached.
         """
        stop_hard = False if self.samples_limit is None else \
            self._n_samples_done > self.samples_limit
        stop_score = False if self.reward_limit is None else \
            self.rewards.max() >= self.reward_limit
        stop_terminal = self._end_cond.all()
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

    def run_swarm(self, state: np.ndarray=None, obs: np.ndarray=None, print_swarm: bool=False):
        """
        Iterate the swarm by either evolving or cloning each walker until a certain condition
        is met.
        :return:
        """
        self.init_swarm(state=state, obs=obs)
        while not self.stop_condition():
            try:
                if self._i_simulation > 1:
                    self.clone_condition()
                self.step_walkers()
                if self._i_simulation > 1:
                    self.clone()
                self._i_simulation += 1
                if self._i_simulation % self.render_every == 0 and print_swarm:
                    print(self)
                    clear_output(True)
            except KeyboardInterrupt:
                break
        if print_swarm:
            print(self)


