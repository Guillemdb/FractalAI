from typing import Iterable, Callable
import copy
import numpy as np
import networkx as nx
from IPython.core.display import clear_output


def normalize_vector(vector: np.ndarray):
    avg = vector.mean()
    if avg == 0:
        return np.ones(len(vector))
    standard = vector / avg
    return standard


def relativize_vector(vector: np.ndarray):
    std = vector.std()
    if std == 0:
        return np.ones(len(vector))
    standard = (vector - vector.mean()) / std
    standard[standard > 0] = np.log(1 + standard[standard > 0]) + 1
    standard[standard <= 0] = np.exp(standard[standard <= 0])
    return standard


def normalize_vector_zero_one(vector):
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
    """This is a class for storing the states and the observations of a Swarm.
    This way is slower than storing it in a numpy array, but it allows to store
    any kind of states and observations."""

    def __init__(self):
        self.states = {}
        self.actions = {}
        self.infos = {}
        self.walker_ids = []

    def __getitem__(self, item):
        states = self.get_states(item)
        actions = self.get_actions(item)
        return states, actions

    def reset(self):
        self.states = {}
        self.actions = {}
        self.infos = {}
        self.walker_ids = []

    def get_states(self, labels: Iterable) -> list:
        return [self.states[label] for label in labels]

    def get_actions(self, labels: Iterable) -> list:
        return [self.actions[label] for label in labels]

    def get_infos(self, labels: Iterable) -> list:
        return [copy.copy(self.infos[label]) for label in labels]

    def append(self, walker_ids: [list, np.ndarray], states: Iterable, actions, infos):
        actions = actions if actions is not None else [None] * len(walker_ids)
        infos = infos if infos is not None else [None] * len(walker_ids)
        for w_id, state, action, info in zip(walker_ids, states, actions, infos):
            if w_id not in self.walker_ids:
                self.states[w_id] = copy.deepcopy(state)
                if actions is not None:
                    self.actions[w_id] = copy.deepcopy(action)
                if infos is not None:
                    self.infos[w_id] = copy.deepcopy(info)
        self.walker_ids = list(set(self.walker_ids))
        self.walker_ids += list(set(walker_ids))

    def update_values(self, walker_ids):
        # This is not optimal, but ensures no memory leak
        new_states = {}
        new_infos = {}
        new_actions = {}
        for w_id in walker_ids:
            new_states[w_id] = self.states[w_id]
            new_actions[w_id] = self.actions[w_id]
            new_infos[w_id] = self.infos.get(w_id)
        self.states = new_states
        self.actions = new_actions
        self.infos = new_infos
        self.walker_ids = walker_ids


class DynamicTree:
    """This is a tree data structure that stores the paths followed by the walkers. It can be
    pruned to delete paths that will no longer be needed. It uses a networkx Graph. If someone
    wants to spend time building a proper data structure, please make a PR, and I will be super
    happy!
    """

    def __init__(self):
        self.data = nx.DiGraph()
        self.data.add_node("0")
        self.root_id = "0"

    def reset(self):
        self.data.remove_edges_from(list(self.data.edges))
        self.data.remove_nodes_from(list(self.data.nodes))
        self.data.add_node("0")
        self.root_id = "0"

    def append_leaf(self, leaf_id: [int, float], parent_id: [int, float], state, action, dt: int):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param dt: parameters taken into account when integrating the action.
        :return:
        """
        self.data.add_node(str(leaf_id), state=state)
        self.data.add_edge(str(parent_id), str(leaf_id), action=action, dt=dt)

    def prune_branch(self, leaf, alive_leafs):
        """This recursively prunes a branch that only leads to an orphan leaf."""
        leaf = str(leaf)
        alive_leafs = [str(le) for le in alive_leafs]
        parent_id = self.data.in_edges([leaf])
        parent_ids = list(parent_id)
        for parent, _ in parent_ids:
            parent = str(parent)
            if parent in [0, "0"] or leaf in [0, "0"]:
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
        nodes = nx.shortest_path(self.data, "0", str(leaf_id))
        states = [self.data.node[n]["state"] for n in nodes]
        actions = [self.data.edges[(n, nodes[i + 1])]["action"] for i, n in enumerate(nodes[:-1])]
        dts = [self.data.edges[(n, nodes[i + 1])]["dt"] for i, n in enumerate(nodes[:-1])]
        return states, actions, dts

    def get_parent(self, node_id):
        return list(self.data.in_edges(str(node_id)))[0][0]

    def get_leaf_nodes(self):
        leafs = []
        for node in self.data.nodes:
            if len(self.data.out_edges([str(node)])) == 0:
                leafs.append(node)
        return leafs


class Swarm:

    """This is the most basic mathematical entity that can be derived from Fractal AI theory.
    It represents a cloud of points that propagates through an state space. Each walker of the
    swarm evolves by either cloning to another walker or perturbing the environment.
    """

    def __init__(
        self,
        env,
        model,
        n_walkers: int = 100,
        balance: float = 1.0,
        reward_limit: float = None,
        samples_limit: int = None,
        render_every: int = 1e10,
        accumulate_rewards: bool = True,
        dt_mean: float = None,
        dt_std: float = None,
        min_dt: int = 1,
        custom_reward: Callable = None,
        custom_end: Callable = None,
        process_obs: Callable = None,
        custom_skipframe: Callable = None,
        keep_best: bool = False,
        can_win: bool = False,
        dist_coef: float = 1,
        entropy_coef: float = 1.0,
    ):
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
        :param accumulate_rewards: Use the accumulated reward when scoring the walkers.
                                  False to use instantaneous reward.
        :param dt_mean: Mean skipframe used for exploring.
        :param dt_std: Standard deviation for the skipframe. Sampled from a normal distribution.
        :param min_dt: Minimum skipframe to be used by the swarm.
        :param custom_reward: Callable for calculating a custom reward function.
        :param custom_end: Callable for calculating custom boundary conditions.
        :param process_obs: Callable for doing custom observation processing.
        :param custom_skipframe: Callable for sampling the skipframe values of the walkers.
        :param keep_best: Keep track of the best accumulated reward found so far.
        :param can_win: If the game can be won when a given score is achieved, set to True. Meant
        to be used with Atari games like Boxing, Pong, IceHockey, etc.
        """

        def default_end(self, infos, old_infos, rewards, **kwargs):
            """Default death function: loose a life or get a
            negative rewards in games you can win."""
            return np.array(
                [
                    int(n.get("lives", -1)) < int(o.get("lives", -1)) or (can_win and r < 0)
                    for n, o, r in zip(infos, old_infos, rewards)
                ]
            ).astype(bool)

        def default_reward(rewards, *args, **kwargs):
            return rewards

        def default_proc_obs(observs):
            return np.array(observs).astype(np.float32)

        # Parameters of the algorithm
        self.samples_limit = samples_limit if samples_limit is not None else np.inf
        self.reward_limit = reward_limit if reward_limit is not None else np.inf
        self._process_obs = process_obs if process_obs is not None else default_proc_obs
        self._model = model
        self._env = env
        self.n_walkers = n_walkers
        self.balance = balance
        self.dist_coef = dist_coef
        self.entropy_coef = entropy_coef
        self.render_every = render_every
        self.accumulate_rewards = accumulate_rewards
        self.dt_mean = dt_mean
        self.dt_std = dt_std
        self.min_dt = min_dt
        self.custom_end = custom_end if custom_end is not None else default_end
        self.custom_reward = custom_reward if custom_reward is not None else default_reward
        self.custom_skipframe = (
            custom_skipframe if custom_skipframe is not None else self._calculate_dt
        )
        self.keep_best = keep_best
        # Environment information sources
        self.observations = None
        self.rewards = np.zeros(self.n_walkers)
        # Internal masks
        # True when the boundary condition is met
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        # Walkers that will clone to a random companion
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        # If true the corresponding walker will not move
        self._not_frozen = np.ones(self.n_walkers, dtype=bool)
        # Processed information sources
        self._clone_idx = None
        self.virtual_rewards = np.zeros(self.n_walkers)
        self.distances = np.zeros(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self.walkers_life = np.ones(self.n_walkers)
        self.dt = np.ones(self.n_walkers, dtype=int)
        self._n_samples_done = 0
        self._i_simulation = 0
        self._game_status = ""
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        self._virtual_reward = None
        # This is for storing states and actions of arbitrary shape and type
        self.data = DataStorage()
        self._pre_clone_ids = [0]
        self._post_clone_ids = [0]
        self._remove_id = [None]
        self._old_rewards = self.rewards
        self.win_flag = False
        self._game_can_be_won = can_win
        self.ends = np.zeros(self.n_walkers, dtype=bool)
        self.terms = np.zeros(self.n_walkers, dtype=bool)

    def __str__(self):
        """Print information about the internal state of the swarm."""

        progress = (
            0 if self.samples_limit is None else (self._n_samples_done / self.samples_limit) * 100
        )

        if self.reward_limit is not None:
            score_prog = (self.rewards.max() / self.reward_limit) * 100
            progress = max(progress, score_prog)

        text = (
            "Environment: {} | Walkers: {} | Deaths: {} | clones {}\n"
            "Total samples: {} Progress: {:.2f}%\n"
            "Reward: mean {:.2f} | Dispersion: {:.2f} | max {:.2f} | min {:.2f} | std {:.2f}\n"
            "Episode length: mean {:.2f} | Dispersion {:.2f} | max {:.2f} | min {:.2f} "
            "| std {:.2f}\n"
            "Dt: mean {:.2f} | Dispersion: {:.2f} | max {:.2f} | min {:.2f} | std {:.2f}\n"
            "Status: {}".format(
                self._env.name,
                self.n_walkers,
                self._end_cond.sum(),
                self._will_clone.sum(),
                self._n_samples_done,
                progress,
                self.rewards.mean(),
                self.rewards.max() - self.rewards.min(),
                self.rewards.max(),
                self.rewards.min(),
                self.rewards.std(),
                self.times.mean(),
                self.times.max() - self.times.min(),
                self.times.max(),
                self.times.min(),
                self.times.std(),
                self.dt.mean(),
                self.dt.max() - self.dt.min(),
                self.dt.max(),
                self.dt.min(),
                self.dt.std(),
                self._game_status,
            )
        )
        return text

    @property
    def env(self):
        return self._env

    @property
    def actions(self):
        return self.data.get_actions(self.walkers_id)

    @property
    def states(self):
        return self.data.get_states(self.walkers_id)

    def seed(self, seed):
        self.env._env.seed(seed)
        np.random.seed(seed)

    def reset(self):
        """Reset the internal data of the swarm, and restores the same values as when
         the Swarm was instantiated."""
        self.observations = None
        self.rewards = np.zeros(self.n_walkers)
        # Internal masks
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        self._not_frozen = np.ones(self.n_walkers, dtype=bool)
        # Processed information sources
        self._clone_idx = None
        self.virtual_rewards = np.zeros(self.n_walkers)
        self.distances = np.zeros(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self.dt = np.ones(self.n_walkers, dtype=int)
        self._n_samples_done = 0
        self._i_simulation = 0
        self._game_status = ""
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        self._virtual_reward = None
        # This is for storing states and actions of arbitrary shape and type
        self.data = DataStorage()
        self._pre_clone_ids = [0]
        self._post_clone_ids = [0]
        self._remove_id = [None]
        self._old_rewards = self.rewards
        self.win_flag = False
        self.ends = np.zeros(self.n_walkers, dtype=bool)
        self.walkers_life = np.ones(self.n_walkers)

    def init_swarm(self, state: np.ndarray = None, obs: np.ndarray = None):
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
        self.observations = self.process_observations(
            np.array([obs.copy() for _ in range(self.n_walkers)])
        )
        self.rewards = np.zeros(self.n_walkers, dtype=np.float32)
        self._end_cond = np.zeros(self.n_walkers, dtype=bool)
        # Internal masks
        self._will_clone = np.zeros(self.n_walkers, dtype=bool)
        # Processed information sources
        self.virtual_rewards = np.zeros(self.n_walkers)
        self.distances = np.zeros(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self._n_samples_done = 0
        self._i_simulation = 0
        self._virtual_reward = None
        # Store data and keep indices
        self.data.reset()
        self.walkers_id = np.zeros(self.n_walkers).astype(int)
        actions = self._model.predict_batch(self.observations)
        states = np.array([copy.deepcopy(state) for _ in range(self.n_walkers)])
        self.data.append(
            walker_ids=self.walkers_id,
            states=states,
            actions=actions,
            infos=[{"terminal": False}] * self.n_walkers,
        )
        self._old_rewards = self.rewards
        self._not_frozen = np.ones(self.n_walkers, dtype=bool)
        self.win_flag = False
        self.ends = np.zeros(self.n_walkers, dtype=bool)
        self.terms = np.zeros(self.n_walkers, dtype=bool)
        self.walkers_life = np.ones(self.n_walkers)

    def _calculate_dt(self, *args, **kwargs):
        """Sample the skipframe from a bounded normal distribution."""
        size = self.n_walkers
        if self.dt_mean is not None and self.dt_std is not None:
            abs_rnd = np.abs(np.random.normal(loc=self.dt_mean, scale=self.dt_std, size=size))
            skipframes = np.maximum(self.min_dt, abs_rnd).astype(int)
        else:
            skipframes = np.ones(size, dtype=int) * self.min_dt
        return skipframes

    def process_observations(self, observs: np.ndarray = None):
        return self._process_obs(observs)

    def step_walkers(self):
        """Sample an action for each walker, and act on the environment. This is how the Swarm
        evolves.
        :return: None.
        """
        # Only step an state if it has not cloned and is not frozen
        # if self.keep_best:
        # self._will_step[-1] = False
        self.dt = self.custom_skipframe(self)
        actions = self._model.predict_batch(self.observations[self._not_frozen])

        states = self.data.get_states(self.walkers_id[self._not_frozen])
        old_infos = self.data.get_infos(self.walkers_id[self._not_frozen])
        new_state, observs, _rewards, terms, infos = self._env.step_batch(
            actions, states=states, n_repeat_action=self.dt[self._not_frozen]
        )
        self.times[self._not_frozen] = (
            self.times[self._not_frozen] + self.dt[self._not_frozen]
        ).astype(np.int32)
        # Calculate custom rewards and boundary conditions
        rewards = self.custom_reward(
            infos=infos,
            old_infos=old_infos,
            rewards=_rewards,
            times=self.times[self._not_frozen],
            old_rewards=self.rewards[self._not_frozen],
        )
        self.ends = self.custom_end(
            self=self,
            infos=infos,
            old_infos=old_infos,
            rewards=_rewards,
            times=self.times[self._not_frozen],
            terminals=terms,
            old_rewards=self.rewards[self._not_frozen],
        )
        # Save data and update sample count
        new_ids = self._n_samples_done + np.arange(self._not_frozen.sum()).astype(int)
        self.walkers_id[self._not_frozen] = new_ids
        self.data.append(walker_ids=new_ids, states=new_state, actions=actions, infos=infos)
        try:
            # self.observations = np.array(self.observations)
            self.observations[self._not_frozen] = self.process_observations(observs).tolist()
        except ValueError as err:
            print(
                len(observs),
                observs[0].shape,
                len(self.process_observations(observs)),
                self.observations[self._not_frozen].shape,
                np.array(observs).shape,
            )
            raise err
        non_neg_reward = np.array(rewards) >= 0
        terms = np.array([inf["terminal"] for inf in infos])
        self.terms = terms
        # Check win condition
        if self._game_can_be_won:
            flag = np.logical_and(terms, non_neg_reward)
            self.win_flag = flag.any()
        # Accumulate if you are solving a trajectory,
        # If you are searching for a point set to False
        if self.accumulate_rewards:
            self.rewards[self._not_frozen] = self.rewards[self._not_frozen] + np.array(rewards)
        else:
            self.rewards[self._not_frozen] = np.array(rewards)
        self._end_cond[self._not_frozen] = np.logical_or(self.ends, terms)
        # Stop all the walkers
        if self._game_can_be_won:
            self._end_cond[self._not_frozen][flag] = False

        self._n_samples_done += self.dt[self._not_frozen].sum()

    def evaluate_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the result applying the relativize function.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even if it is not a proper distance.
        """
        # Get random companion
        idx = np.random.permutation(np.arange(self.n_walkers, dtype=int))
        # Euclidean distance between states (pixels / RAM)
        obs = self.observations.astype(np.float32).reshape((self.n_walkers, -1))
        dist = np.sqrt(np.sum((obs[idx] - obs) ** 2, axis=tuple(range(1, len(obs.shape)))))
        return relativize_vector(dist)

    def normalize_rewards(self) -> np.ndarray:
        """We also apply the relativize function to the rewards"""
        rewards = np.array(self.rewards)
        return relativize_vector(rewards).astype(np.float32)

    def virtual_reward(self) -> np.ndarray:
        """Calculate the virtual reward of the walkers. This quantity is used for determining
        the chance a given walker has of cloning. This scalar gives us a measure of how well a
        walker is solving the exploration vs exploitation problem, with respect to the other
        walkers of the Swarm.
        """
        dist = self.evaluate_distance()  # goes between 0 and 1
        scores = self.normalize_rewards()
        # The balance sets how much preference we are giving exploitation over exploration
        life_score = relativize_vector(self.walkers_life)
        vir_reward = (
            dist ** self.dist_coef * scores ** self.balance * life_score ** self.entropy_coef
        )
        self._virtual_reward = vir_reward
        return vir_reward

    def track_best_walker(self):
        """The last walker represents the best solution found so far. It gets frozen so
         other walkers can always compare to it when cloning."""
        # Last walker stores the best value found so far so other walkers can clone to it
        self._not_frozen[-1] = False
        self._will_clone[-1] = False
        # self._will_step[-1] = False
        best_walker = self.rewards.argmax()
        if best_walker != self.n_walkers - 1 and self.rewards[best_walker] > self.rewards[-1]:
            self.walkers_id[-1] = int(self.walkers_id[best_walker])
            self.observations[-1] = copy.deepcopy(self.observations[best_walker])
            self.rewards[-1] = float(self.rewards[best_walker])
            self.times[-1] = float(self.times[best_walker])
            self._end_cond[-1] = bool(self._end_cond[best_walker])

    def freeze_walkers(self):
        # Walkers reaching the score limit do freeze (do not clone nor step). Last walker is frozen
        self._not_frozen = self.rewards < self.reward_limit

    def get_clone_compas(self):
        alive_walkers = np.arange(self.n_walkers, dtype=int)[np.logical_not(self._end_cond)]
        if len(alive_walkers) > 0:
            self._clone_idx = np.random.choice(alive_walkers, self.n_walkers)
        else:
            self._clone_idx = None
            self._end_cond = np.ones(self.n_walkers, dtype=bool)
            return None
        return self._virtual_reward[self._clone_idx]

    def clone_condition(self):
        """Calculates the walkers that will cone depending on their virtual rewards. Returns the
        index of the random companion chosen for comparing virtual rewards.
        """
        self.freeze_walkers()
        if self.keep_best:
            self.track_best_walker()
        self._pre_clone_ids = list(set(self.walkers_id.astype(int)))
        # Calculate virtual rewards and choose another walker at random
        vir_rew = self.virtual_reward()
        vr_compas = self.get_clone_compas()
        if vr_compas is None:
            self._will_clone = np.zeros(self.n_walkers, dtype=bool)
            return
        # The probability of cloning depends on the relationship of virtual rewards
        # with respect to a randomly chosen walker.
        value = (vr_compas - vir_rew) / np.where(vir_rew > 0, vir_rew, 1e-8)
        clone = (value >= np.random.random()).astype(bool)
        self._will_clone = np.logical_and(clone, self._not_frozen)
        self._will_clone[self._end_cond] = True
        self._will_clone[np.logical_not(self._not_frozen)] = False
        self.walkers_life[self._will_clone] = 0
        self.walkers_life += 1

    def perform_clone(self):
        idx = self._clone_idx
        # A hack that avoid cloning
        if idx is None:
            return
        # This is a hack to make it work on n dimensional arrays
        obs_ix = self._will_clone[
            (...,) + tuple(np.newaxis for _ in range(len(self.observations.shape) - 1))
        ]
        self.observations = np.where(obs_ix, self.observations[idx], self.observations)
        # Using np.where seems to be faster than using a for loop
        self.rewards = np.where(self._will_clone, self.rewards[idx], self.rewards)
        self._end_cond = np.where(self._will_clone, self._end_cond[idx], self._end_cond)
        self.times = np.where(self._will_clone, self.times[idx], self.times)

        self.walkers_id = np.where(self._will_clone, self.walkers_id[idx], self.walkers_id).astype(
            int
        )

    def update_data(self):
        """Update the states and observations of the swarm kept in self.data."""
        self._post_clone_ids = set(self.walkers_id.astype(int))
        self.data.update_values(self._post_clone_ids)

    def clone(self):
        """The clone operator aims to change the distribution of walkers in the state space, by
         cloning some walkers to a randomly chosen companion. After cloning, the distribution of
         walkers will be closer to the reward distribution of the state space.
        1 - Choose a random companion who is alive.
        2 - Calculate the probability of cloning based on their virtual reward relationship.
        3 - Clone if p > random[0,1] or the walker is dead.
        """
        # Boundary conditions(_end_cond) modify the cloning probability.
        if self.keep_best:
            self._will_clone[-1] = False
        self.perform_clone()
        self.update_data()

    def stop_condition(self) -> bool:
        """This sets a hard limit on maximum samples. It also Finishes if all the walkers are dead,
         or the target score reached.
         """
        stop_hard = (
            False if self.samples_limit is None else self._n_samples_done > self.samples_limit
        )
        stop_score = (
            False if self.reward_limit is None else self.rewards.max() >= self.reward_limit
        )
        stop_terminal = self.terms.all() or np.logical_not(self._not_frozen[:-1]).any()
        # Define game status so the user knows why a game stopped. Only used when printing
        if stop_hard:
            self._game_status = "Sample limit reached."
        elif stop_score:
            self._game_status = "Score limit reached."
        elif stop_terminal:
            self._game_status = "All the walkers died."
        elif self.win_flag:
            self._game_status = "The game was won. Congratulations!"
        else:
            self._game_status = "Playing..."
        return stop_hard or stop_score or stop_terminal or self.win_flag

    def run_swarm(
        self, state: np.ndarray = None, obs: np.ndarray = None, print_swarm: bool = False
    ):
        """
        Iterate the swarm by either evolving or cloning each walker until a certain condition
        is met.
        :return:
        """
        self.init_swarm(state=state, obs=obs)
        while not self.stop_condition():
            # try:
            # We calculate the clone condition, and then perturb the walkers before cloning
            # This allows the deaths to recycle faster, and the Swarm becomes more flexible
            # If you choose to freeze some walkers.
            if self._i_simulation > 1:
                self.clone_condition()
                self.clone()
            self.step_walkers()
            self._i_simulation += 1
            if self._i_simulation % self.render_every == 0 and print_swarm:
                print(self)
                clear_output(True)
            # except ValueError:
            #    # TOOO: Check when this stuff can fail
            #    break
        if print_swarm:
            print(self)
