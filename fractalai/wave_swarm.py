import copy
import numpy as np
import networkx as nx
import gym


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


class DynamicTree:
    """This is a tree data structure that stores the paths followed by the walkers. It can be
    pruned to delete paths that will no longer be needed. It uses a networkx Graph. If someone
     wants to spend time building a proper data structure, please make a PR. I will be super happy!
    """
    def __init__(self):
        self.data = nx.DiGraph()
        self.data.add_node(0)
        self._recently_added = []
        self.root_id = 0

    def append_leaf(self, leaf_id: int, parent_id: int, obs, action=None, reward=None, end=None):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param obs: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param reward: reward assigned to the state represented by leaf_id.
        :param end: boolean indicating if the state is terminal.
        :return:
        """
        self.data.add_node(leaf_id, obs=obs, action=action, reward=reward, end=end)
        self.data.add_edge(parent_id, leaf_id)
        self._recently_added.append(leaf_id)

    def prune_branch(self, leaf, alive_leafs):
        """This recursively prunes a branch that only leads to an orphan leaf."""
        parent_id = list(self.data.in_edges(leaf))
        if len(parent_id) == 0:
            raise ValueError
        for parent, _ in parent_id:
            if parent == 0 or leaf == 0:
                return
            if len(self.data.out_edges(leaf)) == 0:
                self.data.remove_edge(parent, leaf)
                self.data.remove_node(leaf)
                if len(self.data.out_edges(parent)) == 0 and parent not in alive_leafs:
                    return self.prune_branch(parent, alive_leafs)
        return

    def prune_tree(self, dead_leafs, alive_leafs):
        """This prunes the orphan leaves that will no longer be used to save memory."""
        for leaf in list(self._recently_added):
            if leaf in dead_leafs:
                self.prune_branch(leaf, alive_leafs)
        self._recently_added = []
        return

    def get_branch(self, leaf_id) -> list:
        """
        Get the observation from the game ended at leaf_id
        :param leaf_id: id of the leaf nodei belonging to the branch that will be recovered.
        :return: Sequence of observations belonging to a given branch of the tree.
        """
        return [self.data.node[n]["obs"] for n in nx.shortest_path(self.data, 0, leaf_id)[1:]]


class WaveSwarm:
    """This is a very silly example on how using FAI we can derive a new tool for solving a
     specific problem. Our objective will be sampling the game with the highest possible score,
     given a bound in computational resources. We are assuming a deterministic environment that we
     can model perfectly.

     We will use a swarm that will explore the state space saving the paths that maximize utility.
     The swarm will sample the environment a limited amount of times, and then it will return the
     trajectories found with the highest utility.

     This algorithm is designed for generating efficiently high quality samples for supervised
     training. This is pretty much the same idea of FMC, but with no feedback loop to update the
      parameters. Instead of approximating an small tree each step, we will construct a huge tree
      that has a time horizon as further away as possible.
     """
    def __init__(self,
                 env_name: str = "MsPacman-v0",
                 n_samples: int = 1500,
                 n_walkers: int=50,
                 n_fixed_steps: int = 1,
                 balance: int=1,
                 skip_frames: int=0,
                 render_every: int=0,
                 until_score: int=None):
        """
        :param env_name: The name of the Atari game to be sampled.
        :param n_samples: Maximum number of samples allowed.
        :param n_walkers: Number of walkers that will use for exploring the space.
        :param n_fixed_steps: The number of times that we will apply the same action.
        :param balance: Coefficient that balances exploration vs exploitation.
        :param render_every: Number of iterations to be performed before updating displayed
        :param until_score: Maximum score that can be reached before stopping the sampling.
         statistics.
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.root_state = self.env.unwrapped.clone_full_state().copy()
        self.balance = balance
        self.render_every = render_every
        self.until_score = until_score

        self.n_actions = self.env.action_space.n
        self.n_fixed_steps = n_fixed_steps
        self.n_walkers = n_walkers
        self.skip_frames = skip_frames
        self.balance = 1
        self._i_simulation = 0
        self._n_samples_done = 0

        self._n_limit_samples = n_samples

        self._death_cond = np.zeros(self.n_walkers, dtype=np.bool)
        self._will_clone = np.zeros(self.n_walkers, dtype=np.bool)
        self._dead_mask = np.zeros(self.n_walkers, dtype=np.bool)
        self._will_step = np.zeros(self.n_walkers, dtype=np.bool)
        self._terminal = np.zeros(self.n_walkers, dtype=np.bool)
        self.times = np.zeros(self.n_walkers)
        self._old_lives = - np.ones(self.n_walkers)
        self.obs = None
        self.rewards = np.ones(self.n_walkers) * -1e5
        # This is an overkill, but its fast anyway
        self.random_actions = np.random.randint(0, self.n_actions,
                                                size=(int(1e6), n_walkers),
                                                dtype=np.uint8)
        self.walkers_id = np.zeros(self.n_walkers)
        self.walkers = None
        self._i_epoch = 0
        self.tree = None
        self._init_swarm()

    def __str__(self):
        efi = (len(self.tree.data.nodes) / self._n_samples_done) * 100
        sam_step = self._n_samples_done / len(self.tree.data.nodes)
        progress = (self._n_samples_done / self._n_limit_samples) * 100
        if self.until_score is not None:
            score_prog = (self.rewards.max() / self.until_score) * 100
            progress = max(progress, score_prog)

        text = "Environment: {}\n" \
               "Samples done: {} Progress {:.2f}%\n" \
               "Reward: mean {:.2f} std {:.2f} max {:.2f} min {:.2f}\n" \
               "Episode length: mean {:.2f} std {:.2f} max {:.2f} min {:.2f}\n" \
               "Efficiency {:.2f}%\n" \
               "Generated {} Examples |" \
               " {:.2f} samples per example.".format(self.env_name, self._n_samples_done,
                                                     progress,
                                                     self.rewards.mean(), self.rewards.std(),
                                                     self.rewards.max(), self.rewards.min(),
                                                     self.times.mean(), self.times.std(),
                                                     self.times.max(), self.times.min(),
                                                     efi, len(self.tree.data.nodes),
                                                     sam_step)
        return text

    def _init_swarm(self):
        """Sync all the swarm so they become a copy of the root state"""
        self._n_samples_done = 0
        self._i_epoch = 0
        obs = self.env.reset()
        reward, lives = 0, 0
        # skip some frames
        for i in range(self.skip_frames):
            obs, _reward, end, info = self.env.step(0)
            reward += reward
            lives = info["ale.lives"]
            if end:
                obs = self.env.reset()
                reward, lives = 0, 0
                break
        # Initialize internal parameters
        self.root_state = self.env.unwrapped.clone_full_state()
        self.walkers = np.array([self.root_state.copy() for _ in range(self.n_walkers)])
        self.walkers_id = np.zeros(self.n_walkers)
        self._will_clone = np.zeros(self.n_walkers, dtype=np.bool)
        self._dead_mask = np.zeros(self.n_walkers, dtype=np.bool)
        self._state_times = np.ones(self.n_walkers)
        self.times = np.zeros(self.n_walkers)
        self._old_lives = np.ones(self.n_walkers) * lives
        self.obs = np.array([obs.copy() for _ in range(self.n_walkers)])
        self.rewards = np.ones(self.n_walkers) * reward
        self.tree = DynamicTree()

    def _step_walkers(self, init_step: bool = False):
        """Sample an action for each walker, and act on the system.
        You can apply the same action multiple times to save some computation.
        :param init_step: If it is the first step, keep track of the initial actions taken.
        :return: None.
        """
        # Only step an state if it has not cloned
        step_ix = np.logical_not(self._will_clone) if not init_step else np.ones(self.n_walkers,
                                                                                 dtype=bool)
        for i, state in enumerate(self.walkers):
            if step_ix[i]:
                self.env.unwrapped.restore_full_state(state)
                old_lives = self._old_lives[i]
                end = False
                action = self.random_actions[self._i_epoch % self.random_actions.shape[0], i]
                # We can choose to apply the same action several times
                for _ in range(self.n_fixed_steps):
                    obs, _reward, _end, info = self.env.step(action)
                    self._n_samples_done += 1
                    self.rewards[i] += _reward
                    # Check boundary conditions
                    end = end or _end
                    self._death_cond[i] = end or info["ale.lives"] < old_lives
                    old_lives = float(info["ale.lives"])
                    # Update data
                    self.times[i] += 1
                    self.obs[i] = obs.copy()
                    self._old_lives[i] = old_lives
                    self._n_samples_done += 1
                    # Keep track of the paths using the tree
                    new_state = self.env.unwrapped.clone_full_state().copy()
                    new_id = self._n_samples_done
                    old_id = self.walkers_id[i]
                    self.tree.append_leaf(new_id, parent_id=old_id, obs=obs)
                    self.walkers_id[i] = new_id
                    self.walkers[i] = new_state
                    # Reset if necessary
                    if _end:
                        self._terminal[i] = True
                        self.env.reset()
                        break
        self._will_step = step_ix

    def _evaluate_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different images
        on a vector of swarm, and normalizes the relative distances between 1 and 2.
        In a more general scenario, any function that quantifies the notion of "how different two
        swarm are" could work, even though it is not a proper distance.
        """
        # Get random companion
        idx = np.random.permutation(np.arange(self.n_walkers, dtype=int))
        obs = np.array(self.obs)
        # Euclidean distance between pixels
        dist = np.sqrt(np.sum((obs[idx] - obs) ** 2, axis=tuple(range(1, len(obs.shape)))))
        # We want time diversity to detect deaths early and have some extra reaction time
        time_div = np.linalg.norm(self.times.reshape((-1, 1)) - self.times[idx].reshape((-1, 1)),
                                  axis=0)
        # This is a distance formula that I just invented that expands the swarm in time
        return normalize_vector(normalize_vector(dist) + normalize_vector(time_div))

    def _virtual_reward(self) -> np.ndarray:
        """Calculate the entropic reward of the walkers. This quantity is used for determining
        the chance a given state has of cloning. This scalar gives us a measure of how well an
        state is solving the exploration vs exploitation problem.
        """
        dist = self._evaluate_distance()  # goes between 0 and 1
        rewards = np.array(self.rewards)
        scores = normalize_vector(rewards) + 1  # Goes between 1 and 2
        # The balance sets how much preference we are giving exploitation over exploration.
        ent_reward = dist * scores ** self.balance
        return ent_reward

    def _clone(self):
        """The clone phase aims to change the distribution of walkers in the state space, by
         cloning some swarm to a randomly chosen companion. After cloning, the distribution of
         walkers will be closer to the reward distribution of the state space.
        1 - Choose a random companion who is alive.
        2 - Calculate the probability of cloning.
        3 - Clone if True or the walker is dead.
        """
        ent_rew = self._virtual_reward()

        # Besides the death condition, zero entropic reward also counts as death when cloning
        deaths = np.array([dead or ent_rew[i] == 0 for i, dead in enumerate(self._death_cond)])
        self._dead_mask = deaths

        idx = np.random.permutation(np.arange(self.n_walkers, dtype=int))

        # The probability of cloning depends on the relationship of entropic rewards
        # with respect to a randomly chosen walker.
        value = (ent_rew[idx] - ent_rew) / np.where(ent_rew > 0, ent_rew, 1e-8)
        clone = (value >= np.random.random()).astype(bool)

        clone[deaths] = True
        clone[deaths[idx]] = False  # Do not clone to a dead companion
        # Actual cloning
        self._will_clone = clone
        old_ids = set(self.walkers_id)
        for i, clo in enumerate(clone):
            if clo:
                self.walkers[i] = self.walkers[idx][i].copy()
                self.obs[i] = self.obs[idx][i].copy()
                self.rewards[i] = float(self.rewards[idx][i])
                self._old_lives[i] = float(self._old_lives[idx][i])
                self.times[i] = float(self.times[idx][i])
                self.walkers_id[i] = copy.deepcopy(self.walkers_id[idx][i])
        # Prune tree to save memory
        dead_leafs = old_ids - set(self.walkers_id)
        self.tree.prune_tree(dead_leafs, set(self.walkers_id))

    def _stop_condition(self) -> bool:
        """This sets a hard limit on maximum samples. It also Finishes if all the walkers are dead,
         or the target score reached.
         """
        stop_hard = self._n_samples_done > self._n_limit_samples
        score = False if self.until_score is None else self.rewards.max() > self.until_score
        return stop_hard or score or self._terminal.all()

    def run_swarm(self):
        """
        Probability distribution over each action for the current state.
        This is done by propagating and cloning the walkers to build a causal cone.
        The cone is n_states wide and depth long.
        """
        self._init_swarm()
        while not self._stop_condition():
            if self._i_simulation > 1:
                self._clone()
            self._step_walkers()
            self._i_simulation += 1
            if self._i_simulation % self.render_every == 0:
                from IPython.core.display import clear_output
                print(self)
                clear_output(True)

    def recover_game(self, index=None) -> list:
        """
        By default, returns the game sampled with the highest score.
        :param index: id of the leaf where the returned game will finish.
        :return: a list containing the observations of the target sampled game.
        """
        if index is None:
            index = self.walkers_id[self.rewards.argmax()]
        return self.tree.get_branch(index)
