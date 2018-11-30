import numpy as np
import copy
import time
from IPython.core.display import clear_output
from fractalai.datasets.data_generator import Swarm, DLTree
from fractalai.swarm import relativize_vector
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

        new_text = (
            "{}\n"
            "Generated {} Examples |"
            " {:.2f} samples per example.\n".format(text, samples, sam_step)
        )
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
                self.tree.append_leaf(
                    idx,
                    parent_id=old_ids[i],
                    state=self.data.get_states([idx])[0],
                    action=self.data.get_actions([idx])[0],
                    dt=self.dt[i],
                    reward=float(self.rewards[i]),
                    terminal=bool(self.data.get_infos([idx])[0]["terminal"]),
                    obs=self.observations[i],
                )

    def recover_game(self, index=None) -> tuple:
        """
        By default, returns the game sampled with the highest score.
        :param index: id of the leaf where the returned game will finish.
        :return: a list containing the observations of the target sampled game.
        """
        if index is None:
            index = self.walkers_id[self.rewards.argmax()]
        return self.tree.get_branch(index)

    def render_game(self, index=None, sleep: float = 0.02):
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


class PesteWave(MLWave):
    def __init__(self, memory_lr: float = 1, peste_balance: float = 1, *args, **kwargs):
        super(PesteWave, self).__init__(*args, **kwargs)
        self.peste_walks = np.ones(self.rewards.shape)
        self.peste_balance = peste_balance
        self.memory_lr = memory_lr
        self.mem_nodes = []

    def __str__(self):
        text = super(PesteWave, self).__str__()
        vals = self.walkers_life * self.memory_lr
        return (
            "Entropy mean: {:.2f}  entropy std {:.2f}\n"
            "Memory nodes {}\n".format(vals.mean(), vals.std(), len(self.mem_nodes)) + text
        )

    def virtual_reward(self) -> np.ndarray:
        """Calculate the virtual reward of the walkers. This quantity is used for determining
                the chance a given walker has of cloning. This scalar gives us a measure of how
                well a
                walker is solving the exploration vs exploitation problem, with respect to the
                other
                walkers of the Swarm.
                """
        vir_reward = super(PesteWave, self).virtual_reward()
        self.peste_walks = self.peste_distance()
        peste_reward = vir_reward * self.peste_walks ** self.peste_balance
        return peste_reward

    def peste_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the result applying the relativize function.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even if it is not a proper distance.
        """
        # Get random companion
        # return np.ones(self.n_walkers)
        peste_obs = self.get_peste_obs()
        # Euclidean distance between states (pixels / RAM)
        obs = self.observations.astype(np.float32).reshape((self.n_walkers, -1))
        peste_obs = peste_obs.reshape((self.n_walkers, -1))
        dist = np.sqrt(np.sum((peste_obs - obs) ** 2, axis=tuple(range(1, len(obs.shape)))))
        return relativize_vector(dist)

    def ___evaluate_distance(self):
        return relativize_vector(self.times)  # np.ones(self.n_walkers)

    def ___get_peste_obs(self):
        peste_compas = self.get_clone_compas()
        if peste_compas is None:
            return self.observations
        observs = []
        for walker, ix in enumerate(peste_compas.astype(int)):
            time_val = (self.times[walker] / self.dt[walker]) / 4
            val = np.random.normal(loc=time_val, scale=time_val)  # * self.memory_lr
            if val <= 0:
                obs = self.tree.get_past_obs(self.walkers_id[walker], int(np.abs(val)))
                # obs = self.observations[ix]
            else:
                obs = self.tree.get_past_obs(self.walkers_id[walker], int(val))
            observs.append(obs)
        return np.array(observs)

    def get_peste_obs(self):
        memory_nodes = list(self.tree.data.nodes)
        ix = max(200, int(len(memory_nodes) * 0.35), self.n_walkers)
        ix_fin = max(50, int(len(memory_nodes) * 0.10))
        mem_batch = memory_nodes  # [-ix:-self.n_walkers]
        self.mem_nodes = mem_batch
        if len(mem_batch) < self.n_walkers / 10:
            return self.observations[np.random.permutation(np.arange(self.n_walkers))]
        mem_batch = mem_batch if len(mem_batch) > self.n_walkers / 10 else memory_nodes
        nodes = np.random.choice(mem_batch, self.n_walkers)

        observs = []
        for node in nodes:
            observs.append(self.tree.data.node[node]["obs"].copy())
        return np.array(observs)


class MLFMC(FractalMC):
    def __init__(self, data_env, true_min_dt, *args, **kwargs):

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

    def render_game(self, index=None, sleep: float = 0.02):
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

    def collect_data(self, render: bool = False, print_swarm: bool = False):
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

            state, obs, _reward, _end, info = self.data_env.step(
                state=state, action=action, n_repeat_action=1
            )
            self.tree.append_leaf(
                i_step,
                parent_id=i_step - 1,
                state=state,
                action=np.ones(self.env.n_actions),
                dt=1,
                reward=np.ones(self.env.n_actions),
                terminal=bool(info["terminal"]),
                obs=obs,
            )

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
            state, obs, _reward, _end, info = self.data_env.step(
                state=state, action=action, n_repeat_action=1
            )
            self.tree.append_leaf(
                i_step,
                parent_id=i_step - 1,
                state=state,
                action=action_dist,
                dt=1,
                reward=reward_dist,
                terminal=bool(_end),
                obs=obs,
            )

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


class PesteFMC(PesteWave):
    def __init__(
        self,
        path_balance: float = 1,
        skip_initial_frames: int = 0,
        min_horizon: int = 10,
        *args,
        **kwargs
    ):
        super(PesteFMC, self).__init__(*args, **kwargs)
        self.peste_walks = np.ones(self.rewards.shape)
        self.path_balance = path_balance
        self.agent_path = DLTree()

        self.skip_initial_frames = skip_initial_frames
        self.min_horizon = min_horizon

        self.init_ids = np.zeros(self.n_walkers).astype(int)
        self.entropy_dist = np.zeros(self.n_walkers).astype(int)

        self._save_steps = []
        self._agent_reward = 0
        self._last_action = None
        self.peste_agent = 1

    @property
    def init_actions(self):
        return self.data.get_actions(self.init_ids)

    def init_swarm(self, state: np.ndarray = None, obs: np.ndarray = None):

        super(PesteFMC, self).init_swarm(state=state, obs=obs)
        self.init_ids = np.zeros(self.n_walkers).astype(int)

    def clone(self):
        super(PesteFMC, self).clone()
        if self._clone_idx is None:
            return
        self.init_ids = np.where(self._will_clone, self.init_ids[self._clone_idx], self.init_ids)

    def update_data(self):
        init_actions = list(set(np.array(self.init_ids).astype(int)))
        walker_data = list(set(np.array(self.walkers_id).astype(int)))
        self.data.update_values(set(walker_data + init_actions))

    def virtual_reward(self) -> np.ndarray:
        """Calculate the virtual reward of the walkers. This quantity is used for determining
                the chance a given walker has of cloning. This scalar gives us a measure of how
                well a
                walker is solving the exploration vs exploitation problem, with respect to the
                other
                walkers of the Swarm.
                """
        vir_reward = super(PesteFMC, self).virtual_reward()
        path_obs = self.get_path_obs()
        self.peste_agent = self.peste_path(path_obs)
        peste_reward = vir_reward * self.peste_agent ** self.path_balance
        return peste_reward

    def peste_path(self, peste_obs) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the result applying the relativize function.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even if it is not a proper distance.
        """
        # Get random companion
        # Euclidean distance between states (pixels / RAM)
        obs = self.observations.astype(np.float32).reshape((self.n_walkers, -1))
        dist = np.sqrt(np.sum((peste_obs - obs) ** 2, axis=tuple(range(1, len(obs.shape)))))
        return relativize_vector(dist)

    def __get_peste_obs(self):
        nodes = np.random.choice(list(self.tree.data.nodes), self.n_walkers)
        observs = []
        for node in nodes:
            if node == "-1":
                node = "0"
            try:
                node_data = self.tree.data.nodes[node]
                observs.append(node_data["obs"].copy())
            except Exception as ex:
                print(node, self.tree.data.nodes[node])
                raise ex
        return np.array(observs)

    def get_path_obs(self):
        nodes = np.random.choice(list(self.agent_path.data.nodes), self.n_walkers)
        observs = []
        for node in nodes:
            if node == "-1":
                node = "0"
            try:
                observs.append(self.agent_path.data.node[node]["obs"].copy())
            except Exception as ex:
                print(node)
                raise ex
        return np.array(observs)

    def _skip_initial_frames(self) -> tuple:
        state, obs = self._env.reset(return_state=True)
        i_step, self._agent_reward, end = 0, 0, False
        info = {}
        _reward = 0
        for i in range(self.skip_initial_frames):
            i_step += 1
            action = 0
            state, obs, _reward, _end, info = self._env.step(
                state=state, action=action, n_repeat_action=self.min_dt
            )
            self.agent_path.append_leaf(
                i_step,
                parent_id=i_step - 1,
                obs=obs,
                reward=_reward,
                terminal=_end,
                state=state,
                action=action,
                dt=self._env.n_repeat_action,
            )
            self._agent_reward += _reward
            self._last_action = action
            end = info.get("terminal", _end)
            if end:
                break
        return state, obs, _reward, end, info, i_step

    def run_agent(self, render: bool = False, print_swarm: bool = False):
        """

        :param render:
        :param print_swarm:
        :return:
        """
        self.agent_path.reset()
        state, obs, _reward, end, info, i_step = self._skip_initial_frames()
        self._save_steps = []

        self.agent_path.append_leaf(
            i_step,
            parent_id=i_step - 1,
            obs=obs,
            reward=_reward,
            terminal=end,
            state=state,
            action=0,
            dt=1,
        )

        while not end and self._agent_reward < self.reward_limit:
            i_step += 1
            self.run_swarm(state=copy.deepcopy(state), obs=obs)
            action = self.weight_actions()

            state, obs, _reward, _end, info = self._env.step(
                state=state, action=action, n_repeat_action=self.min_dt
            )
            self.agent_path.append_leaf(
                i_step,
                parent_id=i_step - 1,
                obs=obs,
                reward=_reward,
                terminal=_end,
                state=state,
                action=action,
                dt=self._env.n_repeat_action,
            )
            self._agent_reward += _reward
            self._last_action = action
            end = info.get("terminal", _end)

            if render:
                self._env.render()
            if print_swarm:
                print(self)
                from IPython.core.display import clear_output

                clear_output(True)

    def weight_actions(self) -> np.ndarray:
        """Gets an approximation of the Q value function for a given state.

        It weights the number of times a given initial action appears in each state of the swarm.
        The the proportion of times each initial action appears on the swarm, is proportional to
        the Q value of that action.
        """
        init_actions = np.array(self.init_actions.copy())
        self.entropy_dist = np.array(
            [init_actions[init_actions == action].sum() for action in range(self.env.n_actions)]
        )
        return self.entropy_dist.argmax()

    def __str__(self):
        text = super(PesteFMC, self).__str__()
        return "Entropy: {} \n".format(self.entropy_dist) + text

    def run_swarm(
        self, state: np.ndarray = None, obs: np.ndarray = None, print_swarm: bool = False
    ):
        """
        Iterate the swarm by evolving and cloning each walker until a certain condition
        is met.
        :return:
        """
        self.reset()
        self.init_swarm(state=state, obs=obs)
        while not self.stop_condition():
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

        if print_swarm:
            print(self)
