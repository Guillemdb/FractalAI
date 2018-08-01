import os
import copy
import string
import random
import numpy as np
import networkx as nx
from fractalai.environment import Environment
from fractalai.swarm_wave import DynamicTree, Swarm


class DataTree(DynamicTree):

    def __init__(self, env: Environment, obs_is_image: bool=False,
                 stack_frames: int=2, img_shape: tuple=(42, 42)):

        super(DataTree, self).__init__()
        self.obs_is_image = obs_is_image
        self.root_id_ds = "0_0"
        self.stack_frames = stack_frames
        self.env = env
        self.dataset = nx.DiGraph()

        self.img_shape = img_shape
        self.width = img_shape[0]
        self.height = img_shape[1]
        self.obs_shape = img_shape + tuple([self.stack_frames])
        self.reset()

    def append_leaf(self, leaf_id: int, parent_id: int, state, action, dt: int):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param dt: parameters taken into account when integrating the action.
        :return:
        """
        super(DataTree, self).append_leaf(leaf_id=leaf_id, parent_id=parent_id, state=state,
                                          action=action, dt=dt)

    # def create_dataset(self):
    #    for node_id in self.data.nodes:
    #        if node_id == 0:
    #            continue
    #        node = self.data.nodes[node_id]
    #        state = node["state"]

    #        parent_id = list(self.data.in_edges([node_id]))[0][0]
    #        edge = self.data.edges[(int(parent_id), int(node_id))]
    #        action, dt = edge["action"], edge["dt"]
    #
    #       self.append_to_dataset(str(node_id), str(parent_id), state, action, dt)

    def get_last_dt_id(self, node_id: str):
        index = -1
        node = False
        while node is not None:
            index += 1
            node_id = "{}_{}".format(node_id.split("_")[0], index)
            node = self.dataset.nodes.get(node_id)

        return "{}_{}".format(node_id.split("_")[0], index - 1)

    # def append_to_dataset(self, leaf_id: str, parent_id: str, state, action, dt: int):
    #    parent_last_child = self.get_last_dt_id(parent_id)
    #    for ti in range(dt):
    #        true_leaf_id = "{}_{}".format(leaf_id, ti)
    #        self.expand_dataset(parent_last_child, true_leaf_id, state, action)
    #        parent_last_child = self.get_last_dt_id(leaf_id)

    def process_action(self, action):
        actions = np.zeros(self.env.action_space.n)
        actions[action] = 1
        return actions

    # def _stack_one_state(self, node_id: str):
    #    old_obs = np.zeros(self.obs_shape)
    #    node = self.dataset.nodes[node_id]
    #    old_obs[:, :, 0] = node["old_frame"].copy()
    #    child_id = node_id
    #    for i in range(1, self.stack_frames):
    #        in_e = list(self.dataset.in_edges([child_id]))
    #        # print(in_e)
    #        if len(in_e) > 0:
    #            parent_id = in_e[0][0]
    #            parent = self.dataset.nodes[parent_id]
    #            old_obs[:, :, i] = parent["old_frame"].copy()
    #            child_id = parent_id
    #    self.dataset.nodes[node_id]["old_obs"] = old_obs.copy()
    #    new_obs = old_obs.copy()
    #    new_obs[:, :, 0] = node["new_frame"].copy()
    #    new_obs[:, :, 1:] = old_obs[:, :, :-1].copy()
    #    self.dataset.nodes[node_id]["new_obs"] = new_obs.copy()

    # def process_stacking(self):
    #    for node_id in self.dataset.nodes:
    #        self._stack_one_state(node_id)

    # def expand_dataset(self, parent, child, state, action):
    #    parent_state = self.dataset.nodes[parent]["state"]
    #    _, obs, reward, end, info = self.env.step(action, parent_state, n_repeat_action=1)
    #    new_obs = self.reshape_frame(obs)
    #    old_obs = copy.copy(self.dataset.nodes[parent]["new_frame"])
    #    # new_obs = copy.copy(self.stack_observations(obs, old_obs))
    #    action = self.process_action(action)
    #    self.dataset.add_node(child, state=state, new_frame=new_obs,
    #                          reward=reward,
    #                          old_frame=old_obs,
    #                          action=action, end=end)
    #    self.dataset.add_edge(parent, child, action=action)

    # def stack_observations(self, new_frame, last_obs):
    #    obs = np.zeros(self.obs_shape)
    #    obs[:, :, 0] = new_frame[:, :, 0]
    #    obs[:, :, 1:] = last_obs[:, :, :-1]
    #    return obs

    def edge_to_example(self, edge_id):
        edge = self.dataset.nodes[edge_id]
        prev_node = self.dataset.nodes[edge_id[0]]
        next_node = self.dataset.nodes[edge_id[1]]
        old_obs = prev_node["obs"]
        new_obs = next_node["obs"]
        action = edge["action"]
        reward = edge["reward"]
        done = edge["end"]
        return old_obs, action, reward, new_obs, done

    def example_generator(self, remove_nodes=False):
        # self.create_dataset()
        # self.process_stacking()
        for node in np.random.permutation(list(self.dataset.edges)):
            yield self.edge_to_example(node)
            if remove_nodes:
                self.dataset.remove_node(node)

    # def reshape_frame(self, frame):
        # breakout
        # return resize_frame(frame[25: 195], height=self.height, width=self.width)[:, :, 0]
        # pong
    #    return resize_frame(frame[32:196, 5:155], height=self.height, width=self.width)[:, :, 0]

    def reset(self):
        super(DataTree, self).reset()
        self.dataset.remove_edges_from(list(self.dataset.edges))
        self.dataset.remove_nodes_from(list(self.dataset.nodes))
        # self.dataset = nx.DiGraph()
        _, obs = self.env.reset()
        if self.obs_is_image:
            new_obs = self.reshape_frame(obs)
            old_obs = np.zeros(self.img_shape)
        else:
            new_obs = obs
            old_obs = np.zeros(len(obs))
        self.dataset.add_node("0_0", old_frame=new_obs, new_frame=old_obs, reward=0,
                              action=self.process_action(0),
                              end=False, state=self.env.get_state())
        self.data.add_node(0)


class DLTree(DynamicTree):

    def append_leaf(self, leaf_id: int, parent_id: int, state, action, dt: int,
                    obs: np.ndarray, reward: float, terminal: bool):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param dt: parameters taken into account when integrating the action.
        :return:
        """
        self.data.add_node(int(leaf_id), state=copy.deepcopy(state), obs=copy.deepcopy(obs))
        self.data.add_edge(int(parent_id), int(leaf_id), action=copy.deepcopy(action),
                           dt=copy.deepcopy(dt), reward=copy.deepcopy(reward),
                           terminal=copy.deepcopy(terminal))

    @property
    def dataset(self):
        return self.data

    def edge_to_example(self, edge_id):
        edge = copy.deepcopy(self.dataset.edges[tuple(edge_id)])
        prev_node = copy.deepcopy(self.dataset.nodes[edge_id[0]])
        next_node = copy.deepcopy(self.dataset.nodes[edge_id[1]])
        old_obs = copy.deepcopy(prev_node["obs"])
        new_obs = copy.deepcopy(next_node["obs"])
        action = copy.deepcopy(edge["action"])
        reward = copy.deepcopy(edge["reward"])
        done = copy.deepcopy(edge["terminal"])
        return old_obs, action, reward, new_obs, done

    def example_generator(self, remove_nodes=False):
        for edge in np.random.permutation(list(self.data.edges)):
            yield self.edge_to_example(edge)
            if remove_nodes:
                self.data.remove_edge(edge)

    def get_branch(self, leaf_id) -> tuple:
        """
        Get the observation from the game ended at leaf_id
        :param leaf_id: id of the leaf node belonging to the branch that will be recovered.
        :return: Sequence of observations belonging to a given branch of the tree.
        """
        nodes = nx.shortest_path(self.data, 0, leaf_id)
        states = [self.data.node[n]["state"] for n in nodes[:-1]]
        actions = [self.data.edges[(n, nodes[i+1])]["action"] for i, n in enumerate(nodes[:-1])]
        # dts = [self.data.edges[(n, nodes[i + 1])]["dt"] for i, n in enumerate(nodes[:-1])]
        ends = [self.data.edges[(n, nodes[i+1])]["terminal"] for i, n in enumerate(nodes[:-1])]
        obs = [self.data.node[n]["obs"] for n in nodes[:-1]]
        new_obs = [self.data.node[n]["obs"] for n in nodes[1:]]
        rewards = [self.data.edges[(n, nodes[i + 1])]["reward"] for i, n in enumerate(nodes[:-1])]
        return states, obs, actions, rewards, new_obs, ends

    def get_state_branch(self, leaf_id):
        nodes = nx.shortest_path(self.data, 0, leaf_id)
        states = [self.data.node[n]["state"] for n in nodes[:-1]]
        actions = [self.data.edges[(n, nodes[i + 1])]["action"] for i, n in enumerate(nodes[:-1])]
        dts = [self.data.edges[(n, nodes[i + 1])]["dt"] for i, n in enumerate(nodes[:-1])]
        return states, actions, dts

    def game_state_generator(self, leaf_id):
        yield self.get_state_branch(leaf_id)

    def game_generator(self):
        for leaf_id in self.get_leaf_nodes():
            yield self.get_branch(leaf_id)

    def one_game_generator(self, leaf_id):
            yield self.get_branch(leaf_id)


class DataGenerator:

    def __init__(self, swarm: Swarm, save_to_disk: bool=False,
                 output_dir: str=None,
                 obs_is_image: bool=False,
                 obs_shape: tuple=(42, 42, 3)):

        self.swarm = swarm
        self.obs_is_image = obs_is_image
        self.save_to_disk = save_to_disk
        self.output_dir = output_dir
        self.swarm.tree = DLTree()

        self.img_shape = obs_shape[:-1]

        self.frame_width = obs_shape[0]
        self.frame_height = obs_shape[1]
        self.stack_frames = obs_shape[2]
        self.obs_shape = obs_shape

    def __str__(self):
        text = self.swarm.__str__()
        if hasattr(self.swarm, "save_data"):
            if self.swarm.save_data:

                sam_step = self.swarm._n_samples_done / len(self.tree.dataset.nodes)
                samples = len(self.tree.dataset.nodes)
        else:
            efi, samples, sam_step = 0, 0, 0
        new_text = "{}\n"\
                   "Generated {} Examples |" \
                   " {:.2f} samples per example.\n".format(text, samples, sam_step)
        return new_text

    @property
    def tree(self):
        return self.swarm.tree

    def game_state_generator(self, print_swarm: bool=False):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.run_swarm(print_swarm=print_swarm)
        walker_ids = self.swarm.walkers_id[self.swarm.rewards.argmax()]
        generator = self.tree.game_state_generator(walker_ids)
        n_nodes = len(self.tree.data.nodes)
        print("Generated {} examples".format(n_nodes))
        print(self)
        for val in generator:
            yield val

    def example_generator(self, print_swarm: bool=False, remove_nodes: bool=True):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.run_swarm(print_swarm=print_swarm)
        generator = self.tree.example_generator(remove_nodes)
        n_nodes = len(self.tree.dataset.nodes)
        print("Generated {} examples".format(n_nodes))
        print(self)
        for val in generator:
            yield val

    def best_game_examples(self):
        """Yields separate examples of the best game played by the swarm one by one."""
        states, obs, actions, rewards, new_obs, ends = next(self.best_game_generator())
        for i in range(len(states)):
            yield obs[i], actions[i], rewards[i], new_obs[i], ends[i]

    def best_game_generator(self, print_swarm: bool=False):
        """Generator that yields the best game found running the swarm."""
        self.tree.reset()
        self.swarm.reset()
        self.swarm.run_swarm(print_swarm=print_swarm)
        walker_ids = self.swarm.walkers_id[self.swarm.rewards.argmax()]
        generator = self.tree.one_game_generator(walker_ids)
        n_nodes = len(self.tree.data.nodes)
        print("Generated {} examples".format(n_nodes))
        print(self)
        for val in generator:
            yield val

    def game_generator(self, print_swarm: bool=False):
        """Yields all the games that originate from each one of the branches of the state tree
        from the swarm."""
        self.tree.reset()
        self.swarm.reset()
        self.swarm.run_swarm(print_swarm=print_swarm)
        generator = self.tree.game_generator()
        n_nodes = len(self.tree.data.nodes)
        print("Generated {} examples".format(n_nodes))
        print(self)
        for val in generator:
            if len(val[0]) > 1:
                yield val

    def batch_generator(self, batch_size: int=16, epochs: int=1, print_swarm: bool=False,
                        remove_nodes: bool=True):
        """Yields the examples conforming the state tree of the swarm as batches."""
        self.tree.reset()
        self.swarm.reset()
        self.swarm.run_swarm(print_swarm=print_swarm)
        n_nodes = len(self.tree.data.nodes)
        print("Generated {} examples".format(n_nodes))
        print(self)
        for i in range(epochs - 1):
            print("epoch {}".format(i + 1))
            for batch in self._batch_generator(batch_size, remove_nodes=False):
                yield batch
        print("epoch {}".format(epochs))
        for batch in self._batch_generator(batch_size, remove_nodes=remove_nodes):
            yield batch

    def _batch_generator(self, batch_size=16, remove_nodes: bool=True):
        generator = self.tree.example_generator(remove_nodes)
        _ = next(generator)
        n_nodes = len(self.tree.data.nodes)
        batch_size = batch_size if batch_size > 0 else n_nodes
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        for data in generator:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])
            if len(terminal_batch) == batch_size - 1 or len(terminal_batch) == n_nodes - 1:
                yield np.array(state_batch), np.array(action_batch),\
                      np.array(reward_batch), np.array(next_state_batch), np.array(terminal_batch)
                state_batch = []
                action_batch = []
                reward_batch = []
                next_state_batch = []
                terminal_batch = []

    def save_run(self, folder: str=None, uid: str=None, print_swarm: bool=False):
        """Saves the best game generated by the swarm."""
        generator = self.best_game_generator(print_swarm=print_swarm)
        _, state_b, action_b, reward_b, next_state_b, terminal_b = next(generator)

        def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
            return ''.join(random.choice(chars) for _ in range(size))

        uid = uid if uid is not None else id_generator()
        folder = folder if folder is not None else self.output_dir

        def file_name(suffix):
            return os.path.join(folder, "{}_{}".format(uid, suffix))
        print("Saving to {} with uid {}".format(folder, uid))
        np.save(file_name("old_s"), state_b)
        np.save(file_name("act"), action_b)
        np.save(file_name("rew"), reward_b)
        np.save(file_name("new_s"), next_state_b)
        np.save(file_name("end"), terminal_b)
        print("Successfully saved")
