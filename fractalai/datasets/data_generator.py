import os
import copy
import random
import string
import networkx as nx
import numpy as np
from fractalai.swarm_wave import SwarmWave, DynamicTree, Swarm


class DLTree(DynamicTree):
    def append_leaf(
        self,
        leaf_id: int,
        parent_id: int,
        state,
        action,
        dt: [int, np.ndarray],
        obs: np.ndarray,
        reward: [float, np.ndarray],
        terminal: bool,
    ):
        """
        Add a new state as a leaf node of the tree to keep track of the trajectories of the swarm.
        :param leaf_id: Id that identifies the state that will be added to the tree.
        :param parent_id: id that references the state of the system before taking the action.
        :param state: observation assigned to leaf_id state.
        :param action: action taken at leaf_id state.
        :param dt: parameters taken into account when integrating the action.
        :return:
        """
        parent_id = str(parent_id)
        leaf_id = str(leaf_id)

        if parent_id == "0":
            self.data.add_node(parent_id, state=copy.deepcopy(state), obs=copy.deepcopy(obs))
        self.data.add_node(leaf_id, state=copy.deepcopy(state), obs=copy.deepcopy(obs))
        self.data.add_edge(
            parent_id,
            leaf_id,
            action=copy.deepcopy(action),
            dt=copy.deepcopy(dt),
            reward=copy.deepcopy(reward),
            terminal=copy.deepcopy(terminal),
        )

    def reset(self):
        self.data.remove_edges_from(list(self.data.edges))
        self.data.remove_nodes_from(list(self.data.nodes))
        self.data.add_node("0")
        self.root_id = "0"

    @property
    def dataset(self):
        return self.data

    def edge_to_example(self, edge_id):
        str_edges = str(edge_id[0]), str(edge_id[1])
        edge = copy.deepcopy(self.dataset.edges[str_edges])
        prev_node = copy.deepcopy(self.dataset.nodes[str(edge_id[0])])
        next_node = copy.deepcopy(self.dataset.nodes[str(edge_id[1])])
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
        nodes = nx.shortest_path(self.data, "0", str(leaf_id))
        states = [self.data.node[n]["state"] for n in nodes[:-1]]
        actions = [self.data.edges[(n, nodes[i + 1])]["action"] for i, n in enumerate(nodes[:-1])]
        ends = [self.data.edges[(n, nodes[i + 1])]["terminal"] for i, n in enumerate(nodes[:-1])]
        obs = [self.data.node[n]["obs"] for n in nodes[:-1]]
        new_obs = [self.data.node[n]["obs"] for n in nodes[1:]]
        rewards = [self.data.edges[(n, nodes[i + 1])]["reward"] for i, n in enumerate(nodes[:-1])]
        return states, obs, actions, rewards, new_obs, ends

    def get_state_branch(self, leaf_id):
        nodes = nx.shortest_path(self.data, "0", str(leaf_id))
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

    def get_past_obs(self, node, n_past: int):
        node = str(node)
        target = node
        for i in range(n_past):
            parents = list(self.data.in_edges(target))
            if len(parents) == 0:
                break
            else:
                target = parents[0][0]  # Update target with parent
        return self.data.node[target]["obs"]


class DataGenerator:
    def __init__(
        self,
        swarm: ["MLFMC", "MLWave"],
        save_to_disk: bool = False,
        output_dir: str = None,
        obs_is_image: bool = False,
        obs_shape: tuple = (42, 42, 3),
    ):

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
        new_text = (
            "{}\n"
            "Generated {} Examples |"
            " {:.2f} samples per example.\n".format(text, samples, sam_step)
        )
        return new_text

    @property
    def tree(self):
        return self.swarm.tree

    def example_generator(
        self, remove_nodes: bool = False, print_val: bool = False, *args, **kwargs
    ):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.collect_data(*args, **kwargs)
        generator = self.tree.example_generator(remove_nodes)
        n_nodes = len(self.tree.dataset.nodes)
        if print_val:
            print("Generated {} examples".format(n_nodes))
        for val in generator:
            yield val

    def game_state_generator(self, print_swarm: bool = False, print_val: bool = False):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.run_swarm(print_swarm=print_swarm)
        walker_ids = self.swarm.walkers_id[self.swarm.rewards.argmax()]
        generator = self.tree.game_state_generator(walker_ids)
        n_nodes = len(self.tree.data.nodes)
        if print_val:
            print("Generated {} examples".format(n_nodes))
        for val in generator:
            yield val

    def best_game_examples(self):
        states, obs, actions, rewards, new_obs, ends = next(self.best_game_generator())
        for i in range(len(states)):
            yield obs[i], actions[i], rewards[i], new_obs[i], ends[i]

    def best_game_generator(self, print_val: bool = False, *args, **kwargs):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.collect_data(*args, **kwargs)
        best_id = self.swarm.get_best_id()
        generator = self.tree.one_game_generator(best_id)
        n_nodes = len(self.tree.data.nodes)
        if print_val:
            print("Generated {} examples".format(n_nodes))
        for val in generator:
            yield val

    def game_generator(self, print_val: bool = False, *args, **kwargs):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.collect_data(*args, **kwargs)
        generator = self.tree.game_generator()
        n_nodes = len(self.tree.data.nodes)
        if print_val:
            print("Generated {} examples".format(n_nodes))
        for val in generator:
            if len(val[0]) > 1:
                yield val

    def batch_generator(
        self,
        batch_size: int = 16,
        epochs: int = 1,
        remove_nodes: bool = True,
        print_val: bool = False,
        *args,
        **kwargs
    ):
        self.tree.reset()
        self.swarm.reset()
        self.swarm.collect_data(*args, **kwargs)
        n_nodes = len(self.tree.data.nodes)
        if print_val:
            print("Generated {} examples".format(n_nodes))
        # print(self)
        for i in range(epochs - 1):
            print("epoch {}".format(i + 1))
            for batch in self._batch_generator(batch_size, remove_nodes=False):
                yield batch
        print("epoch {}".format(epochs))
        for batch in self._batch_generator(batch_size, remove_nodes=remove_nodes):
            yield batch

    def _batch_generator(self, batch_size=16, remove_nodes: bool = True):
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
                yield np.array(state_batch), np.array(action_batch), np.array(
                    reward_batch
                ), np.array(next_state_batch), np.array(terminal_batch)
                state_batch = []
                action_batch = []
                reward_batch = []
                next_state_batch = []
                terminal_batch = []

    def save_run(self, folder: str = None, uid: str = None, *args, **kwargs):
        generator = self.best_game_generator(*args, **kwargs)
        _, state_b, action_b, reward_b, next_state_b, terminal_b = next(generator)

        def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
            return "".join(random.choice(chars) for _ in range(size))

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
