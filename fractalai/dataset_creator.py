import os
import copy
import string
import random
import numpy as np

import networkx as nx
from fractalai.environment import Environment, resize_frame
from fractalai.swarm_wave import SwarmWave, DynamicTree, Swarm


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

    #def append_to_dataset(self, leaf_id: str, parent_id: str, state, action, dt: int):
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

    #def process_stacking(self):
    #    for node_id in self.dataset.nodes:
    #        self._stack_one_state(node_id)

    #def expand_dataset(self, parent, child, state, action):
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

    #def stack_observations(self, new_frame, last_obs):
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

    def example_generator(self, remove_nodes=True):
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
        #if parent_id != 0:
        #    assert(not self.data.nodes[int(parent_id)]["terminal"])
        self.data.add_node(int(leaf_id), state=copy.deepcopy(state), obs=copy.deepcopy(obs))
        self.data.add_edge(int(parent_id), int(leaf_id), action=copy.deepcopy(action),
                           dt=copy.deepcopy(dt), reward=copy.deepcopy(reward),
                           terminal=copy.deepcopy(terminal))

    @property
    def dataset(self):
        return self.data

    def edge_to_example(self, edge_id):
        #print(edge_id)
        edge = copy.deepcopy(self.dataset.edges[tuple(edge_id)])
        prev_node = copy.deepcopy(self.dataset.nodes[edge_id[0]])
        next_node = copy.deepcopy(self.dataset.nodes[edge_id[1]])
        old_obs = copy.deepcopy(prev_node["obs"])
        new_obs = copy.deepcopy(next_node["obs"])
        action = copy.deepcopy(edge["action"])
        reward = copy.deepcopy(edge["reward"])
        done = copy.deepcopy(edge["terminal"])
        return old_obs, action, reward, new_obs, done

    def example_generator(self, remove_nodes=True):
        for edge in np.random.permutation(list(self.data.edges)):
            yield self.edge_to_example(edge)
            #if remove_nodes:
            #    self.data.remove_edge(edge)

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

        self.img_shape = obs_shape[:-1] #if obs_is_image else None

        self.frame_width = obs_shape[0] #if obs_is_image else None
        self.frame_height = obs_shape[1] #if obs_is_image else None
        self.stack_frames = obs_shape[2] #if obs_is_image else None
        self.obs_shape = obs_shape #self.tree.obs_shape #if obs_is_image else None

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
















"""
solving pendulum using actor-critic model
"""

import gym
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K

import tensorflow as tf

import random
from collections import deque

# determines how to assign values to each state, i.e. takes the state
# and action (two-input model) and determines the corresponding value
class ActorCritic:
    def __init__(self, env, sess):
        self.env = env
        self.sess = sess

        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = .995
        self.gamma = .95
        self.tau = .125

        # ===================================================================== #
        #                               Actor Model                             #
        # Chain rule: find the gradient of chaging the actor network params in  #
        # getting closest to the final value network predictions, i.e. de/dA    #
        # Calculate de/dA as = de/dC * dC/dA, where e is error, C critic, A act #
        # ===================================================================== #

        self.memory = deque(maxlen=2000)
        self.actor_state_input, self.actor_model = self.create_actor_model()
        _, self.target_actor_model = self.create_actor_model()

        self.actor_critic_grad = tf.placeholder(tf.float32,
                                                [None, self.env.action_space.shape[
                                                    0]])  # where we will feed de/dC (from critic)

        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output,
                                        actor_model_weights,
                                        -self.actor_critic_grad)  # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)

        # ===================================================================== #
        #                              Critic Model                             #
        # ===================================================================== #

        self.critic_state_input, self.critic_action_input, \
        self.critic_model = self.create_critic_model()
        _, _, self.target_critic_model = self.create_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,
                                         self.critic_action_input)  # where we calcaulte de/dC for feeding above

        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())

    # ========================================================================= #
    #                              Model Definitions                            #
    # ========================================================================= #

    def create_actor_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0], activation='relu')(h3)

        model = Model(input=state_input, output=output)
        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, model

    def create_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)

        action_input = Input(shape=self.env.action_space.shape)
        action_h1 = Dense(48)(action_input)

        merged = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model = Model(input=[state_input, action_input], output=output)

        adam = Adam(lr=0.001)
        model.compile(loss="mse", optimizer=adam)
        return state_input, action_input, model

    # ========================================================================= #
    #                               Model Training                              #
    # ========================================================================= #

    def remember(self, cur_state, action, reward, new_state, done):
        self.memory.append([cur_state, action, reward, new_state, done])

    def _train_actor(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, _ = sample
            predicted_action = self.actor_model.predict(cur_state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input: cur_state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: cur_state,
                self.actor_critic_grad: grads
            })

    def _train_critic(self, samples):
        for sample in samples:
            cur_state, action, reward, new_state, done = sample
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([cur_state, action], reward, verbose=0)

    def train(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        rewards = []
        samples = random.sample(self.memory, batch_size)
        self._train_critic(samples)
        self._train_actor(samples)

    # ========================================================================= #
    #                         Target Model Updating                             #
    # ========================================================================= #

    def _update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)

    def _update_critic_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()

    # ========================================================================= #
    #                              Model Predictions                            #
    # ========================================================================= #

    def act(self, cur_state):
        self.epsilon *= self.epsilon_decay
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(cur_state)

def main():
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make("Pendulum-v0")
    actor_critic = ActorCritic(env, sess)

    num_trials = 10000
    trial_len = 500

    cur_state = env.reset()
    action = env.action_space.sample()
    while True:
        env.render()
        cur_state = cur_state.reshape((1, env.observation_space.shape[0]))
        action = actor_critic.act(cur_state)
        action = action.reshape((1, env.action_space.shape[0]))

        new_state, reward, done, _ = env.step(action)
        new_state = new_state.reshape((1, env.observation_space.shape[0]))

        actor_critic.remember(cur_state, action, reward, new_state, done)
        actor_critic.train()

        cur_state = new_state

if __name__ == "__main__":
    main()
