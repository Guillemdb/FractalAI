import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pickle
import copy
import keras
from keras import Model
import keras.backend as K
from fractalai.environment import Environment
from fractalai.swarm_wave import SwarmWave, relativize_vector
from fractalai.datasets.mlswarm import PesteWave


class act_space:
    def __init__(self):
        self.n = 4294967295


def nn_callable(_class, epochs, batch_size):
    def _dummy():
        return _class(epochs, batch_size)

    return _dummy


def save_obj(obj, path_file):
    with open(path_file, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(path_file):
    with open(path_file, "rb") as f:
        return pickle.load(f)


class DNN:
    def __init__(self):
        self.train_batch_gen = None
        self.test_batch_gen = None
        self.model = None
        self.X = None
        self.y = None

    def reset_train_gen(self, *args, **kwargs):
        raise NotImplementedError

    def reset_test_gen(self, *args, **kwargs):
        raise NotImplementedError

    def set_weights(self, weights: list):
        raise NotImplementedError

    def get_weights(self):
        raise NotImplementedError

    def get_next_batch(self, train: bool = True, *args, **kwargs):
        if train:
            if self.train_batch_gen is None:
                self.reset_train_gen(*args, **kwargs)
            try:
                return next(self.train_batch_gen)
            except StopIteration as si:
                self.reset_train_gen(*args, **kwargs)
                return next(self.train_batch_gen)
        else:
            if self.test_batch_gen is None:
                self.reset_test_gen(*args, **kwargs)
            try:
                return next(self.test_batch_gen)
            except StopIteration as si:
                self.reset_test_gen(*args, **kwargs)
                return None

    def train_on_batch(self, *args, **kwargs) -> [float, np.ndarray]:
        raise NotImplementedError

    def reward_on_batches(self, n_batches: int = 1, *args, **kwargs) -> [float, np.ndarray]:
        return np.mean([self.train_on_batch(*args, **kwargs) for _ in range(n_batches)])

    def update_weights(self, *args, **kwargs):
        pass


def split_chunks(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


class KerasDNN(DNN):
    def __init__(
        self,
        model: Model,
        dataset,
        noise_stepsize: float = 0.01,
        batch_size: int = 31,
        decay: float = 0,
    ):
        super(KerasDNN, self).__init__()
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.noise_stepsize = noise_stepsize
        self.noise_decay = decay
        (self.X_train, self.y_train), (self.X_test, self.y_test) = dataset.load_data()
        self.loss = None
        self.metric = None
        self.X = None
        self.y = None

    @staticmethod
    def is_int(x):
        vals = (np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint8, np.uint32, int)
        return isinstance(x, vals)

    def new_train_gen(self, batch_size: int = None, *args, **kwargs):
        batch_s = batch_size if batch_size is not None else self.batch_size
        new_idx = np.random.permutation(np.arange(len(self.y_train)))
        self.X_train, self.y_train = self.X_train[new_idx], self.y_train[new_idx]
        return split_chunks(list(zip(self.X_train, self.y_train)), batch_s)

    def reset_train_gen(self, batch_size: int = None, *args, **kwargs):
        self.train_batch_gen = self.new_train_gen(batch_size=batch_size, *args, **kwargs)

    def reset_test_gen(self, batch_size: int = None, *args, **kwargs):
        self.test_batch_gen = self.new_test_gen(batch_size=batch_size, *args, **kwargs)

    def new_test_gen(self, batch_size: int = None, *args, **kwargs):
        batch_s = batch_size if batch_size is not None else self.batch_size
        new_idx = np.random.permutation(np.arange(len(self.y_test)))
        self.X_test, self.y_test = self.X_test[new_idx], self.y_test[new_idx]
        return split_chunks(list(zip(self.X_test, self.y_test)), batch_s)

    def set_weights(self, weights: list):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def train_on_batch(self, action, n_repeat_action: int = 1, *args, **kwargs):
        losses, metrics = [], []
        # for i in range(n_repeat_action):
        data = self.get_next_batch(train=True)
        if not self.is_int(n_repeat_action):
            K.set_value(self.model.optimizer.lr, n_repeat_action)
        rate = n_repeat_action if self.is_int(n_repeat_action) else 10
        for i in range(rate):
            while len(data) < self.batch_size:
                data = self.get_next_batch(train=True)
            X, y = list(zip(*data))
            self.X, self.y = np.array(X), np.array(y)
            # Dynamic learning rate

            loss, metric = self.model.train_on_batch(self.X, self.y, *args, **kwargs)
            losses.append(loss)
            metrics.append(metric)

            old_weigths = self.model.get_weights()
            new_weights = self.update_weights(old_weigths, action, n_repeat_action)
            self.model.set_weights(new_weights)
        self.loss, self.metric = np.mean(losses), np.mean(metrics)
        return self.metric  # / self.loss

    def update_weights(self, weights, action, n_repeat_action):
        new_w = []
        self.noise_stepsize = self.noise_stepsize ** (1 + self.noise_decay)
        np.random.seed(action)
        lr = K.get_value(self.model.optimizer.lr)
        self.noise_stepsize = lr
        for w in weights:
            perturbation = (
                n_repeat_action
                if not self.is_int(n_repeat_action)
                else lr * np.random.standard_normal(w.shape)
            )
            w += perturbation
            new_w.append(w)
        return new_w


class DNNSupervised(Environment):
    """Inherit from this class to test the Swarm on a different problem.
     Pretty much the same as the OpenAI gym env."""

    action_space = None
    observation_space = None
    reward_range = None
    metadata = None

    def __init__(self, net: DNN, n_repeat_action: int = 1):
        super(DNNSupervised, self).__init__(name="TrainDNN-v0", n_repeat_action=n_repeat_action)
        self.net = net

    def step(self, action, state=None, n_repeat_action: int = 1, *args, **kwargs) -> tuple:
        """
        Take a simulation step and make the environment evolve.
        :param action: Chosen action applied to the environment.
        :param state: Set the environment to the given state before stepping it.
        :param n_repeat_action: Consecutive number of times that the action will be applied.
        :return:
        """
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        if state is not None:
            self.set_state(state)
        # rewards = [self.net.train_on_batch(action, *args, **kwargs)
        #           for _ in range(n_repeat_action)]
        reward = self.net.train_on_batch(
            action, n_repeat_action, *args, **kwargs
        )  # np.mean(rewards)
        obs = self.net.model.predict(self.net.X)
        info = {"terminal": False, "lives": 0}
        if state is not None:
            data = self.get_state(), obs.copy(), reward, False, info
        else:
            data = obs.copy(), reward, False, info
        return data

    def step_batch(self, actions, states=None, n_repeat_action: [int, np.ndarray] = None) -> tuple:
        """
        Take a step on a batch of states.
        :param actions: Chosen action applied to the environment.
        :param states: Set the environment to the given state before stepping it.
        :param n_repeat_action: Consecutive number of times that the action will be applied.
        :return:
        """
        n_repeat_action = n_repeat_action if n_repeat_action is not None else self.n_repeat_action
        n_repeat_action = (
            n_repeat_action
            if isinstance(n_repeat_action, np.ndarray)
            else np.ones(len(states)) * n_repeat_action
        )
        data = [
            self.step(action, state, n_repeat_action=dt)
            for action, state, dt in zip(actions, states, n_repeat_action)
        ]
        new_states, observs, rewards, terminals, infos = [], [], [], [], []
        for d in data:
            if states is None:
                obs, _reward, end, info = d
            else:
                new_state, obs, _reward, end, info = d
                new_states.append(new_state)
            observs.append(obs)
            rewards.append(_reward)
            terminals.append(end)
            infos.append(info)
        if states is None:
            return observs, rewards, terminals, infos
        else:
            return new_states, observs, rewards, terminals, infos

    def reset(self, return_state: bool = True):
        data = self.step(0)
        if not return_state:
            return data[0]
        else:
            return self.get_state(), data[0]

    def get_state(self):
        """Recover the internal state of the simulation."""
        return self.net.get_weights()

    def set_state(self, state):
        """Set the internal state of the simulation.
        :param state: Target state to be set in the environment.
        :return:
        """
        self.net.set_weights(state)


class NetWave(PesteWave):
    def track_best_walker(self):
        """The last walker represents the best solution found so far. It gets frozen so
         other walkers can always compare to it when cloning."""
        # Last walker stores the best value found so far so other walkers can clone to it
        self._not_frozen[-1] = False
        self._will_clone[-1] = False
        # self._will_step[-1] = False
        best_walker = self.rewards.argmax()
        if best_walker != self.n_walkers - 1:
            self.walkers_id[-1] = int(self.walkers_id[best_walker])
            self.observations[-1] = copy.deepcopy(self.observations[best_walker])
            self.rewards[-1] = float(self.rewards[best_walker])
            self.times[-1] = float(self.times[best_walker])
            self._end_cond[-1] = bool(self._end_cond[best_walker])
            # A free mutable copy
            self.walkers_id[-2] = int(self.walkers_id[best_walker])
            self.observations[-2] = copy.deepcopy(self.observations[best_walker])
            self.rewards[-2] = float(self.rewards[best_walker])
            self.times[-2] = float(self.times[best_walker])
            self._end_cond[-2] = bool(self._end_cond[best_walker])

    def __str__(self):
        text = super(NetWave, self).__str__()
        batches_per_epoch = self.env.net.X_train.shape[0] / self.env.net.batch_size
        batches_network = self._n_samples_done / self.n_walkers
        epochs_net = batches_network / batches_per_epoch

        new_text = (
            "Learning rate: {:.7f} Accuracy: {:.5f}  loss {:.5f}\n"
            "Epochs per net: {:.2f}\n".format(
                self.env.net.noise_stepsize, self.env.net.metric, self.env.net.loss, epochs_net
            )
        )
        return new_text + text

    def peste_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the result applying the relativize function.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even if it is not a proper distance.
        """
        # Get random companion
        peste_obs = self.get_peste_obs()
        # Euclidean distance between states (pixels / RAM)
        # obs = self.observations.astype(np.float32).reshape((self.n_walkers, -1))
        dist = self.wasserstein_distance(np.array(self.observations), peste_obs)
        return relativize_vector(dist)

    @staticmethod
    def _one_distance(ws1, ws2):
        dist = 0
        for w1, w2 in zip(ws1, ws2):
            dist += np.sqrt(np.sum((w1 - w2) ** 2))
        return dist

    @staticmethod
    def wasserstein_distance(x, y):
        def entropy_dist(x, y):
            def hernandez_crossentropy(x, y):
                return 1 + np.log(np.prod(2 - x ** y, axis=2))

            first = hernandez_crossentropy(x, y).mean(axis=1)
            sec = hernandez_crossentropy(y, x).mean(axis=1)
            return np.maximum(first, sec)

        def _wasserstein_distance(x, y):
            from scipy import stats

            def stacked_distance(x, y):
                distances = []
                for i in range(x.shape[0]):
                    dist_val = stats.wasserstein_distance(x[i], y[i])
                    distances.append(dist_val)
                return np.array(distances)

            distances = []
            for i in range(x.shape[0]):
                dist_val = stacked_distance(x[i], y[i]).mean()
                distances.append(dist_val)
            return np.array(distances)

        return _wasserstein_distance(x, y)

    def evaluate_distance(self) -> np.ndarray:
        """Calculates the euclidean distance between pixels of two different arrays
        on a vector of observations, and normalizes the result applying the relativize function.
        In a more general scenario, any function that quantifies the notion of "how different two
        observations are" could work, even if it is not a proper distance.
        """

        # Get random companion
        idx = np.random.permutation(np.arange(self.n_walkers, dtype=int))
        # Euclidean distance between states (pixels / RAM)
        obs = self.observations.astype(np.float32)
        dist = self.wasserstein_distance(obs[idx], obs)  # ** 2
        return relativize_vector(dist)

    def __clone_observations(self, idx: np.ndarray):
        observations = []
        for i, (compa, clone) in enumerate(zip(idx, self._will_clone.tolist())):
            obs = self.observations[compa] if clone else self.observations[i]
            observations.append(copy.deepcopy(obs))
        self.observations = observations

    def __perform_clone(self):
        idx = self._clone_idx
        # A hack that avoid cloning
        if idx is None:
            return
        # This is a hack to make it work on n dimensional arrays
        self.clone_observations(idx)
        # Using np.where seems to be faster than using a for loop
        self.rewards = np.where(self._will_clone, self.rewards[idx], self.rewards)
        self._end_cond = np.where(self._will_clone, self._end_cond[idx], self._end_cond)
        self.times = np.where(self._will_clone, self.times[idx], self.times)

        self.walkers_id = np.where(self._will_clone, self.walkers_id[idx], self.walkers_id).astype(
            int
        )

    def step_walkers(self):
        """Sample an action for each walker, and act on the environment. This is how the Swarm
        evolves.
        :return: None.
        """
        # Only step an state if it has not cloned and is not frozen
        # if self.keep_best:
        # self._will_step[-1] = False
        self.dt = self.custom_skipframe(self)
        observs = [self.observations[i] for i, valid in enumerate(self._not_frozen) if valid]
        actions = self._model.predict_batch(observs)

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
        ob_i = 0
        for i, valid in enumerate(self._not_frozen):
            if valid:
                obs = observs[ob_i]
                ob_i += 1
            else:
                obs = self.observations[i]
            self.observations[i] = obs
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

        self._n_samples_done += self._not_frozen.sum()


class Net:
    def __init__(self, epochs, batch_size):
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=batch_size, shuffle=True, num_workers=1
        )
        self.trainloaderiter = iter(self.trainloader)

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=batch_size, shuffle=False, num_workers=1
        )
        self.testloaderiter = iter(self.testloader)
        self.xs, self.ys = self.get_next_batch()

        self.parameters_descriptions = []
        self.parameters_descriptions.append((6, 3, 5, 5))
        self.parameters_descriptions.append((16, 6, 5, 5))
        self.parameters_descriptions.append((120, 16 * 5 * 5))
        self.parameters_descriptions.append((84, 120))
        self.parameters_descriptions.append((10, 84))

        self.pool = nn.MaxPool2d(2, 2)

        self.epochs = epochs
        self.batch_size = batch_size
        self.cpt_epoch = 0
        self.end = False

    def reset_gen(self, is_train=True):
        if is_train:
            self.trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=1
            )
            self.trainloaderiter = iter(self.trainloader)
        else:
            self.testloaderiter = iter(self.testloader)

    def parameters_description(self):
        return self.parameters_descriptions

    def get_next_batch(self, is_train=True):
        if is_train:
            gen = self.trainloaderiter
            try:
                return next(gen)
            except StopIteration as si:
                self.reset_gen(is_train)
                gen = self.trainloaderiter
                return next(gen)
        else:
            gen = self.testloaderiter
            try:
                return next(gen)
            except StopIteration as si:
                return None

    @staticmethod
    def save_params(parameters, path_parameters=None):
        if path_parameters is None:
            path_parameters = "../parameters/parameters.pkl"
            save_obj(parameters, path_parameters)

    @staticmethod
    def load_params(path_parameters=None):
        if path_parameters is None:
            path_parameters = "../parameters/parameters.pkl"
        return load_obj(path_parameters)

    def forward(self, x, parameters):
        x = self.pool(F.tanh(F.conv2d(x, parameters[0])))
        x = self.pool(F.tanh(F.conv2d(x, parameters[1])))

        x = x.view(-1, 16 * 5 * 5)
        x = F.tanh(F.linear(x, parameters[2]))
        x = F.tanh(F.linear(x, parameters[3]))
        x = F.linear(x, parameters[4])

        return x

    def get_state(self):
        return self.xs, self.ys

    def set_state(self, xs, ys):
        self.xs = xs
        self.ys = ys
        return None

    def step(self, parameters, is_train=True, change_data=False):
        data = self.get_next_batch(is_train=is_train)
        if data is None:
            return None
        self.xs, self.ys = data
        self.xs = self.xs.cuda()
        self.ys = self.ys.cuda()
        outputs = self.forward(*(self.xs, parameters))
        outputs_values, outputs_labels = outputs.max(1)
        acc = self.ys == outputs_labels
        acc = acc.sum().item()
        acc /= float(len(self.ys))
        return acc


class NN_env(Environment):
    def __init__(self, neural_net, step_size, epochs, batch_size):
        self.action_space = act_space()  # space of one seed
        self.observation_space = None  # No observation in fact
        self.reward_range = (0, 1)
        self.metadata = None  # No Need
        self.neural_net = nn_callable(neural_net, epochs, batch_size)()
        self.state_seeds = [np.random.randint(4294967295)]
        self.swarm_seeds = []
        self.step_size = step_size
        self.description = self.neural_net.parameters_description()
        self.param_state = []
        self.empty_parameters = []
        self.init_empty_tensor()
        self.param_state = self.init_from_seeds(self.state_seeds, self.param_state)

    def unwrapped(self):
        return self

    def clone_full_state(self):
        return clone_state()

    def clone_state(self):
        return [self.state_seeds, self.swarm_seeds]

    def restore_full_state(self, state):
        return restore_state(state)

    def restore_state(self, state):
        self.state_seeds, self.swarm_seeds = state
        self.param_state = self.init_from_seeds(self.state_seeds, self.param_state)
        return

    def step(self, seed, change_data=False) -> tuple:
        # x_batch, y_batch, end = self.neural_net.next_batch()
        if change_data:
            self.neural_net.evaluate_on_test(seed)
        else:
            self.swarm_seeds += [seed]
            params = self.walk_from_seeds(self.swarm_seeds, self.param_clone(self.param_state))
        reward = self.neural_net.step(params, is_train=True, change_data=change_data)
        return self.swarm_seeds, reward, self.neural_net.end, None

    def evaluate_on_test(self, seed):
        params = self.walk_from_seeds(seed, self.param_state)
        r_acc = 0
        cpt = 0
        while True:
            r = self.neural_net.step(
                self.empty_parameters1, is_train=False, change_data=change_data
            )
            if r is None:
                break
            r_acc += r
            cpt += 1
        print("========> accuracy on test is ", r_acc / cpt)
        self.neural_net.save_params(params)

    def init_from_seeds(self, seeds, p1) -> np.ndarray:
        # From seeds, reconstruct parameters of NN
        torch.manual_seed(seeds[0])
        for i in range(len(p1)):
            torch.normal(mean=torch.zeros_like(p1[i]), std=0.1, out=p1[i])
        for seed in seeds[1:]:
            torch.manual_seed(seed)
            for i in range(len(p1)):
                p1[i] += torch.normal(
                    mean=torch.zeros_like(self.empty_parameters[i]),
                    std=self.step_size,
                    out=self.empty_parameters[i],
                )
        return p1

    def walk_from_seeds(self, seeds, p1) -> np.ndarray:
        # From seeds, reconstruct parameters of NN
        for seed in seeds:
            torch.manual_seed(seed)
            for i in range(len(p1)):
                p1[i] += torch.normal(
                    mean=torch.zeros_like(self.empty_parameters[i]),
                    std=self.step_size,
                    out=self.empty_parameters[i],
                )
        return p1

    def init_empty_tensor(self):
        for tup in self.description:
            tmp1 = torch.empty(tup)
            tmp1 = tmp1.cuda()
            tmp2 = torch.empty(tup)
            tmp2 = tmp2.cuda()
            self.empty_parameters.append(tmp1)
            self.param_state.append(tmp2)

    def param_clone(self, p):
        param = []
        for i in range(len(p)):
            param.append(p[i].clone())
        return param

    def reset(self):
        self.state_seeds = [np.random.randint(4294967295)]
        return self.swarm_seeds

    def distance_from_seeds(self, obs, idx):
        dists = np.zeros(len(obs))
        for i in range(len(obs)):
            params1 = self.walk_from_seeds(obs[i], self.param_clone(self.param_state))
            params2 = self.walk_from_seeds(obs[idx[i]], self.param_clone(self.param_state))
            d = 0
            for j in range(len(params1)):
                d += torch.dist(params1[j], params2[j])
            dists[i] = d.cpu() / len(params1)
        return dists

    def render(self):  # no need implementation but just to raise the error
        raise NotImplementedError
