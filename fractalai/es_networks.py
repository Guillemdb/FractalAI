import numpy as np
from keras.models import Model, Input, Sequential
from keras.layers import Dense, Conv2D, BatchNormalization, Flatten, Dropout, Concatenate
from keras.regularizers import l1
from keras.optimizers import Adam  # not important as there's no training here.


class RunningStat(object):
    def __init__(self, shape, eps):
        self.sum = np.zeros(shape, dtype=np.float32)
        self.sumsq = np.full(shape, eps, dtype=np.float32)
        self.count = eps

    def increment(self, s, ssq, c):
        self.sum += s
        self.sumsq += ssq
        self.count += c

    @property
    def mean(self):
        return self.sum / self.count

    @property
    def std(self):
        return np.sqrt(np.maximum(self.sumsq / self.count - np.square(self.mean), 1e-2))

    def set_from_init(self, init_mean, init_std, init_count):
        self.sum[:] = init_mean * init_count
        self.sumsq[:] = (np.square(init_mean) + np.square(init_std)) * init_count
        self.count = init_count


class SingleLayerModel:

    def __init__(self, state_shape: tuple, action_shape: tuple,
                 shared_neurons=512, actor_neurons=512, encoder_neurons=512,
                 optimizer="adam", es_all=True, batchnorm=False,
                 dropout=False, stats=None):
        self.stats = stats

        self.batchnorm = batchnorm
        self.dropout = dropout
        self.es_all = es_all

        self.n_shared_neurons = shared_neurons
        self.n_actor_neurons = actor_neurons
        self.n_encoder_neurons = encoder_neurons
        self.optimizer = optimizer
        self.state_input = None
        self.model = None
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.shared_layer = self.create_shared_layers()
        self.actor = self.create_actor_layer()
        self.create_model()

    def predict_action(self, obs):
        if self.stats:
            obs = np.clip((obs - self.stats.mean / self.stats.std), -5, 5)
            s, ssq = obs.sum(), np.square(obs).sum()
            self.stats.increment(s, ssq, 1)
        return self.model.predict(obs.reshape(1, -1))[0]

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def create_model(self):

        self.model = Model(inputs=[self.state_input],
                           outputs=[self.actor])

        self.model.compile(optimizer="adam",
                           loss="mse",
                           )

    def create_actor_layer(self):

        self.actor = Dense(self.action_shape[0], activation='linear',
                           name="actor_out", trainable=False)(self.shared_layer)

        return self.actor

    def create_shared_layers(self):

        self.state_input = Input(shape=self.state_shape, name="state_input")

        self.shared_layer = Dense(self.n_shared_neurons, activation='tanh', name="shared_layer",
                                  kernel_regularizer=l1(0.01))(self.state_input)

        if self.dropout:
            self.shared_layer = Dropout(0.15)(self.shared_layer)
        if self.batchnorm:
            self.shared_layer = BatchNormalization()(self.shared_layer)
        self.shared_layer = Dense(self.n_shared_neurons, activation='tanh',
                                  name="shared_layer_out",
                                  kernel_regularizer=l1(0.01))(self.shared_layer)
        if self.dropout:
            self.shared_layer = Dropout(0.15)(self.shared_layer)
        if self.batchnorm:
            self.shared_layer = BatchNormalization()(self.shared_layer)
        return self.shared_layer


class DHModel:

    def __init__(self, state_shape: tuple, action_shape: tuple, dyn_neurons=512, encode_size=256,
                 reward_neurons=256, shared_neurons=512, actor_neurons=512, encoder_neurons=512,
                 optimizer="adam", es_all=True, batchnorm=False, dropout=False):
        self.batchnorm = batchnorm
        self.dropout = dropout
        self.es_all = es_all
        self.n_dyn_neurons = dyn_neurons
        self.n_reward_neurons = reward_neurons
        self.n_shared_neurons = shared_neurons
        self.n_actor_neurons = actor_neurons
        self.n_encoder_neurons = encoder_neurons
        self.optimizer = optimizer
        self.state_input = None
        self.model = None
        self.action_input = None
        self.shared_action_input = None
        self.state_shape = state_shape
        self.action_shape = action_shape
        self.encode_size = encode_size
        self.shared_layer = self.create_shared_layers()
        self.encoder = self.create_encoder_layers()
        self.reward = self.create_reward_layer()
        self.dynamics = self.create_dynamics_layer()
        self.actor = self.create_actor_layer()
        self.create_model()

    def create_actor_layer(self):

        decode_1 = Dense(self.n_actor_neurons, activation='relu', name="actor_hidden",
                         trainable=False)(self.shared_layer)
        if self.batchnorm:
            decode_1 = BatchNormalization()(decode_1)

        decode_out = Dense(self.action_shape[0], activation='sigmoid',
                           name="actor_out", trainable=False)(decode_1)

        return decode_out

    def create_shared_layers(self):

        self.state_input = Input(shape=self.state_shape, name="state_input")

        shared_1 = Dense(self.n_shared_neurons, activation='relu', name="shared_hidden",
                         kernel_regularizer=l1(0.01))(self.state_input)
        if self.dropout:
            shared_1 = Dropout(0.15)(shared_1)
        if self.batchnorm:
            shared_1 = BatchNormalization()(shared_1)

        self.shared_layer = Dense(self.encode_size, activation='relu', name="shared_layer",
                                  kernel_regularizer=l1(0.01))(shared_1)
        self.shared_layer = Flatten()(self.shared_layer)
        if self.dropout:
            self.shared_layer = Dropout(0.15)(self.shared_layer)
        self.shared_layer = BatchNormalization()(self.shared_layer)
        return self.shared_layer

    def create_encoder_layers(self):
        decode_1 = Dense(self.n_encoder_neurons, activation='relu', name="decode_hidden",
                         kernel_regularizer=l1(0.01))(self.shared_layer)
        if self.batchnorm:
            decode_1 = BatchNormalization()(decode_1)
        if self.dropout:
            decode_1 = Dropout(0.15)(decode_1)
        decode_out = Dense(int(np.prod(self.state_shape)), activation='linear',
                           name="decode_out")(decode_1)
        return decode_out

    def create_reward_layer(self):
        self.action_input = Input(shape=self.action_shape, name="action_input")
        self.shared_action_input = Concatenate(name="shared_action_in")([self.action_input,
                                                                         self.shared_layer])
        reward_1 = Dense(self.n_reward_neurons, activation='relu',
                         name="reward_hidden")(self.shared_action_input)
        if self.batchnorm:
            reward_1 = BatchNormalization()(reward_1)
        if self.dropout:
            reward_1 = Dropout(0.15)(reward_1)
        decode_out = Dense(1, activation='linear', name="reward_out")(reward_1)
        return decode_out

    def create_dynamics_layer(self):
        dynamics = Dense(self.n_dyn_neurons, activation='relu', name="dynamics_hidden",
                         kernel_regularizer=l1(0.01))(self.shared_action_input)
        if self.batchnorm:
            dynamics = BatchNormalization()(dynamics)
        if self.dropout:
            dynamics = Dropout(self.dropout)(dynamics)
        dynamics_out = Dense(int(np.prod(self.state_shape)), activation='linear',
                             name="dynamics_out")(dynamics)
        return dynamics_out

    def create_model(self):
        self.model = Model(inputs=[self.state_input, self.action_input],
                           outputs=[self.actor, self.encoder, self.dynamics, self.reward])

        self.model.compile(optimizer=self.optimizer,
                           loss={'dynamics_out': 'mse',
                                 'decode_out': 'mse',
                                 "actor_out": "mse",
                                 "reward_out": "mse"},
                           loss_weights={'dynamics_out': 1., 'decode_out': 1.,
                                         "actor_out": 0., "reward_out": 0.})

    def get_weights(self):

        weights = []
        if self.es_all:
            weights += self.model.get_layer("shared_hidden").get_weights()
            weights += self.model.get_layer("shared_layer").get_weights()

        weights += self.model.get_layer("actor_hidden").get_weights()
        weights += self.model.get_layer("actor_out").get_weights()
        return weights

    def set_weights(self, weights):

        if self.es_all:
            self.model.get_layer("shared_hidden").set_weights(weights[:2])
            self.model.get_layer("shared_layer").set_weights(weights[2:4])

        self.model.get_layer("actor_hidden").set_weights(weights[-4:-2])
        self.model.get_layer("actor_out").set_weights(weights[-2:])

    def predict_action(self, obs):
        return self.model.predict({"state_input": obs.reshape((1,) + self.state_shape),
                                   "action_input": np.ones(
                                       self.action_shape[0]).reshape(1, -1)})[0][0]

    def predict(self, obs, action, *args, **kwargs):
        return self.model.predict({"state_input": obs,
                                   "action_input": action}, *args, **kwargs)

    def train(self, memory, batch_size=128, data_size=1000, *args, **kwargs):
        experiences = memory.sample(batch_size=data_size)
        state_in = np.array([exp.state0 for exp in experiences])
        action_in = np.array([exp.action for exp in experiences])
        reward_out = np.array([exp.reward for exp in experiences])
        encoder_out = np.array([exp.state0.flatten() for exp in experiences])
        dinamics_out = np.array([exp.state1.flatten() for exp in experiences])
        dummy_out = np.array([np.ones(self.action_shape[0]) for _ in experiences])

        self.model.fit({"state_input": state_in, "action_input": action_in},
                       {'dynamics_out': dinamics_out,
                        'decode_out': encoder_out,
                        "actor_out": dummy_out,
                        "reward_out": reward_out,
                        },
                       batch_size=batch_size, *args, **kwargs)

