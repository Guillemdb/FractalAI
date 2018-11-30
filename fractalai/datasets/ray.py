from typing import Callable
import os
import numpy as np
import random
import string
import ray
from fractalai.datasets.data_generator import DataGenerator


def get_uids(folder):
    files = os.listdir(folder)
    uids = set([file.split("_")[0] for file in files])
    return uids


class GameLoader:

    data_keys = ["states", "obs", "rewards", "ends", "infos", "actions"]

    def __init__(self, folder):
        self.folder = folder
        self.uid_generator = self._uid_generator()
        self.game_generator = self._game_generator()

    def load_one_file(self, uid):
        data = tuple([])
        for key in self.data_keys:
            try:
                loaded = np.load(os.path.join(self.folder, "{}_{}.npy".format(uid, key)))
                data = data + tuple([loaded])
            except:  # TODO: FIX to avoid bare except
                print("failed to load {} attr {}".format(uid, key))
                continue
        return data

    def _uid_generator(self):
        uids = get_uids(self.folder)
        for uid in uids:
            yield uid

    def _game_generator(self):
        for uid in self.uid_generator:
            yield self.load_one_file(uid)

    def load_game(self):
        return next(self.game_generator)


@ray.remote
class RemoteDataSource:
    def __init__(
        self,
        swarm_class=None,
        env_callable=None,
        model_callable=None,
        swarm_kwargs=None,
        generator_kwargs=None,
        worker_id: int = 0,
        data_env_callable: Callable = None,
        seed: int = 1,
        folder=None,
        mode: str = "online",
    ):
        self.folder = folder
        self.mode = mode
        self.worker_id = worker_id
        self.data_env = None
        swarm = swarm_class(env=env_callable(), model=model_callable(), **swarm_kwargs)
        swarm.seed(seed)
        if "load" in self.mode:
            assert folder is not None, "A folder to load data must be specified"
            self.generator = GameLoader(self.folder)
        else:
            self.data_env = (
                data_env_callable() if data_env_callable is not None else env_callable()
            )
            self.generator = DataGenerator(swarm=swarm, **generator_kwargs)

    def get_game_examples(self, with_id: bool = False):
        best_game = self.generator.best_game_generator()
        if with_id:
            return next(best_game), self.worker_id
        else:
            return next(best_game)

    def get_game_states(self, with_id: bool = False):
        best_game = self.generator.game_state_generator()
        if with_id:
            return next(best_game), self.worker_id
        else:
            return next(best_game)

    def recover_game(self, with_id: bool):
        if "load" not in self.mode:
            states, observs, rewards, ends, infos, actions = self.get_data_game()
            if "save" in self.mode:
                self.save_game(states, observs, rewards, ends, infos, actions)
            if with_id:
                return (states, observs, rewards, ends, infos, actions), self.worker_id
            else:
                return states, observs, rewards, ends, infos, actions
        else:
            game = self.generator.load_game()
            if with_id:
                return game, self.worker_id
            else:
                return game

    def get_data_game(self):
        game = self.get_game_states(False)
        observs = []
        actions = []
        rewards = []
        ends = []
        infos = []
        states = []
        for state, action, dt in zip(*game):
            for i in range(dt):
                states.append(state)
                state, obs, reward, end, info = self.data_env.step(action, state=state)
                observs.append(obs)
                actions.append(action)
                ends.append(end)
                infos.append(info)
                rewards.append(reward)
                if end:
                    self.data_env.reset()

        if len(ends) == 0:
            return self.get_data_game()
        print("Generated {} examples with reward {}".format(len(ends), sum(rewards)))
        ends[-1] = True
        return states, observs, rewards, ends, infos, actions

    def save_game(
        self, states, observs, rewards, ends, infos, actions, folder: str = None, uid: str = None
    ):
        def new_id(size=6, chars=string.ascii_uppercase + string.digits):
            return "".join(random.choice(chars) for _ in range(size))

        uid = uid if uid is not None else new_id()
        folder = folder if folder is not None else self.folder

        def file_name(suffix):
            return os.path.join(folder, "{}_{}".format(uid, suffix))

        print("Saving to {} with uid {}".format(folder, uid))
        np.save(file_name("states"), states)
        np.save(file_name("obs"), observs)
        np.save(file_name("rewards"), rewards)
        np.save(file_name("ends"), ends)
        np.save(file_name("infos"), infos)
        np.save(file_name("actions"), actions)
        print("Successfully saved")


class ParallelDataGenerator:
    def __init__(
        self,
        n_actors: int,
        swarm_class: Callable,
        env_callable: Callable,
        model_callable: Callable,
        swarm_kwargs: dict,
        generator_kwargs: dict,
        data_env_callable: Callable = None,
        seed: int = 1,
        folder=None,
        mode: str = "online",
    ):
        self.n_actors = n_actors
        self.generators = [
            RemoteDataSource.remote(
                swarm_class,
                env_callable,
                model_callable,
                swarm_kwargs,
                generator_kwargs,
                i,
                data_env_callable,
                seed + i,
                folder,
                mode,
            )
            for i in range(n_actors)
        ]

    def get_games(self):
        compute_ids = [gen.get_game.remote() for gen in self.generators]
        games = ray.get(compute_ids)
        return games

    def game_stream(self, examples: bool = False, full_game: bool = False):
        if examples:
            remaining_ids = [gen.get_game_examples.remote(with_id=True) for gen in self.generators]
        else:
            remaining_ids = [gen.recover_game.remote(True) for gen in self.generators]
        while True:
            ready_ids, remaining_ids = ray.wait(remaining_ids)
            for ready_id in ready_ids:
                game, worker_id = ray.get(ready_id)
                if examples:
                    new_id = self.generators[worker_id].get_game_examples.remote(True)
                else:
                    new_id = self.generators[worker_id].recover_game.remote(True)
                remaining_ids.append(new_id)
                if full_game:
                    yield game
                else:
                    if not examples:
                        _states, observs, rewards, ends, infos, actions = game
                        for i in range(len(actions)):
                            yield _states[i], observs[i], rewards[i], ends[i], infos[i], actions[i]
                    else:
                        _states, obs, actions, rewards, new_obs, ends = game
                        for i in range(len(rewards)):
                            yield obs[i], actions[i], rewards[i], new_obs[i], ends[i]
