from .environment import Environment, OpenAIEnvironment, AtariEnvironment, \
                         DMControlEnv
from .fractalmc import FractalAI
from .model import RandomDiscreteModel, RandomPongModel, \
                   RandomContinuousModel, RandomMomentumModel, \
                   ContinuousDiscretizedModel
from .monitor import AtariMonitor, AtariMonitorPolicy
from .policy import Policy, GreedyPolicy, PolicyWrapper
from .state import State, MicrostateCore, Microstate, AtariState


def run_demo():
    environment = AtariEnvironment(name='MsPacman-v0',
                                   clone_seeds=True)
    model = RandomDiscreteModel(environment.num_actions)

    agent = FractalAI(policy=GreedyPolicy(env=environment,
                                          model=model),
                      max_samples=300,
                      max_states=15,
                      time_horizon=10,
                      n_fixed_steps=5)

    # TODO fix ambiguous naming
    # policy is not GreedyPolicy or similar, but FractalAI
    AtariMonitorPolicy(policy=agent,
                       directory='videos',
                       force=True).run(skip_frames=80)
