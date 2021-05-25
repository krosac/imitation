"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""
import argparse
import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3

from imitation.algorithms import adversarial, bc
from imitation.data import rollout, types
from imitation.util import logger, util

from stable_baselines3 import A2C, PPO

# Command line arguments
args = argparse.ArgumentParser(description='imitation learning with bc/gail/airl')
args.add_argument('--env',metavar='ENV', type=str, nargs='?', default='Pendulum-v0', help='gym environment name')
args.add_argument('--nenv',metavar='NENV', type=int, nargs='?', default=4, help='number of vector environments')
args.add_argument('--expert',metavar='EXPERT', type=str, nargs='?', default='A2C', help='expert RL algorithm')
args.add_argument('--policy',metavar='POLICY', type=str, nargs='?', default='MlpPolicy', help='policy adopted by expert RL algorithm (MlpPolicy, CnnPolicy, etc.)')
args.add_argument('--expert_timestamps',metavar='EXPERT_ST', type=int, nargs='?', default=3e6, help='timestamps to train the expert')
args.add_argument('--expert_trajs_gen',metavar='EXPERT_TRAJS_GEN', type=int, nargs='?', default=10, help='expert trajectories for imitation learning')
args.add_argument('--il_algo',metavar='IL_ALGO', type=str, nargs='?', default='airl', help='imitation learning algorithm')
args.add_argument('--batch_size',metavar='BS', type=int, nargs='?', default=256, help='expert batch size used in training of imitation learning algorithm')
args.add_argument('--il_rounds',metavar='IL_ROUND', type=int, nargs='?', default=2000, help='training rounds for adversairal training')
args.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='verbose in both expert/imitation training')
args = args.parse_args()

verbose = 1 if args.verbose else 0

# Train expert
env_name = args.env
venv = util.make_vec_env(env_name, n_envs=args.nenv)
if args.expert == 'A2C':
    model = A2C(args.policy, venv, verbose=verbose, tensorboard_log='./'+args.expert+'_'+env_name+'_tensorboard/')
else:
    assert 0, 'Unrecognized expert name!'
model.learn(total_timesteps=args.expert_timestamps, tb_log_name="run")

# Generate trajectories for imitation learning
num_trajs = args.expert_trajs_gen
n_episodes_eval = num_trajs
sample_until_eval = rollout.min_episodes(n_episodes_eval)
expert_trajs = rollout.generate_trajectories(
    model, venv, sample_until=sample_until_eval
)
types.save(args.expert+'_'+env_name+'.pkl', expert_trajs)

'''
# Load pickled test demonstrations.
with open("tests/data/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)
# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)
'''
transitions = rollout.flatten_trajectories(expert_trajs)


# Learn from expert trajectories
tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

if args.il_algo == 'bc':
    # Train BC on expert data.
    # BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
    # dictionaries containing observations and actions.
    logger.configure(tempdir_path / "BC/")
    bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
    bc_trainer.train(n_epochs=1)
elif args.il_algo == 'gail':
    # Train GAIL on expert data.
    # GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
    # iterates over dictionaries containing observations, actions, and next_observations.
    logger.configure(tempdir_path / "GAIL/")
    gail_trainer = adversarial.GAIL(
        venv,
        expert_data=transitions,
        expert_batch_size=args.batch_size,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=verbose, n_steps=1024),
        init_tensorboard=True,
        init_tensorboard_graph=True
    )
    gail_trainer.train(total_timesteps=2048*args.il_rounds)
elif args.il_algo == 'airl':
    # Train AIRL on expert data.
    logger.configure(tempdir_path / "AIRL/")
    airl_trainer = adversarial.AIRL(
        venv,
        expert_data=transitions,
        expert_batch_size=args.batch_size,
        gen_algo=sb3.PPO("MlpPolicy", venv, verbose=verbose, n_steps=1024),
        init_tensorboard=True,
        init_tensorboard_graph=True
    )
    airl_trainer.train(total_timesteps=2048*args.il_rounds)
else:
    assert 0, 'Unrecognized imitation learning algorithm! (valid options: bc, gail, airl)'
