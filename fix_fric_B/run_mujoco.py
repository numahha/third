#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
from baselines.trpo_mpi import trpo_mpi
import gym, my_env

def train(args):
    import baselines.common.tf_util as U
    sess = U.single_threaded_session()
    sess.__enter__()

    rank = MPI.COMM_WORLD.Get_rank()
    if rank == 0:
        logger.configure(args.outdir)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)
    workerseed = args.seed + 10000 * MPI.COMM_WORLD.Get_rank()

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    def temp_friction_dist():
        return args.fric_fix

    env = make_mujoco_env(args.env, workerseed)
    env.env.env.friction_dist = temp_friction_dist
    trpo_mpi.learn(env, policy_fn, timesteps_per_batch=5000, max_kl=0.01, cg_iters=20, cg_damping=0.1,
        max_timesteps=args.num_timesteps, gamma=0.995, lam=0.97, vf_iters=5, vf_stepsize=1e-3)
    env.close()

def main():
    parser = mujoco_arg_parser()
    parser.add_argument("--outdir", type=str, default="./logs/")
    parser.add_argument("--fric_fix", type=float, default=1.0)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()

