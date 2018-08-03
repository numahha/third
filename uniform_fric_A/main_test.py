#!/usr/bin/env python3
# noinspection PyUnresolvedReferences
from mpi4py import MPI
from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines import logger
from baselines.ppo1.mlp_policy import MlpPolicy
import matplotlib.pyplot as plt
import baselines.common.tf_util as U
import numpy as np
import gym, my_env


def sim_episodes(env, pi,
                  sample_num=100,
                  stochastic=True):
    rAll_list=[]
    for i in range(0,sample_num):
        print(i)
        s = env.reset()
        rAll =0
        d = False
        while d==False:
            ac, vpred = pi.act(stochastic, s)
            s,r,d,_ = env.step(ac)
            rAll += r
        rAll_list.append(rAll)

    return rAll_list


def test(args):
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

    env = make_mujoco_env(args.env, workerseed)
    pi = policy_fn("pi", env.observation_space, env.action_space)
    U.load_state(args.indir)


    fric=args.fric_min
    fric_list=[]
    mean_list=[]
    p10_list =[]
    p20_list =[]
    p80_list =[]
    p90_list =[]
    rAll_list = []
    while fric<args.fric_max+1.0e-6:

        print("fric =",fric)
        def temp_friction_dist():
            return fric
        env.env.env.friction_dist = temp_friction_dist
        temp_rAll_list = sim_episodes(env, pi)
        temp_rAll_np = np.array(temp_rAll_list)
        fric_list.append(fric)
        mean_list.append(np.average(temp_rAll_np))
        p10_list.append(np.percentile(temp_rAll_np, 10))
        p20_list.append(np.percentile(temp_rAll_np, 20))
        p80_list.append(np.percentile(temp_rAll_np, 80))
        p90_list.append(np.percentile(temp_rAll_np, 90))
        rAll_list.extend(temp_rAll_list)
        fric+=0.1


    rAll_np = np.array(rAll_list)
    plt.plot(fric_list, mean_list, label='mean')
    plt.plot(fric_list, p10_list, label='10th')
    plt.plot(fric_list, p20_list, label='20th')
    plt.plot(fric_list, p80_list, label='80th')
    plt.plot(fric_list, p90_list, label='90th')
    plt.plot(fric_list, np.ones(len(mean_list))*np.average(rAll_np), label='total_mean')
    plt.plot(fric_list, np.ones(len(mean_list))*np.percentile(rAll_np,10), label='total_10th')
    figtitle = args.indir
    figtitle += ':  Mean = '+('%03.2f' % (np.average(rAll_np)) )
    figtitle += ':  Std = '+('%03.2f' % (np.std(rAll_np)) )
    plt.title(figtitle)
    plt.ylabel('Episode Reward')
    plt.xlabel('friction')
    plt.legend()
    plt.savefig(args.outdir+"performance.eps")
    plt.close()


    env.close()

def main():
    parser = mujoco_arg_parser()
    parser.add_argument("--outdir", type=str, default="./outdir/")
    parser.add_argument("--indir", type=str, default=None)
    parser.add_argument("--fric_min", type=float, default=1.0)
    parser.add_argument("--fric_max", type=float, default=1.0)
    args = parser.parse_args()
    test(args)

if __name__ == '__main__':
    main()

