from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from mpi4py import MPI
from collections import deque
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from contextlib import contextmanager
import datetime

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred = pi.act(stochastic, ob)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def epi_generator(pi, env, stochastic):
    # Initialize state variables
    ac = env.action_space.sample()
    new = True
    ob = env.reset()

    obs = []
    rews = []
    vpreds = []
    news = []
    acs = []
    prevacs = []
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        vpreds.append(vpred)
        news.append(new)
        acs.append(ac)
        prevacs.append(prevac)
        ob, rew, new, _ = env.step(ac)
        rews.append(rew)
        if new:
            break

    obs = np.array(obs)
    rews = np.array(rews, 'float32')
    vpreds = np.array(vpreds, 'float32')
    news = np.array(news, 'int32')
    acs = np.array(acs)
    prevacs = np.array(prevacs)
    _, vpred = pi.act(stochastic, ob)
    nextvpred = vpred * (1 - new)
    episode = dict(
                obs=obs,
                rews=rews,
                vpreds=vpreds,
                news=news,
                acs=acs,
                prevacs=prevacs,
                nextvpred=nextvpred
                )
    return episode

def traj_set_generator(pi, env, ep_batch_size, epsilon, stochastic):
    rank=MPI.COMM_WORLD.Get_rank()
    while True:
        epi_set = []
        all_ep_rets = []
        for i in range(0,ep_batch_size):
            epi_set.append(epi_generator(pi, env, stochastic))
            all_ep_rets.append(sum(epi_set[i]["rews"]))
            #print(rank,i,all_ep_rets[i])
        # TODO: modify to use MPI appropriately
        #epi_set=MPI.COMM_WORLD.gather(epi_set,root=0)
        #all_ep_rets=MPI.COMM_WORLD.gather(all_ep_rets,root=0)
        #if rank == 0:
        print("epsilon is ",epsilon)
        Q_epsilon = np.percentile(all_ep_rets, epsilon)
        idx = [i for i,ret in enumerate(all_ep_rets) if Q_epsilon>=ret ]
        #MPI.COMM_WORLD.bcast(idx, root=0)
        #print("Q_epsilon,idx")
        #print(rank,Q_epsilon,idx)
        #MPI.COMM_WORLD.barrier()
        count=0
        for i in idx:
            if count==0:
                obs, rews, vpreds, news, acs, prevacs = epi_set[i]["obs"], epi_set[i]["rews"], epi_set[i]["vpreds"], epi_set[i]["news"], epi_set[i]["acs"], epi_set[i]["prevacs"]
                ep_rets = [all_ep_rets[i]]
                ep_lens = [len(epi_set[i]["rews"])]
            else:
                obs = np.concatenate([obs, epi_set[i]["obs"]], axis=0)
                rews = np.concatenate([rews, epi_set[i]["rews"]], axis=0)
                vpreds = np.concatenate([vpreds, epi_set[i]["vpreds"]], axis=0)
                news = np.concatenate([news, epi_set[i]["news"]], axis=0)
                acs = np.concatenate([acs, epi_set[i]["acs"]], axis=0)
                prevacs = np.concatenate([prevacs, epi_set[i]["prevacs"]], axis=0)
                ep_rets.append(all_ep_rets[i])
                ep_lens.append(len(epi_set[i]["rews"]))
            count+=1
            nextvpred = epi_set[i]["nextvpred"]
        yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": nextvpred,
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}

def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_fn, *,
        timesteps_per_batch=0, # what to train on
        episodes_per_batch=0,
        max_kl, cg_iters,
        gamma, lam, # advantage estimation
        entcoeff=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_timesteps=0, max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        input_dir=None,
        trj_percentile=100
        ):
    nworkers = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    entbonus = entcoeff * meanent

    vferr = tf.reduce_mean(tf.square(pi.vpred - ret))

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]

    dist = meankl

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    vfadam = MpiAdam(vf_var_list)

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)
    klgrads = tf.gradients(dist, var_list)
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    fvp = U.flatgrad(gvp, var_list)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg], losses)
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
        return out

    U.initialize()
    th_init = get_flat()
    MPI.COMM_WORLD.Bcast(th_init, root=0)
    set_from_flat(th_init)
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)
    if input_dir is not None:
        U.load_state(input_dir)
    # Prepare for rollouts
    # ----------------------------------------
    if False is ( (timesteps_per_batch==0) ^ (episodes_per_batch==0) ):
        print("Specify either timesteps or episodes per batch!")
        print(timesteps_per_batch)
        print(episodes_per_batch)
        exit(1)
    if timesteps_per_batch:
        seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)
    else:
        # you can use CVaR objective using trj_percentile < 100
        seg_gen = traj_set_generator(pi, env, episodes_per_batch, trj_percentile, stochastic=True)


    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0])==1

    logger.log(datetime.datetime.now())

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate

        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        args = seg["ob"], seg["ac"], atarg
        fvpargs = [arr[::5] for arr in args]
        def fisher_vector_product(p):
            return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

        assign_old_eq_new() # set old parameter values to new parameter values
        with timed("computegrad"):
            *lossbefore, g = compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
        if np.allclose(g, 0):
            logger.log("Got zero gradient. not updating")
        else:
            with timed("cg"):
                stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
            assert np.isfinite(stepdir).all()
            shs = .5*stepdir.dot(fisher_vector_product(stepdir))
            lm = np.sqrt(shs / max_kl)
            # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
            fullstep = stepdir / lm
            expectedimprove = g.dot(fullstep)
            surrbefore = lossbefore[0]
            stepsize = 1.0
            thbefore = get_flat()
            for _ in range(10):
                thnew = thbefore + fullstep * stepsize
                set_from_flat(thnew)
                meanlosses = surr, kl, *_ = allmean(np.array(compute_losses(*args)))
                improve = surr - surrbefore
                logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                if not np.isfinite(meanlosses).all():
                    logger.log("Got non-finite value of losses -- bad!")
                elif kl > max_kl * 1.5:
                    logger.log("violated KL constraint. shrinking step.")
                elif improve < 0:
                    logger.log("surrogate didn't improve. shrinking step.")
                else:
                    logger.log("Stepsize OK!")
                    break
                stepsize *= .5
            else:
                logger.log("couldn't compute a good step")
                set_from_flat(thbefore)
            if nworkers > 1 and iters_so_far % 20 == 0:
                paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
                assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):

            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()
    logger.log(datetime.datetime.now())
    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
