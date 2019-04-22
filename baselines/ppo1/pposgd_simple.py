from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque
from multiprocessing import Process, Queue, current_process
import IPython
import os
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def traj_segment_generator(pi, env, horizon, stochastic):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

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

######################### Save model / Jie Xu ##########################
import os
def play_one_round(pi, env):
    ac = env.action_space.sample() # not used, just so we have the datatype
    ob = env.reset()
    rewards = 0
    itr = 0
            
    # obs = []
    while True:
        prevac = ac
        ac, vpred = pi.act(False, ob)
        # ac, vpred = pi.act(True, ob)
        ob, rew, done, _ = env.step(ac)
        # obs.append(ob)

        rewards += rew
        itr += 1

        if done:
            print("rewards:", rewards)
            # pi.ob_rms.update(np.array(obs))

            return rewards
########################################################################

def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
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

######################### Jie Xu ##################3##
def build_graph_only(env, policy_fn,*,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    sess = U.get_session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    return pi

def get_policy_parameters(pi):
    # output for runningmeanst
    mean, std = pi.get_mean_std()

    with U.get_session().as_default() as sess:
        with tf.variable_scope('pol'):
            # W0, b0
            x = pi.policy_tensor0
            weights_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0')
            bias_tensor = bias_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/bias:0')
            W0, b0 = sess.run([weights_tensor, bias_tensor])

            # W_hidden, b_hidden
            W_hidden = []
            b_hidden = []
            for i in range(len(pi.policy_tensor_hidden)):
                x = pi.policy_tensor_hidden[i]
                weights_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0')
                bias_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/bias:0')
                W, b = sess.run([weights_tensor, bias_tensor])
                W_hidden.append(W)
                b_hidden.append(b)

            # W1, b1
            x = pi.policy_tensor1
            weights_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0')
            bias_tensor = bias_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/bias:0')
            W1, b1 = sess.run([weights_tensor, bias_tensor])

    return W0, b0, W_hidden, b_hidden, W1, b1, mean, std

def get_mlp_shared_policy_parameters(pi):
    # output for runningmeanst
    mean, std = pi.get_mean_std()

    with U.get_session().as_default() as sess:
        with tf.variable_scope('pol'):
            # W0, b0
            x = pi.policy_tensor0
            weights_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0')
            bias_tensor = bias_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/bias:0')
            W0, b0 = sess.run([weights_tensor, bias_tensor])

            # W_hidden, b_hidden
            W_hidden_shared = []
            b_hidden_shared = []
            for x in pi.policy_tensor_hidden_shared:
                weights_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/kernel:0')
                bias_tensor = tf.get_default_graph().get_tensor_by_name(os.path.split(x.name)[0] + '/bias:0')
                W, b = sess.run([weights_tensor, bias_tensor])
                W_hidden_shared.append(W)
                b_hidden_shared.append(b)

    return W0, b0, W_hidden_shared, b_hidden_shared

#######################################################

def learn(env, play_env, policy_fn, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        ######################### Save model / Jie Xu ##########################
        model_directory,
        save_model_interval,
        save_model_with_prefix, # Save the model with this prefix after save_model_interval iters
        restore_model_from_file,# Load the states/model from this file.
        ########################################################################
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        play=False
        ):
    sess = U.get_session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    ######################### Save model / Jie Xu ##########################
    #with U.get_session().as_default() as sess:
    #    writer = tf.summary.FileWriter("/home/eanswer/Projects/ReinforcementLearning/tensorflow_tutorial")
    #    writer.add_graph(tf.get_default_graph())
    # Resume model if a model file is provided
    if restore_model_from_file:
        saver=tf.train.Saver()
        saver.restore(tf.get_default_session(), os.path.join(model_directory, restore_model_from_file))
        logger.log("Loaded model from {}".format(os.path.join(model_directory, restore_model_from_file)))
        print("load")
    ########################################################################
    if play and restore_model_from_file:
        ######################### Jie Xu ############################
        r = []
        for times in range(100):
            r.append(play_one_round(pi, play_env))
            if times % 10 == 0:
                print('running average [', times, ']:', np.mean(r))
        #############################################################
    else:
        # Prepare for rollouts
        # ----------------------------------------
        #seg_gen = traj_segment_generator_parallel(pi, env, timesteps_per_actorbatch, stochastic=True)
        seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)
        
        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        #assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

        ################# Record training results / Jie Xu #####################
        best_rew = 0.0
        training_rewards_file = os.path.join(model_directory, "rewards.txt")
        fp = open(training_rewards_file, "w")
        fp.close()
        ########################################################################
        
        while True:
            ################# play trained model / Jie Xu #####################
            if iters_so_far % 50 == 0:
                play_one_round(pi, play_env)
            ###################################################################

            if callback: callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************"%iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_"+name, lossval)
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
            if MPI.COMM_WORLD.Get_rank()==0:
                logger.dump_tabular()
            
            ######################### Save model / Jie Xu ##########################
            if iters_so_far % save_model_interval == 0:
                if save_model_with_prefix:
                    saver = tf.train.Saver()
                    with U.get_session().as_default() as sess:
                        modelF= os.path.join(model_directory, save_model_with_prefix+"_afterIter_"+str(iters_so_far)+".ckpt")
                        save_path = saver.save(sess, modelF)
                        logger.log("Saved model to file :{}".format(modelF))
            ########################################################################

            ######################### Save Best model / Jie Xu ##########################
            best_rew *= 0.999
            if np.mean(rewbuffer) > best_rew:
                best_rew = np.mean(rewbuffer)
                saver = tf.train.Saver()
                with U.get_session().as_default() as sess:
                    modelF= os.path.join(model_directory, "best_model.ckpt")
                    save_path = saver.save(sess, modelF)
                    logger.log("Saved best model to file :{}".format(modelF))
            ########################################################################

            ################# Record training results / Jie Xu #####################
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                fp = open(training_rewards_file, "a")
                fp.write("%f %f\n" % (np.mean(rewbuffer), np.mean(lenbuffer)))
                fp.close()
            ########################################################################

        ######################### Save model / Jie Xu ##########################
        if save_model_with_prefix:
            saver = tf.train.Saver()
            with U.get_session().as_default() as sess:
                modelF= os.path.join(model_directory, save_model_with_prefix+"_final.ckpt")
                save_path = saver.save(sess, modelF)
                logger.log("Saved model to file :{}".format(modelF))
        ########################################################################
    
    return pi

###########################################
def learn_shared(env, play_env, policy_fn, *,
          timesteps_per_actorbatch, # timesteps per actor per update
          clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
          optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
          gamma, lam, # advantage estimation
          ######################### Save model / Jie Xu ##########################
          model_directory,
          save_model_interval,
          save_model_with_prefix, # Save the model with this prefix after save_model_interval iters
          restore_model_from_file,# Load the states/model from this file.
          ########################################################################
          max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
          callback=None, # you can do anything in the callback, since it takes locals(), globals()
          adam_epsilon=1e-5,
          schedule='constant', # annealing for stepsize parameters (epsilon and adam)
          play=False
          ):
    sess = U.get_session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_fn("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
                                                   for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    ######################### Save model / Jie Xu ##########################
    #with U.get_session().as_default() as sess:
    #    writer = tf.summary.FileWriter("/home/eanswer/Projects/ReinforcementLearning/tensorflow_tutorial")
    #    writer.add_graph(tf.get_default_graph())
    # Resume model if a model file is provided
    if restore_model_from_file:
        saver=tf.train.Saver()
        saver.restore(tf.get_default_session(), model_directory+restore_model_from_file)
        logger.log("Loaded model from {}".format(model_directory+restore_model_from_file))
        print("load")
    ########################################################################
    if play and restore_model_from_file:
        ######################### Jie Xu ############################
        r = []
        for times in range(100):
            r.append(play_one_round(pi, play_env))
            if times % 10 == 0:
                print('running average [', times, ']:', np.mean(r))
                #############################################################
    else:
        # Prepare for rollouts
        # ----------------------------------------
        #seg_gen = traj_segment_generator_parallel(pi, env, timesteps_per_actorbatch, stochastic=True)
        seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

        episodes_so_far = 0
        timesteps_so_far = 0
        iters_so_far = 0
        tstart = time.time()
        lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
        rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

        assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

        ################# Record training results / Jie Xu #####################
        training_rewards_file = os.path.join(model_directory, "rewards.txt")
        fp = open(training_rewards_file, "w")
        fp.close()
        ########################################################################

        while True:
            ################# play trained model / Jie Xu #####################
            if iters_so_far % 5 == 0:
                play_one_round(pi, play_env)
            ###################################################################

            if callback: callback(locals(), globals())
            if max_timesteps and timesteps_so_far >= max_timesteps:
                break
            elif max_episodes and episodes_so_far >= max_episodes:
                break
            elif max_iters and iters_so_far >= max_iters:
                break
            elif max_seconds and time.time() - tstart >= max_seconds:
                break

            if schedule == 'constant':
                cur_lrmult = 1.0
            elif schedule == 'linear':
                cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
            else:
                raise NotImplementedError

            logger.log("********** Iteration %i ************"%iters_so_far)

            seg = seg_gen.__next__()
            add_vtarg_and_adv(seg, gamma, lam)

            # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
            ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
            vpredbefore = seg["vpred"] # predicted value function before udpate
            atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
            d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
            optim_batchsize = optim_batchsize or ob.shape[0]

            if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

            assign_old_eq_new() # set old parameter values to new parameter values
            logger.log("Optimizing...")
            logger.log(fmt_row(13, loss_names))
            # Here we do a bunch of optimization epochs over the data
            for _ in range(optim_epochs):
                losses = [] # list of tuples, each of which gives the loss for a minibatch
                for batch in d.iterate_once(optim_batchsize):
                    *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                    adam.update(g, optim_stepsize * cur_lrmult)
                    losses.append(newlosses)
                logger.log(fmt_row(13, np.mean(losses, axis=0)))

            logger.log("Evaluating losses...")
            losses = []
            for batch in d.iterate_once(optim_batchsize):
                newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                losses.append(newlosses)
            meanlosses,_,_ = mpi_moments(losses, axis=0)
            logger.log(fmt_row(13, meanlosses))
            for (lossval, name) in zipsame(meanlosses, loss_names):
                logger.record_tabular("loss_"+name, lossval)
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
            if MPI.COMM_WORLD.Get_rank()==0:
                logger.dump_tabular()

            ######################### Save model / Jie Xu ##########################
            if iters_so_far % save_model_interval == 0:
                if save_model_with_prefix:
                    saver = tf.train.Saver()
                    with U.get_session().as_default() as sess:
                        modelF= model_directory+save_model_with_prefix+"_afterIter_"+str(iters_so_far)+".ckpt"
                        save_path = saver.save(sess, modelF)
                        logger.log("Saved model to file :{}".format(modelF))
            ########################################################################

            ################# Record training results / Jie Xu #####################
            if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
                fp = open(training_rewards_file, "a")
                fp.write("%f %f\n" % (np.mean(rewbuffer), np.mean(lenbuffer)))
                fp.close()
            ########################################################################

        ######################### Save model / Jie Xu ##########################
        if save_model_with_prefix:
            saver = tf.train.Saver()
            with U.get_session().as_default() as sess:
                modelF= model_directory+save_model_with_prefix+"_final.ckpt"
                save_path = saver.save(sess, modelF)
                logger.log("Saved model to file :{}".format(modelF))
                ########################################################################

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
