from baselines.common.mpi_running_mean_std import RunningMeanStd
import baselines.common.tf_util as U
import tensorflow as tf
import gym
from baselines.common.distributions import make_pdtype
import numpy as np

class MlpPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            # init function for variable getting
            self._init_get_functions()

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            # obz = ob
            last_out = obz
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]

        with tf.variable_scope('pol'):
            last_out = obz
            self.policy_tensor0 = None
            self.policy_tensor_hidden = []
            self.policy_tensor1 = None
            for i in range(num_hid_layers):
                dense = tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0))
                # last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                last_out = tf.nn.tanh(dense)
                if i == 0:
                    self.policy_tensor0 = dense
                else:
                    self.policy_tensor_hidden.append(dense)

            if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                mean = tf.layers.dense(last_out, pdtype.param_shape()[0]//2, name='final', kernel_initializer=U.normc_initializer(0.01))
                self.policy_tensor1 = mean
                logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
                pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
            else:
                pdparam = tf.layers.dense(last_out, pdtype.param_shape()[0], name='final', kernel_initializer=U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])
    
    # init function for variable getting
    def _init_get_functions(self):
        self.get_mean_std = U.function([], [self.ob_rms.mean, self.ob_rms.std])
        
    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []

class MlpSharedPolicy(object):
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name
            # init function for variable getting
            self._init_get_functions()

    def _init(self, ob_space, ac_space, ac_dof, hid_size, num_hid_layers, gaussian_fixed_var=True):
        # We require the observation space to be a tuple of space.Box of the same type.
        assert isinstance(ob_space, gym.spaces.Box)
        n = ob_space.shape[0]
        # Check ac_space.
        assert isinstance(ac_space, gym.spaces.Box)
        self.pdtype = pdtype = make_pdtype(ac_space)

        sequence_length = None
        # ob[None, model, ob].
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        with tf.variable_scope('vf'):
            obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
            last_out = tf.reshape(obz, shape=[-1] + list(ob_space.shape[1:]))
            # last_out = [data0, model0, ob]
            #            [data0, model1, ob]
            #            [data0, model2, ob]
            #            [data1, model0, ob]
            #            [data1, model1, ob]
            #            [data1, model2, ob]
            #            [data2, model0, ob]
            #            [data2, model1, ob]
            #            [data2, model2, ob]
            for i in range(num_hid_layers):
                last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name="fc%i"%(i+1), kernel_initializer=U.normc_initializer(1.0)))
            self.vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:,0]
            self.vpred = tf.reduce_sum(tf.reshape(self.vpred, shape=[-1, n]), axis=1)

        with tf.variable_scope('pol'):
            swap_idx = np.arange(len(obz.shape))
            swap_idx[0:2] = [1, 0]
            last_out = tf.transpose(obz, swap_idx)
            last_out = tf.reshape(last_out, shape=[-1] + list(ob_space.shape[1:]))
            # last_out = [data0, model0, ob]
            #            [data1, model0, ob]
            #            [data2, model0, ob]
            #            [data0, model1, ob]
            #            [data1, model1, ob]
            #            [data2, model1, ob]
            #            [data0, model2, ob]
            #            [data1, model2, ob]
            #            [data2, model2, ob]
            self.policy_tensor0 = None
            self.policy_tensor_hidden = []
            self.policy_tensor1 = None
            for i in range(num_hid_layers):
                dense = tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0))
                # last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i'%(i+1), kernel_initializer=U.normc_initializer(1.0)))
                last_out = tf.nn.tanh(dense)
                if i == 0:
                    self.policy_tensor0 = dense
                else:
                    self.policy_tensor_hidden.append(dense)

            last_outs = tf.split(last_out, num_or_size_splits=n, axis=0)
            pd_means = []
            pd_vars = []
            self.policy_tensor1 = []
            for i, last_out, dof in zip(range(n), last_outs, ac_dof):
                if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
                    mean = tf.layers.dense(last_out, dof, name='final%i'%(i), kernel_initializer=U.normc_initializer(0.01))
                    self.policy_tensor1.append(mean)
                    logstd = tf.get_variable(name="logstd%i"%(i), shape=[1, dof], initializer=tf.zeros_initializer())
                    pd_means.append(mean)
                    pd_vars.append(mean * 0.0 + logstd)
                else:
                    pdparam = tf.layers.dense(last_out, dof*2, name='final%i'%(i), kernel_initializer=U.normc_initializer(0.01))
                    mean, var = tf.split(pdparam, num_or_size_splits=2, axis=len(pdparam.shape)-1)
                    pd_means.append(mean)
                    pd_vars.append(var)
            pdparams = tf.concat(pd_means + pd_vars, axis=1)
            assert len(pdparams.shape) == 2

        self.pd = pdtype.pdfromflat(pdparams)

        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    # init function for variable getting
    def _init_get_functions(self):
        self.get_mean_std = U.function([], [self.ob_rms.mean, self.ob_rms.std])
    def act(self, stochastic, ob):
        ac1, vpred1 =  self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    def get_initial_state(self):
        return []
