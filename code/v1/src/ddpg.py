#!/usr/bin/env python3

"""Reinforcement-learning based best-response policies."""
import logging
from numpy import array, float32
from random import Random
from time import time

from model import Model
from test import test_model1, test_model2,test_model3, test_model_credit, test_attack_action, test_defense_action, test_defense_newest
from listutils import *

from keras.models import Sequential
from keras.layers import Dense

import itertools
import numpy as np   
import tensorflow as tf
import time
import random
import matplotlib.pyplot as plt
import pickle

#####################  hyper parameters  ####################

MAX_EPISODES = 500#200
MAX_EP_STEPS = 200#1000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.95     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 25000
BATCH_SIZE = 64#int(0.01*MEMORY_CAPACITY) # only 30% data of the memory are seleted as mini-batch
LEARNING_STEP = 1 # steps that ddpg learns
EPSILON_DISCOUNT = 0.995

TEST_EPISODES = 500#150
MAX_TEST_STEPS = 200

attacker_ind = 0
defender_ind = 0

class DDPGbase(object):
    def __init__(self, a_dim, s_dim,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim = a_dim, s_dim,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        self.Qvalue = q
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def Q_value(self, s, a):
        return self.sess.run(self.Qvalue, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        raise NotImplementedError

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        raise NotImplementedError

class DDPGdefend(DDPGbase):
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, 64, activation=tf.nn.tanh, name='l1', trainable=trainable)
            h2 = tf.layers.dense(h1, 32, activation=tf.nn.tanh, name='l2', trainable=trainable)
            #h3 = tf.layers.dense(h2, 64, activation=tf.nn.tanh, name='l3', trainable=trainable)
            a = tf.layers.dense(h2, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net1, 32, activation=tf.nn.relu, trainable=trainable)
            #net3 = tf.layers.dense(net2, 64, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)

class DDPGattack(DDPGbase):
    def _build_a(self, s, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, 64, activation=tf.nn.tanh, name='l1', trainable=trainable)
            h2 = tf.layers.dense(h1, 32, activation=tf.nn.tanh, name='l2', trainable=trainable)
            #h3 = tf.layers.dense(h2, 64, activation=tf.nn.tanh, name='l3', trainable=trainable)
            a = tf.layers.dense(h2, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = 64
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net2 = tf.layers.dense(net1, 32, activation=tf.nn.relu, trainable=trainable)
            #net3 = tf.layers.dense(net2, 64, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)

class DDPGlearning:
    """
    Learning algorithm inspired by Q-learning. 
    States are represented as lists of arbitrary floats, while actions are represented as normalized lists of floats (i.e., floats are greater than or equal to zero and sum up to one).
    Differences compared to Q-learning are described in the documentation of the relevant functions.
    """

    def __init__(self, mode, state_size, action_size):
        """
        Construct a learning algorithm object.
        :param state_size: Length of lists representing states.
        :param action_size: Length of lists representing actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.mode = mode
        if self.mode == "defend":
            self.ddpg = DDPGdefend(action_size, state_size)
        else:
            self.ddpg = DDPGattack(action_size, state_size)

    def buffer_init(self, model, initial_state, state_observe):
        """
        Initialization of the replay buffer
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        """
        global_state = initial_state
        for i in range(MEMORY_CAPACITY):
            # Generate random actions
            attack_random_action = np.random.uniform(0, 1, size=len(model.attack_types))
            defense_random_action = np.random.uniform(0, 1, size=model.horizon*len(model.alert_types))
            defense_random_action = defense_random_action/sum(defense_random_action)
            # Make the random actions feasible
            alpha = model.make_attack_feasible(list(attack_random_action)) # alpha is a list
            delta = model.make_investigation_feasible(global_state.N, unflatten_list(list(defense_random_action), len(model.alert_types))) # delta is a list            
            # State trasition
            state = np.array(state_observe(global_state),dtype=np.float32)
            next_global_state = model.next_state(global_state, delta, alpha)            
            # Set the replay buffer            
            next_state = np.array(state_observe(next_global_state), dtype=np.float32)
            defender_loss = next_global_state.U - global_state.U
            global_state = next_global_state
            if self.mode == "defend":
                self.ddpg.store_transition(state, defense_random_action, -1*defender_loss, next_state)
            else:
                self.ddpg.store_transition(state, attack_random_action, defender_loss, next_state)

    def learn(self, model, initial_state, state_observe, state_update, rnd=Random(0)):
        """
        Q-learning based algorithm for learning the best actions in every state.
        Note that due to performance reasons, the state-value function (i.e., Q) is not updated after every self.step, but only in batches.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        :param state_update: Updates the state based on an action. Function, takes a state (see state_observe) and an action (normalized list of floats), return the next state (may be arbitrary object).
        :param rnd: Random number generator.
        """
        logging.info("DDPG training starts.")

        # Initilize replay buffer
        self.buffer_init(model, initial_state, state_observe)
        
        # DDPG training process
        epsilon = 1.0
        for i in range(MAX_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            total_reward = 0.0
            exploit_reward = []
            exploit_flag = True
            for j in range(MAX_EP_STEPS):
                # epsilon-greedy
                if np.random.uniform() >= epsilon:
                    exploit_flag = True
                    action = self.ddpg.choose_action(state) #action has been normalized by choos_action()
                else:
                    exploit_flag = False
                    if self.mode == "defend":
                        action = normalized([rnd.random() for i in range(self.action_size)])
                    else:
                        action = [rnd.random() for i in range(self.action_size)]
                    action = np.array(action)

                (next_global_state, loss) = state_update(global_state, list(action))
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                reward = -1.0*loss
                self.ddpg.store_transition(state, action, reward, next_state)

                if self.ddpg.pointer > MEMORY_CAPACITY and (j+1) % LEARNING_STEP == 0:
                    self.ddpg.learn()

                global_state = next_global_state
                state = next_state
                total_reward += reward
                if exploit_flag:
                    exploit_reward.append(reward)

                if j == MAX_EP_STEPS-1:
                    ave_reward = total_reward/MAX_EP_STEPS
                    #logging.info("Episode {}, Average reward in each step {}".format(i, ave_reward))
                    if (i+1) % (MAX_EPISODES/10) == 0:
                        if len(exploit_reward) == 0:
                            logging.info("Episode {}, Ave reward {}, Ave exploitation reward unavailable".format(i, ave_reward))
                        else:
                            logging.info("Episode {}, Ave reward {}, Ave exploitation reward {}".format(i, ave_reward, sum(exploit_reward)/len(exploit_reward)))
                    epsilon = epsilon*EPSILON_DISCOUNT
                    break

        # DDPG test
        logging.info("DDPG test starts.")
        total_reward = 0            
        for i in range(TEST_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            episode_reward = 0.0
            for j in range(MAX_TEST_STEPS):
                action = self.ddpg.choose_action(state)
                #action = np.array([0.2,0.2,0.2,0.2,0.2,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)
                (next_global_state, loss) = state_update(global_state, list(action))
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                global_state = next_global_state
                state = next_state
                step_reward = -1.0*loss
                episode_reward += GAMMA**j*step_reward
            #logging.info("Episode {}, Average reward in each step {}".format(i, episode_reward))
            total_reward += episode_reward
        if TEST_EPISODES != 0:
            ave_reward = total_reward/TEST_EPISODES
            logging.info("Expected discount reward {}".format(ave_reward))

        # DDPG test
        logging.info("DDPG test with non-strategic baseline.")
        total_reward = 0            
        for i in range(TEST_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            episode_reward = 0.0
            for j in range(MAX_TEST_STEPS):
                #action = self.ddpg.choose_action(state)
                #action = np.array([1,0,0,0,0,0,0,0], dtype=np.float32)
                if self.mode == "defend":
                    action = np.array([0.2,0.2,0.2,0.2,0.2,0,0,0,0,0,0,0,0,0,0], dtype=np.float32)
                else:
                    action = np.array([0.125,0.125,0.125,0.125,0.125,0.125,0.125,0.125], dtype=np.float32)
                (next_global_state, loss) = state_update(global_state, list(action))
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                global_state = next_global_state
                state = next_state
                step_reward = -1.0*loss
                episode_reward += GAMMA**j*step_reward
            #logging.info("Episode {}, Average reward in each step {}".format(i, episode_reward))
            total_reward += episode_reward
        if TEST_EPISODES != 0:
            ave_reward = total_reward/TEST_EPISODES
            logging.info("Expected discount reward {}".format(ave_reward))

    def learn_from_mix(self, model, initial_state, state_observe, state_update, op_profile, op_strategy, rnd=Random(0)):
        """
        Q-learning based algorithm for learning the best actions in every state against mixed strategy of the opponent.
        Note that due to performance reasons, the state-value function (i.e., Q) is not updated after every self.step, but only in batches.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        :param state_update: Updates the state based on an action. Function, takes a state (see state_observe), an action (normalized list of floats) and oppoent's action sampled from its mixed strategy, return the next state (may be arbitrary object).
        :param op_profile: List, action profile of the opponent.
        :param op_strategy: List, mixed strategy of the opponent.
        :param rnd: Random number generator.
        """
        logging.info("DDPG training starts.")

        # Initialize the replay buffer       
        self.buffer_init(model, initial_state, state_observe)

        # DDPG training process
        epsilon = 1.0
        op_actions = np.random.choice(op_profile, MAX_EPISODES, p=op_strategy)

        for i in range(MAX_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            total_reward = 0.0
            exploit_reward = []
            exploit_flag = True
            
            op_action = op_actions[i]
            
            for j in range(MAX_EP_STEPS):
                # epsilon-greedy
                if np.random.uniform() >= epsilon:
                    exploit_flag = True
                    action = self.ddpg.choose_action(state) #action has been normalized by choos_action()
                else:
                    exploit_flag = False
                    if self.mode == "defend":
                        action = normalized([rnd.random() for i in range(self.action_size)])
                    else:
                        action = [rnd.random() for i in range(self.action_size)]
                    action = np.array(action)

                (next_global_state, loss) = state_update(global_state, list(action), op_action)
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                reward = -1.0*loss
                self.ddpg.store_transition(state, action, reward, next_state)

                if self.ddpg.pointer > MEMORY_CAPACITY and (j+1) % LEARNING_STEP == 0:
                    self.ddpg.learn()

                global_state = next_global_state
                state = next_state
                total_reward += reward
                if exploit_flag:
                    exploit_reward.append(reward)

                if j == MAX_EP_STEPS-1:
                    #print('Episode:', i, ' Reward: %i' % int(total_reward), 'Explore: %.5f' % epsilon, )
                    ave_reward = total_reward/MAX_EP_STEPS
                    #logging.info("Episode {}, Average reward in each step {}".format(i, ave_reward))
                    if (i+1) % (MAX_EPISODES/10) == 0:
                        if len(exploit_reward) == 0:
                            logging.info("Episode {}, Ave reward {}, Ave exploitation reward unavailable".format(i, ave_reward))
                        else:
                            
                            logging.info("Episode {}, Ave reward {}, Ave exploitation reward {}".format(i, ave_reward, sum(exploit_reward)/len(exploit_reward)))
                    epsilon = epsilon*EPSILON_DISCOUNT
                    break                                

        # DDPG test
        logging.info("DDPG test starts.")
        total_reward = 0
        op_actions = np.random.choice(op_profile, MAX_EPISODES, p=op_strategy)            
        for i in range(TEST_EPISODES):
            global_state = initial_state
            state = np.array(state_observe(global_state),dtype=np.float32)
            episode_reward = 0.0
            op_action = op_actions[i]
            for j in range(MAX_TEST_STEPS):
                # Choose the best action by the actor network
                action = self.ddpg.choose_action(state)

                (next_global_state, loss) = state_update(global_state, list(action), op_action)
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                global_state = next_global_state
                state = next_state
                step_reward = -1.0*loss
                episode_reward += GAMMA**j*step_reward
            #logging.info("Episode {}, Average reward in each step {}".format(i, episode_reward))
            total_reward += episode_reward
        if TEST_EPISODES != 0:
            ave_reward = total_reward/TEST_EPISODES
            logging.info("Expected discount reward {}".format(ave_reward))

    def policy(self, model, state):
        """
        Get the action given by the state
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param state: State of the alert prioritization problem (i.e., Model.State object).(one-dimensional list) given a model and a state.
        :retuen: the action based on the policy
        """        
        feasible_action = None
        if self.mode == "defend":
            state_array = np.array(flatten_lists(state.N),dtype=np.float32)
            action = list(self.ddpg.choose_action(state_array))
            delta = model.make_investigation_feasible(state.N, unflatten_list(action, len(model.alert_types)))
            feasible_action = delta 
        else:
            state_array = np.array(flatten_state(state),dtype=np.float32)
            action = list(self.ddpg.choose_action(state_array))
            alpha = model.make_attack_feasible(action)            
            feasible_action = alpha
        return feasible_action

class DefenderBestResponse:
    """Best-response investigation policy for the defender."""
    def __init__(self, model, alpha):
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param alpha: Attack policy. Function, takes a model and a state, returns the probability of mounting attacks (one-dimensional list) given a model and a state.
        """
        self.mode = "defend"
        self.agent = DDPGlearning(self.mode, len(model.alert_types) * model.horizon, len(model.alert_types) * model.horizon)
        def state_update(state, action):
            """
            State update function for QLearning.learn.
            :param state: State of the alert prioritization problem (i.e., Model.State object).
            :param action: Action represented as a normalized list of floats.
            :return: Next state (i.e., Model.State object).
            """
            delta = model.make_investigation_feasible(state.N, unflatten_list(action, len(model.alert_types))) # make_investigation_feasible ``unnormalizes'' the action
            next_state = model.next_state(state, delta, alpha)
            loss = next_state.U - state.U
            return (next_state, loss)
        self.agent.learn(model,
                        Model.State(model),
                        lambda state: flatten_lists(state.N),
                        state_update)
        #pickle.dump(self.agent, open('model/defender-'+str(defender_ind)+'.pickle','wb'))
        #global defender_ind
        #defender_ind += 1
        tf.reset_default_graph()

class DefenderOracle:
    """Best-response investigation policy for the defender against mix strategy of the attacker."""
    def __init__(self, model, attack_profile, attack_strategy):
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param attack_profile: List of attack policies.
        :param attack_strategy: List of probablities of choosing policy from the attack profile 
        """
        self.mode = "defend"
        self.agent = DDPGlearning(self.mode, len(model.alert_types) * model.horizon, len(model.alert_types) * model.horizon)
        def state_update(state, action, alpha):
            """
            State update function for QLearning.learn.
            :param state: State of the alert prioritization problem (i.e., Model.State object).
            :param action: Action represented as a normalized list of floats.
            :param alpha: Attack policy sampled from the attack_profile.
            :return: Next state (i.e., Model.State object).
            """
            delta = model.make_investigation_feasible(state.N, unflatten_list(action, len(model.alert_types))) # make_investigation_feasible ``unnormalizes'' the action
            next_state = model.next_state(state, delta, alpha)
            loss = next_state.U - state.U
            return (next_state, loss)
        self.agent.learn_from_mix(model,
                                 Model.State(model),
                                 lambda state: flatten_lists(state.N),
                                 state_update,
                                 attack_profile,
                                 attack_strategy)
        tf.reset_default_graph()

class AttackerBestResponse:
    """Best-response attack policy for the attacker."""
    def __init__(self, model, delta):   
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param delta: defense policy. Function, takes a model and a state, returns the portion of budget allocated for each type of alerts with all ages given a model and a state.
        """
        self.mode = "attack"
        state_size = model.horizon*(len(model.alert_types) + len(model.attack_types) + len(model.alert_types) * len(model.attack_types))
        action_size = len(model.attack_types)
        self.agent = DDPGlearning(self.mode, state_size, action_size)
        def state_update(state, action):
            """
            State update function for QLearning.learn.
            :param state: State of the alert prioritization problem (i.e., Model.State object).
            :param action: Action represented as a normalized list of floats.
            :return: Next state (i.e., Model.State object).
            """
            alpha = model.make_attack_feasible(action)      
            next_state = model.next_state(state, delta, alpha)
            loss = -1.0 * (next_state.U - state.U)
            return (next_state, loss)
        self.agent.learn(model,
                        Model.State(model),
                        lambda state: flatten_state(state),
                        state_update)
        tf.reset_default_graph()

class AttackerOracle:
    """Best-response attack policy for the attacker against mixed strategy of defender."""
    def __init__(self, model, defense_profile, defense_strategy):
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param defense_profile: List of defense policies.
        :param defense_strategy: List of probablities of choosing policy from the defense profile 
        """       
        self.mode = "attack"
        state_size = model.horizon*(len(model.alert_types) + len(model.attack_types) + len(model.alert_types) * len(model.attack_types))
        action_size = len(model.attack_types)
        self.agent = DDPGlearning(self.mode, state_size, action_size)
        def state_update(state, action, delta):
            """
            State update function for QLearning.learn.
            :param state: State of the alert prioritization problem (i.e., Model.State object).
            :param action: Action represented as a normalized list of floats.
            :param delta: defense policy sampled from the defense_profile.
            :return: Next state (i.e., Model.State object).
            """
            alpha = model.make_attack_feasible(action)      
            next_state = model.next_state(state, delta, alpha)
            loss = -1.0 * (next_state.U - state.U)
            return (next_state, loss)                        
        self.agent.learn_from_mix(model,
                        Model.State(model),
                        lambda state: flatten_state(state),                        
                        state_update,
                        defense_profile,
                        defense_strategy)
        tf.reset_default_graph()            

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s / %(levelname)s: %(message)s', level=logging.DEBUG)
    model = test_model_credit()    
    for i in range(1):
        defender = DefenderBestResponse(model, test_attack_action)
        attacker = AttackerBestResponse(model, test_defense_action)
        attacker = AttackerBestResponse(model, test_defense_newest)
 
