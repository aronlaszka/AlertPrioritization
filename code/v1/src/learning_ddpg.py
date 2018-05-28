#!/usr/bin/env python3

"""Reinforcement-learning based best-response policies."""
import logging
from numpy import array, float32
from random import Random
from time import time

from model import Model
from test import test_model, test_attack_action

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

MAX_EPISODES = 25
MAX_EP_STEPS = 1000
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = int(1.0*MEMORY_CAPACITY) # only 30% data of the memory are seleted as mini-batch

RENDER = False


TEST_EPISODES = 300
MAX_TEST_STEPS = 1000


def normalized(vect):
    """
    Normalize the given vector.
    :param vect: Vector represented as a list of floats.
    :return: Normalized vector represented as a list of floats.
    """
    factor = 1 / sum(vect)
    return [element * factor for element in vect]

def normal(action):
    sum_action = 0
    for i in range(action.shape[1]):
        sum_action += action[0][i]
    for i in range(action.shape[1]):
        action[0][i] = action[0][i]/sum_action
    return action

def make_action_feasible(action):
    for i in range(action.shape[0]):
        if action[i] < 0:
            action[i] = 0.000001
    action = action / sum(action)
    return action

class DDPG(object):
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
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            h1 = tf.layers.dense(s, 64, activation=tf.nn.tanh, name='l1', trainable=trainable)
            h2 = tf.layers.dense(h1, 32, activation=tf.nn.tanh, name='l2', trainable=trainable)
            #h3 = tf.layers.dense(h2, 32, activation=tf.nn.tanh, name='l3', trainable=trainable)
            a = tf.layers.dense(h2, self.a_dim, activation=tf.nn.softmax, name='a', trainable=trainable)
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
            #net3 = tf.layers.dense(net2, 32, activation=tf.nn.relu, trainable=trainable)
            return tf.layers.dense(net2, 1, trainable=trainable)  # Q(s,a)

class DDPGlearning:
    """
    Learning algorithm inspired by Q-learning. 
    States are represented as lists of arbitrary floats, while actions are represented as normalized lists of floats (i.e., floats are greater than or equal to zero and sum up to one).
    Differences compared to Q-learning are described in the documentation of the relevant functions.
    """

    def __init__(self, state_size, action_size):
        """
        Construct a learning algorithm object.
        :param state_size: Length of lists representing states.
        :param action_size: Length of lists representing actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.ddpg = DDPG(action_size, state_size)

    def learn(self, initial_state, state_observe, state_update, rnd=Random(0)):
        """
        Q-learning based algorithm for learning the best actions in every state.
        Note that due to performance reasons, the state-value function (i.e., Q) is not updated after every self.step, but only in batches.
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        :param state_update: Updates the state based on an action. Function, takes a state (see state_observe) and an action (normalized list of floats), return the next state (may be arbitrary object).
        :param rnd: Random number generator.
        """
        logging.info("DDPG training starts.")

        # generate random states for state initilization in each eposide        
        states = []
        global_state = initial_state
        states.append(global_state)
        for i in range(MEMORY_CAPACITY):
            action = normalized([rnd.random() for i in range(self.action_size)])
            (next_global_state, loss) = state_update(global_state, action)
            states.append(next_global_state)
            global_state = next_global_state

        # ddpg training process
        epsilon = 0.0
        epsilon_max = 1.0
        t1 = time.time()
        rewards = []
        mode_list = ["ddpg", "uniform_all", "uniform_youngest"]
        mode = "ddpg"
        if mode == "ddpg":
            for i in range(MAX_EPISODES):
                global_state = random.choice(states)
                state = np.array(state_observe(global_state),dtype=np.float32)
                total_reward = 0.0
                exploit_reward = []
                exploit_flag = True
                for j in range(MAX_EP_STEPS):
                    # epsilon-greedy
                    if np.random.uniform() < epsilon:
                        exploit_flag = True
                        action = self.ddpg.choose_action(state) #action has been normalized by choos_action()
                    else:
                        exploit_flag = False
                        action = normalized([rnd.random() for i in range(self.action_size)])
                        action = np.array(action)
                    
                    (next_global_state, loss) = state_update(global_state, list(action))
                    next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                    reward = -1.0*loss
                    self.ddpg.store_transition(state, action, reward, next_state)

                    if self.ddpg.pointer > MEMORY_CAPACITY and (j+1)%20 == 0:
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
                        if len(exploit_reward) == 0:
                            logging.info("Episode {}, Ave reward {}, Ave exploitation reward unavailable".format(i, ave_reward))
                        else:
                            logging.info("Episode {}, Ave reward {}, Ave exploitation reward {}".format(i, ave_reward, sum(exploit_reward)/len(exploit_reward)))
                        rewards.append(ave_reward)
                        epsilon = epsilon+0.05 if epsilon < epsilon_max else epsilon_max
                        break
         
            pickle.dump(rewards, open('pomdp.pickle','wb'))  
        logging.info("DDPG test starts.")
        total_reward = 0            
        for i in range(TEST_EPISODES):
            global_state = random.choice(states)
            state = np.array(state_observe(global_state),dtype=np.float32)
            episode_reward = 0.0
            for j in range(MAX_TEST_STEPS):
                # Choose the best action by the actor network
                if mode == "ddpg":
                    action = self.ddpg.choose_action(state)
                if mode == "uniform_all":
                    action = np.array([0.16,0.16,0.16,0.16,0.16,0.16], dtype=np.float32)
                if mode == "uniform_youngest":
                    action = np.array([0.0,1.0,0.0,0.0,0.0,0.0], dtype=np.float32)                               
                (next_global_state, loss) = state_update(global_state, list(action))
                next_state = np.array(state_observe(next_global_state), dtype=np.float32)
                global_state = next_global_state
                state = next_state
                step_reward = -1.0*loss
                episode_reward += step_reward
            episode_reward = episode_reward/MAX_TEST_STEPS
            logging.info("Episode {}, Average reward in each step {}".format(i, episode_reward))
            total_reward += episode_reward
        ave_reward = total_reward/TEST_EPISODES
        logging.info("Average reward in each step {}".format(ave_reward))

def flatten_lists(lists):
    """
    Construct a single list from a list of lists.
    :param lists: List of lists.
    :return: Single list that contains all the elements of all the lists, in the same order.
    """  
    return [element for inner in lists for element in inner]
    
def unflatten_list(lst, dim):
    """
    Construct a list of lists from a single list.
    :param lst: List of elements, size must be a multiple of dim.
    :param dim: Number of elements in each inner list.
    :return: List of lists that contain all the elements of the list.
    """
    ##print(len(lst))
    ##print(dim)
    assert((len(lst) % dim) == 0)
    lists = []
    for i in range(len(lst) // dim):
        lists.append([lst[j] for j in range(i * dim, (i + 1) * dim)])
    return lists
    
class DefenderBestResponse:
    """Best-response investigation policy for the defender."""
    def __init__(self, model, alpha):
        """
        Construct a best-response object using QLearning.
        :param model: Model of the alert prioritization problem (i.e., Model object).
        :param alpha: Attack policy. Function, takes a model and a state, returns the probability of mounting attacks (one-dimensional list) given a model and a state.
        """
        agent = DDPGlearning(len(model.alert_types) * model.horizon, len(model.alert_types) * model.horizon)
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
        agent.learn(Model.State(model),
                        lambda state: flatten_lists(state.N),
                        state_update)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s / %(levelname)s: %(message)s', level=logging.DEBUG)
    model = test_model()
    DefenderBestResponse(model, test_attack_action)
