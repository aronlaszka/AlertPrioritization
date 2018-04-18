#!/usr/bin/env python3

"""Reinforcement-learning based best-response policies."""
import logging
from numpy import array, float32
from random import Random
from time import time

from model import Model
from test import test_model, test_attack_action

from utils import ReplayBuffer
from utils import ActorNetwork
from utils import CriticNetwork


import itertools
import numpy as np   
import tensorflow as tf


BUFFER_SIZE = 10000
BATCH_SIZE = 32
GAMMA = 0.99
TAU = 0.001     # target Network HyperParameters
LRA = 0.01#0.0001    # learning rate for Actor
LRC = 0.01#0.001     # lerning rate for Critic

np.random.seed(133)
EXPLORE = 1000
episode_count = 2#2000 # number of episodes
max_steps = 1000#10000 # self.steps in each episode
reward = 0

#Tensorflow GPU optimization
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
from keras import backend as K
K.set_session(sess)

def normalized(vect):
    """
    Normalize the given vector.
    :param vect: Vector represented as a list of floats.
    :return: Normalized vector represented as a list of floats.
    """
    factor = 1 / sum(vect)
    return [element * factor for element in vect]

class DDPGAgent:
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
        self.actor = ActorNetwork(sess, self.state_size, self.action_size, BATCH_SIZE, TAU, LRA)
        self.critic = CriticNetwork(sess, self.state_size, self.action_size, BATCH_SIZE, TAU, LRC)
        self.buff = ReplayBuffer(BUFFER_SIZE)    
        self.step = 0
        self.epsilon = 1
        
    def learn(self, initial_state, state_observe, state_update, rnd=Random(0)):
        """
        Q-learning based algorithm for learning the best actions in every state.
        Note that due to performance reasons, the state-value function (i.e., Q) is not updated after every self.step, but only in batches.
        :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
        :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
        :param state_update: Updates the state based on an action. Function, takes a state (see state_observe) and an action (normalized list of floats), return the next state (may be arbitrary object).
        :param rnd: Random number generator.
        """
        logging.info("DDPG algorithm starts.")
        for i in range(episode_count):
            logging.info("Episode: %s Replay Buffer: %s", str(i), str(self.buff.count()))
            state_t = initial_state 
            s_t = np.array([state_observe(initial_state)]) # initial observation state
            total_reward = 0.

            for j in range(max_steps):
                loss = 0 
                self.epsilon -= 1.0 / EXPLORE 
                a_t = np.zeros([1, self.action_size])
                noise_t = np.zeros([1, self.action_size]) # a noise process to select action
                
                # selct action according to the current policy and exploration noise
                a_t_original = self.actor.model.predict(s_t)
                for k in range(self.action_size):
                    noise_t[0][k] = max(self.epsilon, 0)*np.random.normal(0, 0.1, 1)[0] # the noise follows Gaussian distribution
                    a_t[0][k] = a_t_original[0][k]+noise_t[0][k]
                    if a_t[0][k] < 0:
                        a_t[0][k] = 0.05 # before normalization, we should make sure a_t[0][k] >= 0
                # normalize the action
                sum_action = sum(a_t[0])
                for k in range(self.action_size):
                    a_t[0][k] = a_t[0][k]*1.0/sum_action    
                #print(j, a_t[0])
                (state_t1, loss_t) = state_update(state_t, list(a_t[0]))
                r_t = -1*loss_t # reward of the defender is -1*its loss 
                s_t1 = np.array([state_observe(state_t1)])

                self.buff.add(s_t[0], a_t[0], r_t, s_t1[0]) # add replay buffer

                #Do the batch update
                batch = self.buff.getBatch(BATCH_SIZE)
                states = np.asarray([e[0] for e in batch])
                actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_states = np.asarray([e[3] for e in batch])
                y_t = np.asarray([[0] for e in batch])



                target_q_values = self.critic.target_model.predict([new_states, self.actor.target_model.predict(new_states)]) 

                for k in range(len(batch)):
                    y_t[k] = rewards[k] + GAMMA*target_q_values[k]

                loss += self.critic.model.train_on_batch([states,actions], y_t) 
                a_for_grad = self.actor.model.predict(states)
                grads = self.critic.gradients(states, a_for_grad)
                self.actor.train(states, grads)
                self.actor.target_train()
                self.critic.target_train()

                total_reward += r_t
                s_t = s_t1
                state_t = state_t1

                self.step += 1

            logging.info("TOTAL REWARD @ %s-th Episode  : Reward %s", str(i), str(total_reward))
            logging.info("Total self.step: %s", str(self.step))
            logging.info("")

        logging.info("Finish.")

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
        agent = DDPGAgent(len(model.alert_types) * model.horizon, len(model.alert_types) * model.horizon)
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
