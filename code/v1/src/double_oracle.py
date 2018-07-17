#!/usr/bin/env python3

from scipy import optimize as op
from listutils import *
from model import Model
from test import test_model, test_attack_action, test_defense_action, test_defense_newest
from listutils import *
from ddpg import DefendMixedAttack, AttackMixedDefense

import numpy as np
import random
import logging
import pickle

"""Implementation of double-oracle algorithms."""

MAX_EPISODES = 500
MAX_STEPS = 50
GAMMA = 0.9
MAX_ITERATION = 300
INITIAL_ACTION_SIZE = 10

def find_mixed_NE(payoff):
    """
    Function for returning mixed strategies of the first step of double oracle iterations.
    :param payoff: Two dimensinal array. Payoff matrix of the players. The row is defender and column is attcker. 
    :return: List, mixed strategy of the attacker and defender at NE by solving maxmini problem. 
    """ 
    # This implementation is based on page 88 of the book multiagent systems (Shoham etc.)
    n_action = payoff.shape[0]
    c = np.zeros(n_action)
    c = np.append(c, 1)
    A_ub = np.concatenate((payoff, np.full((n_action, 1), -1)), axis=1)
    b_ub = np.zeros(n_action)
    A_eq = np.full(n_action, 1)
    A_eq = np.append(A_eq, 0)
    A_eq = np.expand_dims(A_eq, axis=0)
    b_eq = np.array([1])
    bound = ()
    for i in range(n_action):
        bound += ((0, None),)
    bound += ((None, None),)
    res_attacker = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound)
    c = -c
    A_ub = np.concatenate((-payoff.T, np.full((n_action, 1), 1)), axis=1)
    res_defender = op.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=bound)
    return list(res_attacker.x[0:n_action]), list(res_defender.x[0:n_action]), res_attacker.fun

def get_payoff(model, states, attack_policy, defense_policy):
    """
    Function for computing the payoff of the defender given its strategy and the strategy of the attacker. 
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param states: a random collection of the states used as the initial state of each episode
    :param attack_policy: Function, takes a model and a state, returns the portion of budget allocated for each type of attacks given a model and a state.
    :param defense_policy: Function, takes a model and a state, returns the portion of budget allocated for each type of alerts with all ages given a model and a state.
    :return: The expected discounted reward. 
    """
    total_discount_reward = 0  
    for i in range(MAX_EPISODES):
        state = random.choice(states)
        episode_reward = 0.0
        for j in range(MAX_STEPS):
            next_state = model.next_state(state, defense_policy, attack_policy)
            loss = next_state.U - state.U
            state = next_state
            step_reward = -1.0*loss
            episode_reward += GAMMA**j*step_reward
        #print(i, episode_reward)
        total_discount_reward += episode_reward
    ave_discount_reward = total_discount_reward/MAX_EPISODES
    return ave_discount_reward

def get_payoff_mixed(model, states, attack_profile, defense_profile, attack_strategy, defense_strategy):
    """
    Function for computing the payoff of the defender given its mixed strategy and the mixed strategy of the attacker. 
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param states: a random collection of the states used as the initial state of each episode
    :param attack_profile: List of attack policies.ks given a model and a state.
    :param defense_profile: List of defense policies.    
    :param attack_strategy: List of probablities of choosing policy from the attack profile 
    :param defense_strategy: List of probablities of choosing policy from the defense profile 
    :return: The expected discounted reward. 
    """
    total_discount_reward = 0
    attack_policies = np.random.choice(attack_profile, MAX_EPISODES, p=attack_strategy)
    defense_policies = np.random.choice(defense_profile, MAX_EPISODES, p=defense_strategy)  
    for i in range(MAX_EPISODES):
        state = random.choice(states)
        episode_reward = 0.0
        defense_policy = defense_policies[i]
        attack_policy = attack_policies[i]
        for j in range(MAX_STEPS):
            next_state = model.next_state(state, defense_policy, attack_policy)
            loss = next_state.U - state.U
            state = next_state
            step_reward = -1.0*loss
            episode_reward += GAMMA**j*step_reward
        #print(i, episode_reward)
        total_discount_reward += episode_reward
    ave_discount_reward = total_discount_reward/MAX_EPISODES
    return ave_discount_reward

def update_profile(model, states, payoff, attack_profile, 
        defense_profile, attack_policy, defense_policy):
    """
    Function for updating the payoff matrix and the action profile of defender and attacker
    :param states: a random collection of the states used as the initial state of each episode
    :param model: Model of the alert prioritization problem (i.e., Model object).
    :param payoff: Two dimensinal array. Payoff matrix of the players. The row is defender and column is attcker. 
    :param attack_profile: List of attack policies.
    :param defense_profile: List of defense policies.
    :param attack_policy: New pure strategy of the attacker
    :param defense_policy: New pure strategy og the defender
    :return: updated payoff matrix, attack_profile and defense_profile
    """
    n_action = payoff.shape[0]

    # A new row and column will be added to the payoff matrix    
    new_payoff_col = np.array([])
    new_payoff_row = np.array([])

    # First get the new column    
    for i in range(len(defense_profile)):
        new_payoff = get_payoff(model, states, attack_policy, defense_profile[i])
        new_payoff_col = np.append(new_payoff_col, new_payoff)
    new_payoff_col = np.expand_dims(new_payoff_col, axis=0)
    attack_profile.append(attack_policy)    
    payoff = np.concatenate((payoff, new_payoff_col.T), axis=1)

    # Second, get the new row
    for j in range(len(attack_profile)):
        new_payoff = get_payoff(model, states, attack_profile[j], defense_policy)
        new_payoff_row = np.append(new_payoff_row, new_payoff)
    new_payoff_row = np.expand_dims(new_payoff_row, axis=0)
    defense_profile.append(defense_policy)
    payoff = np.concatenate((payoff, new_payoff_row), axis=0)
    return payoff, attack_profile, defense_profile

def test_mixed_NE():
    """
    Test find_mixed_NE with the Matching Pennies game and Rock-Paper-Scissors on page 58 of Multiagent Systems.
    The mixed strategy at NE should be 0.5 for each player
    """
    #payoff = np.array([[1,-1],[-1,1]]) # Matching Pennies
    payoff = np.array([[0,-1,1],[1,0,-1],[-1,1,0]]) # Rock, Paper, Scissors
    attack_strategy, defense_strategy, utility = find_mixed_NE(payoff)
    print(attack_strategy, defense_strategy)

def test_get_payoff():
    # First generate random states for future use
    model = test_model()
    initial_state = Model.State(model)
    states = []
    global_state = initial_state
    states.append(global_state)
    for i in range(10*MAX_EPISODES):
        next_global_state = model.next_state(global_state, test_defense_action, test_attack_action)
        states.append(next_global_state)
        global_state = next_global_state
    # Second, compute the expected discount reward as the payoff
    test_payoff = get_payoff(model, states, test_attack_action, test_defense_action)    

def test_payoff_mixed():
    # Generate random states for future use
    model = test_model()
    initial_state = Model.State(model)
    states = []
    global_state = initial_state
    states.append(global_state)
    for i in range(10*MAX_EPISODES):
        next_global_state = model.next_state(global_state, test_defense_newest, test_attack_action)
        states.append(next_global_state)
        global_state = next_global_state

    for i in range(10):
        print(get_payoff(model, states, test_attack_action, test_defense_newest))
        print(get_payoff_mixed(model, states, [test_attack_action], [test_defense_newest], [1.0], [1.0]))    

def double_oracle():
    # Generate random states for future use
    model = test_model()
    initial_state = Model.State(model)
    states = []
    global_state = initial_state
    states.append(global_state)
    for i in range(10*MAX_EPISODES):
        next_global_state = model.next_state(global_state, test_defense_newest, test_attack_action)
        states.append(next_global_state)
        global_state = next_global_state    

    # Initialize random action profiles
    print("Initializing...")
    attack_profile = []
    defense_profile = []
    for i in range(INITIAL_ACTION_SIZE):
        attack_action = np.random.uniform(0, 1, size=len(model.attack_types))
        defense_action = np.random.uniform(0, 1, size=model.horizon*len(model.alert_types))
        defense_action = defense_action/sum(defense_action)
        def attack_initial_policy(model, state):
            action = list(attack_action)
            alpha = model.make_attack_feasible(action)
            return alpha
        def defense_initial_policy(model, state):
            action = list(defense_action)
            delta = model.make_investigation_feasible(state.N, unflatten_list(action, len(model.alert_types)))
            return delta
        attack_profile.append(attack_initial_policy)
        defense_profile.append(defense_initial_policy)

    # Initialize the payoff matrix
    payoff = np.zeros((INITIAL_ACTION_SIZE, INITIAL_ACTION_SIZE))
    for i in range(INITIAL_ACTION_SIZE):
        for j in range(INITIAL_ACTION_SIZE):
            payoff[i][j] = get_payoff(model, states, attack_profile[j], defense_profile[i])
    payoff_record = []

    # Compute new strategies and actions
    for i in range(MAX_ITERATION):
        attack_strategy, defense_strategy, utility = find_mixed_NE(payoff)
        payoff_record.append(utility)

        print("###########################################################################################")
        print("Iteration", i)
        print("The utility of the defender (solved by LP) from the start to current iteration:")
        print(payoff_record)

        utility_defender_newest = get_payoff_mixed(model, states, attack_profile, [test_defense_newest], attack_strategy, [1.0])
        print("The utility when defender deviates from NE and uniformly distributes the budget among newest alerts:")
        print(utility_defender_newest)

        utility_attacker_uniform = get_payoff_mixed(model, states, [test_attack_action], defense_profile, [1.0], defense_strategy)
        print("The utility when attacker deviates by uniformly distributes the budget among all attack types. ")
        print(utility_attacker_uniform)

        # Get new response to the mixed strategy
        attack_response = AttackMixedDefense(model, defense_profile, defense_strategy)
        defense_response = DefendMixedAttack(model, attack_profile, attack_strategy)
        
        attack_policy = attack_response.agent.policy
        defense_policy = defense_response.agent.policy

        # Test the terminate condition
        attack_pure_utility = -1*get_payoff_mixed(model, states, [attack_policy], defense_profile, [1.0], defense_strategy)
        defense_pure_utility = get_payoff_mixed(model, states, attack_profile, [defense_policy], attack_strategy, [1.0])
        print("The utility of pure attack:", attack_pure_utility)
        print("The utility of pure defense:", defense_pure_utility)
        if -1*utility >= attack_pure_utility and utility >= defense_pure_utility:
            print("No pure strategy found, terminate")
            break

        # Update profile    
        payoff, attack_profile, defense_profile = update_profile(model, states, payoff, attack_profile, defense_profile, attack_policy, defense_policy)

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s / %(levelname)s: %(message)s', level=logging.DEBUG)
    #test_mixed_solution()
    #test_get_payoff()
    #test_payoff_mixed()
    double_oracle()