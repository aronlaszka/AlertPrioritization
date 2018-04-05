#!/usr/bin/env python3

"""Reinforcement-learning based best-response policies."""

import logging
from numpy import array, float32
from random import Random
from time import time

from keras.models import Sequential
from keras.layers import Dense

from model import Model
from test import test_model, test_attack_action

EPSILON = 0.000001 # small value (for floating-point imprecision)

def normalized(vect):
  """
  Normalize the given vector.
  :param vect: Vector represented as a list of floats.
  :return: Normalized vector represented as a list of floats.
  """
  factor = 1 / sum(vect)
  return [element * factor for element in vect]

class QLearning:
  """
  Learning algorithm inspired by Q-learning. 
  States are represented as lists of arbitrary floats, while actions are represented as normalized lists of floats (i.e., floats are greater than or equal to zero and sum up to one).
  Differences compared to Q-learning are described in the documentation of the relevant functions.
  """
  BEST_ACTION_ITERATIONS = 10 # small value because querying the neural network is very slow in the current implementation
  LEARN_INITIAL_ITERATIONS = 16384 # 131072
  LEARN_ITERATIONS = 16384 # 131072
  TRAIN_EPOCHS = 5
  DISCOUNT = 0.5
  
  def __init__(self, state_size, action_size):
    """
    Construct a learning algorithm object.
    :param state_size: Length of lists representing states.
    :param action_size: Length of lists representing actions.
    """
    self.state_size = state_size
    self.action_size = action_size
    self.regressor = Sequential()
    self.regressor.add(Dense(64, input_dim=state_size + action_size, kernel_initializer='normal', activation='relu'))
    self.regressor.add(Dense(32, kernel_initializer='normal', activation='relu'))
    self.regressor.add(Dense(1, kernel_initializer='normal'))
    self.regressor.compile(loss='mean_squared_error', optimizer='adam')
    self.regressor.summary()

  def Q(self, state, action):
    """
    Compute the value of an action in a given state.
    :param state: State represented as a list of floats.
    :param action: Action represented as a normalized list of floats.
    :return: Value of action in given state (i.e., Q(s, a)).
    """
    assert(len(state) == self.state_size)
    assert(len(action) == self.action_size)
    assert(abs(sum(action) - 1) < EPSILON)
    assert(min(action) >= 0)
    output = self.regressor.predict(array([state + action]))
    for value in output:
      return value[0]
  
  def Q_batch(self, state_action):
    """
    Compute the value of actions in given states.
    :param state_action: List of tuples (state, action), where state is represented as a list of floats, action is represented as a normalized list of floats.
    :return: List of values of the actions in the given states (i.e., Q(state, action)).
    """
    inputs = []
    for (state, action) in state_action:
      assert(len(state) == self.state_size)
      assert(len(action) == self.action_size)
      assert(abs(sum(action) - 1) < EPSILON)
      assert(min(action) >= 0)
      inputs.append(state + action)
    output = self.regressor.predict(array(inputs))
    values = []
    for value in output:
      values.append(value[0])
    assert(len(state_action) == len(values))
    return values
  
  def best_action(self, state):
    """
    Finds the best action in a given state based on the action-value function (i.e., Q) and a simple hill-climbing heuristic. 
    Note that the value is minimized (instead of maximized).
    :param state: State represented as a list of floats.
    :return: Best action, represented as a normalized list of floats.
    """
    assert(len(state) == self.state_size)
    action = [0.1] * self.action_size
    value = self.Q(state, normalized(action))
    for i in range(QLearning.BEST_ACTION_ITERATIONS):
      for j in range(self.action_size):
        action[j] += 1
        changed = self.Q(state, normalized(action))
        if changed < value:
          value = changed
        else:
          action[j] -= 1
    logging.debug("N: {}, action (unnormalized): {}".format(state, action))
    return normalized(action)
    
  def best_action_batch(self, states):
    """
    Finds the best actions in given states based on the action-value function (i.e., Q) and a simple hill-climbing heuristic. 
    Note that the value is minimized (instead of maximized).
    :param states: List of states represented as lists of floats.
    :return: List of tuples (action, value), where action is the best action, represented as a normalized list of floats, and value is the value of action in state (i.e., Q(state, action)).
    """     
    best_actions = [[1] * self.action_size for s in range(len(states))]
    best_values = [float("inf") for s in range(len(states))]        
    for i in range(QLearning.BEST_ACTION_ITERATIONS):  
      logging.debug("best_action_batch (iteration: {})".format(i)) 
      state_action = []
      changed_actions = []  
      for s in range(len(states)):        
        for a in range(self.action_size):       
          action = list(best_actions[s])
          action[a] += 1        
          state_action.append((states[s], normalized(action)))
          changed_actions.append(action)        
      values = self.Q_batch(state_action)
      assert(len(values) == len(state_action))    
      j = 0    
      for s in range(len(states)):      
        for a in range(self.action_size):      
          if (values[j] < best_values[s]):
            best_values[s] = values[j]
            best_actions[s] = changed_actions[j]
          j += 1         
      assert(j == len(state_action))        
    return [(normalized(best_actions[s]), best_values[s]) for s in range(len(states))]
    
  def update(self, state_action_value):
    """
    Updates the state-value function based on the given combinations of states, actions, and values.
    :param state_action_value: List of tuples (state, action, value), where state is represented as a list of floats, action is represented as a normalized list of floats, and value is a float.
    """
    input_data = []
    output_data = []
    for (state, action, value) in state_action_value:
      assert(len(state) == self.state_size)
      assert(len(action) == self.action_size)
      assert(abs(sum(action) - 1) < EPSILON)
      assert(min(action) >= 0)
      input_data.append(state + action)
      output_data.append(value)
    input_train = array(input_data, dtype=float32)
    output_train = array(output_data, dtype=float32)  
    self.regressor.fit(input_train, output_train, epochs=QLearning.TRAIN_EPOCHS, batch_size=512)
    
  def learn(self, initial_state, state_observe, state_update, rnd=Random(0)):
    """
    Q-learning based algorithm for learning the best actions in every state.
    Note that due to performance reasons, the state-value function (i.e., Q) is not updated after every step, but only in batches.
    :param initial_state: Initial state, represented as an arbitrary object (note that this can be of a different format than the states used in other functions of QLearning).
    :param state_observe: Observes the state. Function, takes either initial_state or a state output by state_update, returns a list of floats (of length state_size).
    :param state_update: Updates the state based on an action. Function, takes a state (see state_observe) and an action (normalized list of floats), return the next state (may be arbitrary object).
    :param rnd: Random number generator.
    """
    # first phase (consider only immediate loss from action)
    state = initial_state
    state_action_value = []
    for i in range(QLearning.LEARN_INITIAL_ITERATIONS):
      action = normalized([rnd.random() for a in range(self.action_size)])
      (next_state, loss) = state_update(state, action)
      state_action_value.append((state_observe(state), action, loss))
      state = next_state
    start_time = time()
    self.update(state_action_value)
    duration = time() - start_time
    logging.info("First phase completed.")
    logging.info("Training duration: {}".format(duration))
    
    # temporary code for performance measurement
#    start_time = time()
#    state_action = []
#    for i in range(10000):
#      action = normalized([rnd.random() for i in range(self.action_size)])
#      state = normalized([rnd.random() for i in range(self.state_size)])
#      state_action.append((state, action))
#      self.Q(state, action)
#    self.Q_batch(state_action)
#    duration = time() - start_time
#    logging.info("Test duration: {}".format(duration))
        
    # second phase (consider value of next state)
    state = initial_state
    state_action_loss = []
    next_states = []
    for i in range(QLearning.LEARN_ITERATIONS):
      action = normalized([rnd.random() for a in range(self.action_size)])
      (next_state, loss) = state_update(state, action)
      state_action_loss.append((state_observe(state), action, loss))
      next_states.append(state_observe(next_state))
      state = next_state
    action_value = self.best_action_batch(next_states)
    assert(len(action_value) == len(next_states))
    state_action_value = []
    for i in range(QLearning.LEARN_ITERATIONS):
      (state, action, loss) = state_action_loss[i]
      value = loss + QLearning.DISCOUNT * action_value[i][1]
      state_action_value.append((state, action, value))
    self.update(state_action_value)
    logging.info("Second phase completed.")

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
    q = QLearning(len(model.alert_types) * model.horizon, len(model.alert_types) * model.horizon)
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
    q.learn(Model.State(model),
            lambda state: flatten_lists(state.N),
            state_update)
            
if __name__ == "__main__":
  logging.basicConfig(format='%(asctime)s / %(levelname)s: %(message)s', level=logging.DEBUG)
  model = test_model()
  DefenderBestResponse(model, test_attack_action)

