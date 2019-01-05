#!/usr/bin/env python3

"""Test instances for the alert prioritization model."""

from model_simple import PoissonDistribution, NormalDistribution, AlertType, AttackType, Model
import numpy as np

EPSILON = 0.00001

'''
def test_model_fraud(def_budget, adv_budget):
  """
  Creat a test model by using the UCI German Credit dataset (H = 3, |T| = 5, |A| = 8).
  :return: Model object. 
  """
  alert_types =  [AlertType(1, PoissonDistribution(0), "t1"), 
                  AlertType(1, PoissonDistribution(17), "t2"),
                  AlertType(1, PoissonDistribution(0), "t3"),
                  AlertType(1, PoissonDistribution(14), "t4"),
                  AlertType(1, PoissonDistribution(23), "t5")]
  attack_types = [AttackType([0, 0.1], 1, [0.09, 0.44, 0.08, 0.02, 0.27], "a1"),
                  AttackType([0, 1], 1, [0.06, 0.55, 0.09, 0.0, 0.2], "a2"),
                  AttackType([0, 10], 1, [0, 0.94, 0.02, 0.0, 0.0], "a3")]
  #model = Model(2, alert_types, attack_types, 10, 3)
  model = Model(2, alert_types, attack_types, def_budget, adv_budget)
  return model
'''

def test_model_fraud(def_budget, adv_budget):
  """
  Creat a test model by using the Credit Fraud dataset (H = 1, |T| = 3, |A| = 3).
  :return: Model object. 
  """
  alert_types =  [AlertType(1, PoissonDistribution(17), "t2"),
                  AlertType(1, PoissonDistribution(14), "t4"),
                  AlertType(1, PoissonDistribution(23), "t5")]
  attack_types = [AttackType([0.1], 1, [0.44, 0.08, 0.27], "a1"),
                  AttackType([1], 2, [0.55, 0.09, 0.2], "a2"),
                  AttackType([10], 3, [0.94, 0.02, 0.0], "a3")]
  #model = Model(2, alert_types, attack_types, 10, 3)
  model = Model(1, alert_types, attack_types, def_budget, adv_budget)
  return model

'''
def test_model_suricata2(def_budget, adv_budget):
  """
  Creat a test model by using the IDS dataset (H = 1, |T| = 10, |A| = 7).
  :return: Model object. 
  """
  alert_types =  [AlertType(1.0, PoissonDistribution(7200), "t1"), 
                  AlertType(1.0, PoissonDistribution(44100), "t2"),
                  AlertType(1.0, PoissonDistribution(1600), "t3"),
                  AlertType(1.0, PoissonDistribution(7300), "t4"),
                  AlertType(1.0, PoissonDistribution(17400), "t5"),
                  AlertType(1.0, PoissonDistribution(4000), "t6"),
                  AlertType(1.0, PoissonDistribution(10200), "t7"),
                  AlertType(1.0, PoissonDistribution(0), "t8"),
                  AlertType(1.0, PoissonDistribution(0), "t9"),
                  AlertType(1.0, PoissonDistribution(0), "t10")]
  attack_types = [AttackType([3.6], 120.0, [1230,0,0,0,0,0,0,0,4768,0], "a1"),
                  AttackType([6.0], 60.0, [0,4,2,106,0,54,0,0,0,0], "a2"),
                  AttackType([4.0], 74.0, [0,0,0,0,0,24,0,30350,0,42], "a3"),
                  AttackType([3.6], 20.0, [0,0,4,0,10,0,0,0,0,0], "a4"),
                  AttackType([1.4], 52.0, [710,2,862,12,0,80,600,54448,0,768], "a5"),
                  AttackType([1.4], 80.0, [138,0,320,30,0,0,0,0,0,0], "a6"),
                  AttackType([2.7], 62.0, [0,0,6,0,0,0,0,0,0,7192], "a7")]
  #model = Model(1, alert_types, attack_types, 500.0, 120.0)
  model = Model(1, alert_types, attack_types, def_budget, adv_budget)
  return model
'''

def test_model_suricata(def_budget, adv_budget):
  """
  Creat a test model by using the IDS dataset (H = 1, |T| = 7, |A| = 7).
  :return: Model object. 
  """
  alert_types =  [AlertType(1.0, PoissonDistribution(7200), "t1"), 
                  AlertType(1.0, PoissonDistribution(44100), "t2"),
                  AlertType(1.0, PoissonDistribution(1600), "t3"),
                  AlertType(1.0, PoissonDistribution(7300), "t4"),
                  AlertType(1.0, PoissonDistribution(17400), "t5"),
                  AlertType(1.0, PoissonDistribution(4000), "t6"),
                  AlertType(1.0, PoissonDistribution(10200), "t7")]
  attack_types = [AttackType([3.6], 120.0, [1230,0,0,0,0,0,0], "a1"),
                  AttackType([6.0], 60.0, [0,4,2,106,0,54,0], "a2"),
                  AttackType([4.0], 74.0, [0,0,0,0,0,24,0], "a3"),
                  AttackType([3.6], 20.0, [0,0,4,0,10,0,0], "a4"),
                  AttackType([1.4], 52.0, [710,2,862,12,0,80,600], "a5"),
                  AttackType([1.4], 80.0, [138,0,320,30,0,0,0], "a6"),
                  AttackType([2.7], 62.0, [0,0,6,0,0,0,0], "a7")]
  #model = Model(1, alert_types, attack_types, 1000.0, 120.0)
  model = Model(1, alert_types, attack_types, def_budget, adv_budget)
  return model

def test_defense_action(model, state):
  """
  Compute a basic investigation action (i.e., number of alerts to investigate), which distributes the defender's budget uniformly among alert types and ages.
  :param model: Model of the alert prioritization problem (i.e., Model object).
  :param state: State of the alert prioritization problem (i.e., Model.State object).
  :return: Number of alerts to investigate. Two-dimensional array, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
  """
  budget = model.def_budget / (model.horizon * len(model.alert_types))
  delta = []
  for h in range(model.horizon):
    delta.append([min(int(budget / model.alert_types[t].cost), state.N[h][t]) for t in range(len(model.alert_types))])
  return delta

def test_defense_newest(model, state):
  """
  Compute a basic investigation action (i.e., number of alerts to investigate), which distributes the defender's budget uniformly among newest alert.
  :param model: Model of the alert prioritization problem (i.e., Model object).
  :param state: State of the alert prioritization problem (i.e., Model.State object).
  :return: Number of alerts to investigate. Two-dimensional array, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
  """
  budget_for_newest = model.def_budget / len(model.alert_types) # We distribute the budget to the newest alerts
  delta = []
  for h in range(model.horizon):
    if h == 0:
      delta.append([min(int(budget_for_newest / model.alert_types[t].cost), state.N[h][t]) for t in range(len(model.alert_types))])
    else:
      delta.append([0] * len(model.alert_types))
  return delta
  
def test_defense_fraud(model, state):
  """
  Compute a basic investigation action (i.e., number of alerts to investigate), which distributes the defender's budget on some specific alerts.
  :param model: Model of the alert prioritization problem (i.e., Model object).
  :param state: State of the alert prioritization problem (i.e., Model.State object).
  :return: Number of alerts to investigate. Two-dimensional array, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
  """  
  delta = []
  means = [0, 17, 0, 14, 23] # The mean of each false positive alert types
  ratio = np.array(means)/np.sum(means)
  for h in range(model.horizon):
    if h == 0:
      delta.append([min(int(model.def_budget*ratio[t] / model.alert_types[t].cost), state.N[h][t]) for t in range(len(model.alert_types))])
    else:
      delta.append([0] * len(model.alert_types))
  return delta

def test_defense_suricata(model, state):
  """
  Compute an investigation action based on the built-in priorities of Suricata
  :param model: Model of the alert prioritization problem (i.e., Model object).
  :param state: State of the alert prioritization problem (i.e., Model.State object).
  :return: Number of alerts to investigate. Two-dimensional array, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
  """
  delta = []
  for h in range(model.horizon):
    delta.append([0]*len(model.alert_types))
  remain_budget = model.def_budget
  used_budget = 0.0
  #alert_priority = np.array([2,1,2,3,3,1,3,1,1,1])
  alert_priority = np.array([2,1,2,3,3,1,3])
  for i in range(np.unique(alert_priority).shape[0]):
    if remain_budget > 0:
      index_priority = np.where(alert_priority == i+1)[0]
      for j in index_priority:
        delta[0][j] = min(int(remain_budget / index_priority.shape[0] / model.alert_types[j].cost), state.N[0][j])
        used_budget += delta[0][j] * model.alert_types[j].cost
      remain_budget = model.def_budget - used_budget
    else:
      break
  return delta

def test_attack_action(model, state):
  """
  Compute a basic attack action (i.e., probability of mouting attacks), which distributes the adversary's budget uniformly among attack types.
  :param model: Model of the alert prioritization problem (i.e., Model object).
  :param state: State of the alert prioritization problem (i.e., Model.State object).
  :return: Probability of mounting attacks. One-dimensional array, alpha[a] is the probability of mounting an attack of type a.
  """
  budget = model.adv_budget / len(model.attack_types)
  alpha = [min(budget / a.cost, 1) for a in model.attack_types]
  return alpha

if __name__ == "__main__":
  model = test_model_suricata(1000, 120)
  state = Model.State(model)
  #print(test_defense_action(model, state))
  #print(test_attack_action(model, state))
  i = 0
  while i < 15:
    print('#############################')
    print(i)
    print('state:', state)
    print('defender:', test_defense_action(model, state))
    print('attacker:', test_attack_action(model, state))
    state = model.next_state(state, test_defense_newest, test_attack_action)
    i += 1

