#!/usr/bin/env python3

"""Test instances for the alert prioritization model."""

from model import PoissonDistribution, AlertType, AttackType, Model

def test_model1():
  """
  Create a very simple test instance of the model (H = 3, |T| = 2, |A| = 2).
  :return: Model object.
  """
  alert_types =  [AlertType(1, PoissonDistribution(100), "t1"), 
                  AlertType(1, PoissonDistribution(80), "t2")]
  attack_types = [AttackType([0, 0.5, 0.8], 70, [0.2, 0.9], "a1"),
                  AttackType([0, 1.2, 1.5], 100, [0.5, 0.8], "a2")]                    
  model = Model(3, alert_types, attack_types, 100, 100)
  return model

def test_model2():
  """
  Create a very simple test instance of the model (H = 3, |T| = 2, |A| = 2).
  :return: Model object.
  """  
  alert_types =  [AlertType(1, PoissonDistribution(100), "t1"), 
                  AlertType(1, PoissonDistribution(80), "t2")]
  attack_types = [AttackType([0, 0.8, 1.4], 100, [0.2, 0.9], "a1"),
                  AttackType([0, 0.6, 1.0], 100, [0.6, 0.4], "a2")]                    
  model = Model(3, alert_types, attack_types, 100, 100)
  return model
  
def test_model3():
  """
  Create a very simple test instance of the model (H = 3, |T| = 6, |A| = 4).
  :return: Model object.
  """
  alert_types =  [AlertType(1, PoissonDistribution(100), "t1"), 
                  AlertType(1, PoissonDistribution(90), "t2"),
                  AlertType(1, PoissonDistribution(70), "t3"),
                  AlertType(1, PoissonDistribution(65), "t4"),
                  AlertType(1, PoissonDistribution(80), "t5"),
                  AlertType(1, PoissonDistribution(75), "t6")]  
  attack_types = [AttackType([0, 0.2, 0.4], 75, [0.2, 0.9, 0.8, 0.1, 0.4, 0.6], "a1"),
                  AttackType([0, 0.8, 1.2], 100, [0.1, 0.7, 0.2, 0.5, 0.8, 0.6], "a2"),
                  AttackType([0, 0.5, 0.9], 89, [0.1, 0.1, 0.5, 0.5, 0.5, 0.9], "a3"),
                  AttackType([0, 0.8, 0.9], 64, [0.6, 0.2, 0.1, 0.5, 0.3, 0.8], "a4")]
  model = Model(3, alert_types, attack_types, 300, 200)
  return model

def test_model_credit():
  """
  Creat a test model by using the UCI German Credit dataset (H = 3, |T| = 5, |A| = 8).
  :return: Model object. 
  """
  alert_types =  [AlertType(1, PoissonDistribution(355), "t1"), 
                  AlertType(1, PoissonDistribution(82), "t2"),
                  AlertType(1, PoissonDistribution(22), "t3"),
                  AlertType(1, PoissonDistribution(36), "t4"),
                  AlertType(1, PoissonDistribution(8), "t5")]
  attack_types = [AttackType([0, 1.0, 1.5], 1, [0.37, 0.31, 0.07, 0.0, 0.0], "a1"),
                  AttackType([0, 0.5, 0.9], 1, [0.37, 0.0, 0.0, 0.0, 0.0], "a2"),
                  AttackType([0, 0.3, 0.5], 1, [0.37, 0.0, 0.0, 0.07, 0.0], "a3"),
                  AttackType([0, 0.2, 0.3], 1, [0.37, 0.0, 0.0, 0.07, 0.0], "a4"),
                  AttackType([0, 0.4, 0.7], 1, [0.37, 0.0, 0.0, 0.07, 0.0], "a5"),
                  AttackType([0, 0.2, 0.4], 1, [0.37, 0.0, 0.0, 0.0, 0.0], "a6"),
                  AttackType([0, 1.0, 1.2], 1, [0.37, 0.31, 0.0, 0.0, 0.0], "a7"),
                  AttackType([0, 1.1, 1.3], 1, [0.37, 0.0, 0.0, 0.0, 0.11], "a8")]
  model = Model(3, alert_types, attack_types, 250, 100)
  return model

def test_defense_action(model, state):
  """
  Compute a basic investigation action (i.e., number of alerts to investigate), which distributes the defender's budget uniformly among alert types and ages.
  :param model: Model of the alert prioritization problem (i.e., Model object).
  :param state: State of the alert prioritization problem (i.e., Model.State object).
  :return: Number of alerts to investigate. Two-dimensional array, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
  """
  budget = model.def_budget / (model.horizon * len(model.alert_types))
  budget_for_newest = model.def_budget / len(model.alert_types) # We distribute the budget to the newest alerts
  delta = []
  for h in range(model.horizon):
    delta.append([min(int(budget / model.alert_types[t].cost), state.N[h][t]) for t in range(len(model.alert_types))])
    #if h == 0:
      #delta.append([min(int(budget_for_newest / model.alert_types[t].cost), state.N[h][t]) for t in range(len(model.alert_types))])
    #  delta.append([min(int(model.def_budget), state.N[h][0]), 0])
    #else:
    #  delta.append([0]*len(model.alert_types))
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
  model = test_model_credit()
  state = Model.State(model)
  print(test_defense_action(model, state))
  print(test_attack_action(model, state))
  i = 0
  while i < 100:
    print(i, state)
    state = model.next_state(state, test_defense_newest(model, state), test_attack_action)
    i += 1

