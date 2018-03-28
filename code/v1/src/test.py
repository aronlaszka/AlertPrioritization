#!/usr/bin/env python3

"""Test instances for the alert prioritization model."""

from model import PoissonDistribution, AlertType, AttackType, Model

def test_model():
  """
  Create a very simple test instance of the model (H = 3, |T| = 2, |A| = 2).
  :return: Model object.
  """
  alert_types =  [AlertType(1, PoissonDistribution(100), "t1"), 
                  AlertType(1, PoissonDistribution(100), "t2")]   
  attack_types = [AttackType([0, 1, 1], 100, [0.2, 0.9], "a1"),
                  AttackType([0, 1, 1], 100, [0.9, 0.2], "a2")]
  model = Model(3, alert_types, attack_types, 100, 100)
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
  model = test_model()
  state = Model.State(model)
  print(test_defense_action(model, state))
  print(test_attack_action(model, state))
  i = 0
  while True:
    print(i, state)
    state = model.next_state(state, test_defense_action(model, state), test_attack_action)
    i += 1

