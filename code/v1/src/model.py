#!/usr/bin/env python3

"""Model of the alert prioritization problem, including alert and attack types."""

from numpy.random import poisson
from numpy import product
from random import Random

class PoissonDistribution:
  """Poisson distribution with fixed mean."""
  def __init__(self, lam=100):
    """
    Construct a Poisson distribution object.
    :param lam: Mean of the distribution.
    """
    self.mean = lam
    self.generate = lambda: poisson(lam)

class AlertType:
  """Representation of an alert type."""
  def __init__(self, cost=1, false_alerts=PoissonDistribution(), name=None):
    """
    Construct an alert type object.
    :param cost: Cost of investigating an alert of this type (i.e., C_t).
    :param false_alerts: Distribution of false alerts of this type (i.e., F_t).
    :param name: Name of this alert type.
    """
    assert(cost > 0)
    self.cost = cost
    self.false_alerts = false_alerts
    self.name = name

class AttackType:
  """Representation of an attack type."""
  def __init__(self, loss, cost, pr_alert, name):
    """
    Construct an attack type object.
    :param loss: Loss inflicted by an undetected attack of this type for various ages (i.e., L_{h,a}). Single-dimensional list of floats, length equal to the time horizon.
    :param cost: Cost of mounting an attack of this type (i.e., E_a).
    :param pr_alert: Probability of triggering an alert for various alert types (i.e., P_{a,t}). Single-dimensional list of floats, length equal to the number of alert types.
    :param name: Name of this attack type.
    """
    assert(cost > 0)
    assert(min(loss) >= 0)
    assert(min(pr_alert) >= 0.0)
    assert(max(pr_alert) <= 1.0)
    self.loss = loss
    self.cost = cost
    self.pr_alert = pr_alert
    self.name = name

class Model:
  """Game-theoretic model of the alert prioritization problem."""
  def __init__(self, horizon, alert_types, attack_types, def_budget, adv_budget):
    """
    Construct a model object.
    :param horizon: Time horizon (i.e., H).
    :param alert_types: Types of alerts (i.e., T). Single-dimensional list of AlertType objects.
    :param attack_types: Types of attacks (i.e., A). Single-dimensional list of AttackType objects.
    :param def_budget: Budget of the defender (i.e., B).
    :param adv_budget: Budget of the adversary (i.e., D).
    """
    assert(horizon > 0)
    for a in attack_types:
      assert(len(a.loss) == horizon)
      assert(len(a.pr_alert) == len(alert_types))
    assert(def_budget > 0)
    assert(adv_budget > 0)
    self.horizon = horizon
    self.alert_types = alert_types
    self.attack_types = attack_types
    self.def_budget = def_budget
    self.adv_budget = adv_budget
    self.rnd = Random(0)
    
  def is_feasible_investigation(self, N, delta):
    """
    Determine if a given investigation action is feasible in a given state.
    :param N: Number of yet uninvestigated alerts. Two-dimensional list, N[h][t] is the number of uninvestigated alerts of type t raised h time steps ago.
    :param delta: Number of alerts to investigate. Two-dimensional list, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
    :return: True if the investigation action is feasible, false otherwise.
    """
    assert(len(N) == self.horizon)
    assert(len(delta) == self.horizon)
    for h in range(self.horizon):
      assert(len(N[h]) == len(self.alert_types))
      assert(len(delta[h]) == len(self.alert_types))
    cost = 0.0
    for h in range(self.horizon):
      for t in range(len(self.alert_types)):
        if delta[h][t] > N[h][t]:
          return False
        cost += self.alert_types[t].cost * delta[h][t]
    return cost <= self.def_budget
    
  def make_investigation_feasible(self, N, delta):
    """
    Compute an investigation action that is feasible in a given state and resembles the given investigation action.
    :param N: Number of yet uninvestigated alerts. Two-dimensional list, N[h][t] is the number of uninvestigated alerts of type t raised h time steps ago.
    :param delta: Investigation action (see function is_feasible_investigation).
    :return: Feasible investigation action.
    """
    assert(len(N) == self.horizon)
    assert(len(delta) == self.horizon)
    cost = 0.0
    for h in range(self.horizon):
      for t in range(len(self.alert_types)):
        cost += self.alert_types[t].cost * delta[h][t]
    if cost == 0:
      return delta
    factor = self.def_budget / cost
    delta_feasible = []
    for h in range(self.horizon):
      delta_feasible.append([min(int(delta[h][t] * factor), N[h][t]) for t in range(len(self.alert_types))])
    return delta_feasible  
    
  def is_feasible_attack(self, alpha):
    """
    Determine if a given attack action is feasible.
    :param alpha: Probability of mounting attacks. One-dimensional list, alpha[a] is the probability of mounting an attack of type a.
    :return: True if the attack action is feasible, false otherwise.
    """
    assert(len(alpha) == len(self.attack_types))
    cost = 0.0
    for a in range(len(self.attack_types)):
      cost += self.attack_types[a].cost * alpha[a]
    return cost <= self.adv_budget
    
  class State:
    """State of the game in a certain time step."""
    def __init__(self, model, N=None, M=None, R=None, U=None):
      """
      Construct a state object.
      :param model: Model of the alert prioritization problem (i.e., Model object).
      :param N: Number of yet uninvestigated alerts. Two-dimensional list, N[h][t] is the number of uninvestigated alerts of type t raised h time steps ago.
      :param M: Indicator of undetected attacks. Two-dimensional list, M[h][a] == 1 if an attack of type a was mounted h time steps ago, M[h][a] == 0 otherwise.
      :param R: Indicator of true alerts. Three-dimensional list, R[h][a][t] == 1 if an alert of type t was raised due to an attack of type a mounted h time steps ago, R[h][a][t] == 0 otherwise.
      :param U: Cumulative loss sustained by the defender.
      """
      self.model = model
      if N is None:
        self.N = [[0 for t in range(len(model.alert_types))] for h in range(model.horizon)]
        self.M = [[0 for a in range(len(model.attack_types))] for h in range(model.horizon)]
        self.R = [[[0 for t in range(len(model.alert_types))] for a in range(len(model.attack_types))] for h in range(model.horizon)]
        self.U = 0.0
      else:
        assert(len(N) == model.horizon)
        assert(len(M) == model.horizon)
        assert(len(R) == model.horizon)
        for h in range(model.horizon):
          assert(len(N[h]) == len(model.alert_types))
          assert(len(M[h]) == len(model.attack_types))
          assert(len(R[h]) == len(model.attack_types))
          for a in range(len(model.attack_types)):
            assert(len(R[h][a]) == len(model.alert_types))
        self.N = N
        self.M = M
        self.R = R
        self.U = U
    def __str__(self):
      return "N: {}, M: {}, R: {}, U: {}".format(self.N, self.M, self.R, self.U)
         
  def next_state(self, state, delta, alpha, rnd=None):
    """
    Compute the next state of the game given a defense action and an adversarial strategy. Note that the defense action delta is the specific number of alerts delta to investigate, while the adversary strategy alpha is a policy that returns the attack action given the state of the game.
    :param state: State of the alert prioritization problem (i.e., Model.State object).
    :param delta: Number of alerts to investigate. Two-dimensional list, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
    :param alpha: Attack policy. Function, takes a model and a state, returns the probability of mounting attacks (one-dimensional list) given a model and a state.
    :param rnd: Random number generator.
    :return: Next state (i.e., Model.State object).
    """
    assert(self.is_feasible_investigation(state.N, delta))
    if rnd is None:
      rnd = self.rnd
    next = Model.State(self)
    # 1. Attack investigation
    for a in range(len(self.attack_types)):
      for h in range(1, self.horizon):
        if (state.M[h-1][a] == 1) and (
            rnd.random() < product([1 - state.R[h-1][a][t] * delta[h-1][t] / max(state.N[h-1][t], 1) for t in range(len(self.alert_types))])):
          next.M[h][a] = 1
        else:
          next.M[h][a] = 0
    # 2. and 3. Aging of alerts and true alerts
    for t in range(len(self.alert_types)):
      for h in range(1, self.horizon):
        next.N[h][t] = state.N[h-1][t] - delta[h-1][t]
        for a in range(len(self.attack_types)):
          next.R[h][a][t] = state.R[h-1][a][t]
    # 4. Attacks
    pr_attacks = alpha(self, state)
    assert(self.is_feasible_attack(pr_attacks))
    for a in range(len(self.attack_types)):
      if rnd.random() < pr_attacks[a]:
        next.M[0][a] = 1
      else:
        next.M[0][a] = 0
    # 5. Losses
    next.U = state.U + sum((
               sum((
                 self.attack_types[a].loss[h] * next.M[h][a] 
               for h in range(self.horizon))) 
             for a in range(len(self.attack_types))))
    # 6. True alerts
    for a in range(len(self.attack_types)):
      for t in range(len(self.alert_types)):
        if rnd.random() < self.attack_types[a].pr_alert[t] * next.M[0][a]:
          next.R[0][a][t] = 1
        else:
          next.R[0][a][t] = 0
    # 7. Alerts
    for t in range(len(self.alert_types)):
      next.N[0][t] = self.alert_types[t].false_alerts.generate() + sum((next.R[0][a][t] for a in range(len(self.attack_types))))    
    return next

