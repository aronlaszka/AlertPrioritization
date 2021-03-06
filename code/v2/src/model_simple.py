#!/usr/bin/env python3

"""Model of the alert prioritization problem, including alert and attack types."""

from numpy.random import poisson, normal
from numpy import product
from random import Random
import numpy as np
import copy
import math
import scipy

EPSILON = 0.00001

def prob_investigation(n, m, k, r):
    """
    Get the the probability of an attack being investigated.
    It is the probability that at least one true alert is investigated.
    :param n: Number of alert type t.
    :param m: Number of true alerts if an attack a is mounted.
    :param k: Number of investigations.
    :param r: A boolean value showing whether an attack a raises alert t.
    """
    result = 0
    #print(n, m, k, r)
    if n == 0 or r == 0:
        result = 0
    else:
        result = 1 - product([1 - m*1.0/(n-i) for i in range(int(k))])
    return result 

class PoissonDistribution:
  """Poisson distribution with fixed mean."""
  def __init__(self, lam=100):
    """
    Construct a Poisson distribution object.
    :param lam: Mean of the distribution.
    """
    self.mean = lam
    self.generate = lambda: poisson(lam)

class NormalDistribution:
  """Poisson distribution with fixed mean and std."""
  def __init__(self, mu, sigma):
    """
    Construct a Normal distribution object.
    :param mu: Mean of the distribution.
    :param sigma: Std of the distribution.
    :return: An interger generated by the distribution
    """
    self.mu = mu
    self.sigma = sigma
    self.generate = lambda: int(normal(mu, sigma))

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
    #assert(min(pr_alert) >= 0.0)
    #assert(max(pr_alert) <= 1.0)
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
    if len(delta) != self.horizon:
      print(delta)
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
    #return cost  <= self.def_budget
    if cost  <= self.def_budget:
      return True
    else:
      print(cost, self.def_budget)
      return False
    
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
      delta_feasible.append([min(int(delta[h][t] * factor-1), N[h][t]) for t in range(len(self.alert_types))])
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
    #return cost <= self.adv_budget
    if cost - EPSILON <= self.adv_budget:
      return True
    else:
      print(cost-EPSILON, self.adv_budget)
      return False

  def make_attack_feasible(self, alpha):
    """
    Compute an attack action that is feasible and resembles the given attack action.
    :param alpha: Attack action (see function is_feasible_attack).
    :return: Feasible attack action.
    """    
    assert(len(alpha) == len(self.attack_types))
    cost_alpha = 0.0
    for a in range(len(self.attack_types)):
      cost_alpha += self.attack_types[a].cost * alpha[a]
    if cost_alpha == 0:
      return alpha
    factor = self.adv_budget / cost_alpha
    alpha_feasible = []
    for a in range(len(self.attack_types)):
      alpha_feasible.append(max(alpha[a] * factor - EPSILON, 0))
    return alpha_feasible    

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
         
  def next_state(self, mode, state, delta, alpha, rnd=None):
    """
    Compute the next state of the game given a defense action and an adversarial strategy. Note that the defense action delta is the specific number of alerts delta to investigate, while the adversary strategy alpha is a policy that returns the attack action given the state of the game.
    :param state: State of the alert prioritization problem (i.e., Model.State object).
    :param delta: Number of alerts to investigate. Two-dimensional list, delta[h][t] is the number of alerts to investigate of type t raised h time steps ago.
    :param alpha: Attack policy. Function, takes a model and a state, returns the probability of mounting attacks (one-dimensional list) given a model and a state.
    :param rnd: Random number generator.
    :return: Next state (i.e., Model.State object).
    """
    if isinstance(delta, list):
      delta = delta
    else:
      delta = delta(self, state)
    assert(self.is_feasible_investigation(state.N, delta))
    if rnd is None:
      rnd = self.rnd
    next = Model.State(self)

    # 1. Attack investigation
    M_now = copy.deepcopy(state.M)
    for a in range(len(self.attack_types)):
      for h in range(self.horizon):
        coin = np.random.random()
        fact = product([1 - np.sign(state.R[h][a][t]) * prob_investigation(state.N[h][t], self.attack_types[a].pr_alert[t], delta[h][t], state.R[h][a][t])  for t in range(len(self.alert_types))])
        #fact = product([scipy.special.comb(state.N[h][t]-state.R[h][a][t], delta[h][t], exact=True) / scipy.special.comb(state.N[h][t], delta[h][t], exact=True) for t in range(len(self.alert_types))])
        if (state.M[h][a] == 1) and (coin < fact):
          state.M[h][a] = 1
        else:
          state.M[h][a] = 0

    def_loss = 0.0
    if mode == 'new':
      for a in range(len(self.attack_types)):
        if M_now[0][a] == 1 and state.M[0][a] == 1:
          def_loss += self.attack_types[a].loss[0]
        if M_now[0][a] == 1 and state.M[0][a] == 0:
          def_loss -= self.attack_types[a].loss[0]
    else:
      for a in range(len(self.attack_types)):
        def_loss += self.attack_types[a].loss[0] * state.M[0][a]
    
    # 2. Attacks
    if isinstance(alpha, list):
      pr_attacks = alpha
    else:
      pr_attacks = alpha(self, state)
    if not self.is_feasible_attack(pr_attacks):
      print(pr_attacks)
    assert(self.is_feasible_attack(pr_attacks))
    for a in range(len(self.attack_types)):
      if np.random.random() < pr_attacks[a]:
        next.M[0][a] = 1
      else:
        next.M[0][a] = 0

    next.U = state.U + def_loss

    # 3. True alerts
    for a in range(len(self.attack_types)):
      for t in range(len(self.alert_types)):
        #next.R[0][a][t] = self.attack_types[a].pr_alert[t] * next.M[0][a]
        if np.random.random() < self.attack_types[a].pr_alert[t] * next.M[0][a]:
          next.R[0][a][t] = math.ceil(self.attack_types[a].pr_alert[t])
        else:
          next.R[0][a][t] = 0

    # 4. Alerts
    for t in range(len(self.alert_types)):
      next.N[0][t] = self.alert_types[t].false_alerts.generate() + sum((next.R[0][a][t] for a in range(len(self.attack_types))))
    return next

