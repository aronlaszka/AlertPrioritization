from math import exp, factorial, floor
from itertools import permutations
from scipy.optimize import linprog
from numpy.random import seed, sample, choice
from time import clock

import sys

class Game:
    def __init__(self, T, A, B, D, G, C, K, R, F):        
        self.T = list(range(T))
        self.A = list(range(A))
        self.B = B
        self.D = D
        self.G = G
        self.C = C
        self.K = K
        self.R = R
        self.F = F
        self.cache_PD = {}
        
    def PD(self, p, a):
        # returns PD(p, a) for prioritization p (which must given as a tuple of alert types) and attack a
        if (p,a) not in self.cache_PD:
          Brange = list(range(self.B + 1))
          def detection(b, t):
            # computes the probability that the attack is detected using alerts of type t with budget b
            return self.R[a][t] * sum((self.F[t](0.5, k) for k in range(floor(b / float(self.C[t]))))) 
          if len(p) == 1:
            self.cache_PD[(p,a)] = detection(self.B, p[0])
          else:
            #PD = {len(self.T) - 1: {b: detection(b, p[-1]) for b in Brange}}
            PD = {len(p) - 1: {b: detection(b, p[-1]) for b in Brange}}
            #for i in range(len(self.T) - 2, -1, -1):
            for i in range(len(p) - 2, -1, -1):
              # note that this will probably crash with C > 1 because we might refer to PD[i+1][<0]
              PD[i] = {b: detection(b, p[i]) + (1 - self.R[a][p[i]]) * sum((self.F[p[i]](1, j) * PD[i+1][b - j * self.C[p[i]]] for j in range(floor(b / float(self.C[p[i]])) + 1))) for b in Brange}
            self.cache_PD[(p,a)] = PD[0][self.B]
        return self.cache_PD[(p,a)]
    
    def generateColumnGreedy(self, reducedCost):
      types = list(self.T)
      prio = []
      while len(types) > 0:
        optType = None
        maxCost = float("-inf")
        for t in types:
          cost = reducedCost(tuple(prio + [t]))
          if cost > maxCost:
            maxCost = cost
            optType = t
        prio = prio + [optType]
        types.remove(optType)
      if maxCost > 0.0:
        return tuple(prio)
      else:
        return None
        
    def optimalPrioritizationAttack(self, a, method):
        def diff(p, other):
            return (1 - self.PD(p, other)) * self.G[other] - (1 - self.PD(p, a)) * self.G[a]
        if method is "ColumnGeneration":
            prios = [tuple(self.T)] # start with an arbitrary ordering
        elif method is "Exhaustive":
            prios = list(permutations(self.T))
        while True:
            # primal
            c = [-self.PD(p,a) for p in prios]
            b_ub = []
            A_ub = []            
            for other in self.A:
                b_ub.append(self.K[other] - self.K[a])
                A_ub.append([diff(p, other) for p in prios])
            b_ub.append(1)
            A_ub.append([1] * len(prios))
            b_ub.append(-1)
            A_ub.append([-1] * len(prios))
            result = linprog(c=c, b_ub=b_ub, A_ub=A_ub) 
            if not result.success:
                #print(result)
                #raise Error()  
                loss =  float("inf")
                strategy = None
                break

            loss = (1 + result.fun) * self.D[a]
            strategy = [(prios[p], result.x[p]) for p in range(len(prios)) if result.x[p] > 0.0]
            #print("############################################################################")
            #print("a = {}".format(a))
            #print("loss = {}".format(loss))
            #print(result)
            if method is "Exhaustive":
                break
                        
            # dual
            dual_c = b_ub
            dual_A_ub = []
            for col in range(len(A_ub[0])):
                dual_A_ub.append([-A_ub[row][col] for row in range(len(A_ub))])
            dual_b_ub = c
            dual_result = linprog(c=dual_c, b_ub=dual_b_ub, A_ub=dual_A_ub)
            if not dual_result.success:
                print(result)
                raise Error()
            
            def reduced_cost(p):
                return self.PD(p, a) - sum((diff(p, other) * dual_result.x[other] for other in self.A))
            column = self.generateColumnGreedy(reduced_cost)
            if (column is None) or (column in prios):
                break
            prios.append(column)
        return {'loss': loss, 'strategy': strategy}  

    def optimalPrioritization(self, method="ColumnGeneration"):
        loss_opt = float("inf")
        strat_opt = None
        for a in self.A:
            result = self.optimalPrioritizationAttack(a, method)
            if result['loss'] < loss_opt:
                loss_opt = result['loss']
                strat_opt = result['strategy']   
        return {'loss': loss_opt, 'strategy': strat_opt}

def poisson(mean, time, k):
    param = mean * time
    return param**k * exp(-param) / factorial(k)  

def prio_transform(prio):
    n_alert = len(prio)
    prio_after = [0]*n_alert
    priority = 1
    for i in range(n_alert):
        prio_after[prio[i]] = priority
        priority += 1
    return prio_after

def aics_policy(def_budget):
    T = 3 # Number of alert types
    A = 3 # Number of attack types
    B = def_budget # Defender's budget
    D = [9.374, 12.14, 16.03] # Defender's loss 
    G = [9.374, 12.14, 16.03] # attacker's gain
    C = [1, 1, 1] # Cost of investigating an alert
    K = [1, 3, 2] # Cost of mounting an attack
    P = [[1.0*0.9, 0.67*0.9, 0.0], # Probability of a type of alert raised by an attack
         [0.01*0.9, 0.96*0.9, 0.13*0.9],
         [0.0*0.9, 0.45*0.9, 0.94*0.9]]
    R = {a: {t: P[a][t] for t in range(T)} for a in range(A)}
    FP_mean = [10, 47, 39] # The mean of false positive alerts
    F = {t: lambda time, k: poisson(FP_mean[t], time, k) for t in range(T)}
    game = Game(T, A, B, D, G, C, K, R, F)
    result =  game.optimalPrioritization("Exhaustive")['strategy']
    prio_profile = []
    prob_profile = []
    for soln in result:
        prio_profile.append(prio_transform(list(soln[0])))
        prob_profile.append(soln[1])
    return prio_profile, prob_profile

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("python aics.py [def_budget]")
        sys.exit(1)

    def_budget = int(sys.argv[1])
    prio_profile, prob_profile = aics_policy(def_budget)
    print(prio_profile, prob_profile)

