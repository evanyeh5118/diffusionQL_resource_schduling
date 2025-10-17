import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

from .Helpers.WirelessModel import WirelessModel

class RewardKernel:
    def __init__(self, params):
        self.Type1RewardKernel = Type1RewardKernel(params)
        self.Type1Constraint = Type1Constraint(params)
        self.N_user = params['N_user']

    def getReward(self, u, r):
        w = np.ones(self.N_user).astype(int)
        alpha = 1.0
        self.Jtype1 = self.Type1RewardKernel.getReward(u, w, r)
        self.Jc = self.Type1Constraint.getPenalty(w, r, alpha)

        return self.Jtype1 + self.Jc
        
class Type1RewardKernel:
    def __init__(self, params):
        self.B = params['B']
        self.wirelessModel = WirelessModel(params)

    def getReward(self, u, w, r):
        J = 1-np.sum(w * u * self.wirelessModel.successfulPacketCDF(r))  / (np.sum(w*u)+1e-10)
        return J
    
class Type1Constraint:
    def __init__(self, params):
       self.B = params['B']

    def getPenalty(self, w, r, alpha):
        cstCost = (alpha*self.B - np.sum(w*r))**2 if np.sum(w*r) - alpha*self.B > 0 else 0
        return cstCost
