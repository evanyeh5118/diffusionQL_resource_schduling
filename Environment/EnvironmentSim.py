import pickle
import numpy as np

from .Helpers.Simulators import SimulatorType1
from .Helpers.TrafficGenerator import TrafficGenerator

def createEnv(envParams, trafficDataParentPath):    
    with open(f'{trafficDataParentPath}/trafficData_{envParams["dataflow"]}_LenWindow{envParams["LEN_window"]}.pkl', 'rb') as f:
        trafficData = pickle.load(f)
    trafficGenerator = TrafficGenerator(envParams)
    trafficGenerator.registerDataset(
        trafficData['trafficSource_train_actual'], trafficData['trafficSource_test_actual'],
        trafficData['trafficTarget_train_predicted'], trafficData['trafficTarget_test_predicted']
    )
    simEnv = Environment(envParams, trafficGenerator)
    simEnv.selectMode(mode="train", type="data")
    return simEnv


class Environment:
    def __init__(self, params, trafficGenerator):
        self.B = params['B']
        self.r_bar = params['r_bar']
        self.LEN_window = params['LEN_window']
        self.N_user = params['N_user']
        self.failuresTotal = 0
        self.activeTotal = 0
        self.simulatorType1 = SimulatorType1(params)
        self.trafficGenerator = trafficGenerator
        self.u = self.trafficGenerator.updateTraffic()

    def selectMode(self, mode="train", type="markov"):
        self.trafficGenerator.selectModeAndType(mode=mode, type=type)

    def updateStates(self):
        self.trafficGenerator.updateTraffic()
        self.u, self.u_predicted = self.trafficGenerator.getUserStates()
        self.u = self.u.astype(int)
        self.u_predicted = self.u_predicted.astype(int)
        self.simulatorType1.wirelessModel.randomlySwitchParameters()
        
    def getStates(self):
        return self.u.copy(), self.u_predicted.copy()
    
    def applyActions(self, r):
        w = np.ones(self.N_user).astype(int)
        kappa = 1.0
        countFailedType1, countActiveType1 = self.simulatorType1.step(self.u, w, r, kappa)
        self.failuresTotal += countFailedType1
        self.activeTotal += countActiveType1
        reward = countFailedType1 / countActiveType1
        return reward
    
    def getPacketLossRate(self):
            return self.failuresTotal/self.activeTotal
    
    def reset(self):
        self.failuresTotal = 0
        self.activeTotal = 0
        self.simulatorType1.reset()
        self.trafficGenerator.reset()
        self.updateStates()

