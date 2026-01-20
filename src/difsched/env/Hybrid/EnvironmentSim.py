import pickle
import random
import numpy as np

from .Helpers.Simulators import SimulatorType1, SimulatorType2
from .Helpers.TrafficGenerator import TrafficGenerator

def createEnv(envParams, trafficDataParentPath):    
    '''
    with open(f'{trafficDataParentPath}/trafficData_{envParams["dataflow"]}_LenWindow{envParams["LEN_window"]}.pkl', 'rb') as f:
        trafficData = pickle.load(f)
    trafficGenerator = TrafficGenerator(envParams)
    trafficGenerator.registerDataset(
        trafficData['trafficSource_train_actual'], trafficData['trafficSource_test_actual'],
        trafficData['trafficTarget_train_predicted'], trafficData['trafficTarget_test_predicted']
    )
    '''
    #with open(f'{trafficDataParentPath}/trafficData_{envParams["dataflow"]}_LenWindow{envParams["LEN_window"]}.pkl', 'rb') as f:
    #    trafficData = pickle.load(f)
    with open(f'{trafficDataParentPath}/combined_flows_forward_1ms_20_train_predictions.pkl', 'rb') as f:
        trainData = pickle.load(f)
    with open(f'{trafficDataParentPath}/combined_flows_forward_1ms_20_test_predictions.pkl', 'rb') as f:
        testData = pickle.load(f)
    trafficGenerator = TrafficGenerator(envParams)
    trafficGenerator.registerDataset(
        np.array(trainData['actual']).astype(int), np.array(testData['predicted']).astype(int),
        np.array(trainData['actual']).astype(int), np.array(testData['predicted']).astype(int)
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
        self.env_sigmas = params['sigma_list']
        self.failuresTotal = 0
        self.activeTotal = 0
        self.simulatorType1 = SimulatorType1(params)
        self.simulatorType2 = SimulatorType2(params)
        self.trafficGenerator = trafficGenerator
        self.u = self.trafficGenerator.updateTraffic()

    def selectMode(self, mode="train", type="markov"):
        self.trafficGenerator.selectModeAndType(mode=mode, type=type)

    def updateStates(self):
        self.trafficGenerator.updateTraffic()
        self.u, self.u_predicted = self.trafficGenerator.getUserStates()
        self.u = self.u.astype(int)
        self.u_predicted = self.u_predicted.astype(int)
        current_sigma = random.choice(range(len(self.env_sigmas)))
        self.simulatorType1.wirelessModel.swichSigma(current_sigma)
        self.simulatorType2.wirelessModel.swichSigma(current_sigma)
        
    def getStates(self):
        return self.u.copy(), self.u_predicted.copy()
    
    def applyActions(self, w, r, M, alpha):
        countFailedType1, countActiveType1 = self.simulatorType1.step(self.u, w, r, alpha)
        countFailedType2, countActiveType2 = self.simulatorType2.step(self.u, w, M, alpha)
        self.failuresTotal += countFailedType1+countFailedType2
        self.activeTotal += countActiveType1+countActiveType2
        reward = (countFailedType1+countFailedType2) / (countActiveType1+countActiveType2+1e-10)
        return reward
    
    def getPacketLossRate(self):
        return self.failuresTotal/(self.activeTotal+1e-10)
    
    def reset(self):
        self.failuresTotal = 0
        self.activeTotal = 0
        self.simulatorType1.reset()
        self.simulatorType2.reset()
        self.trafficGenerator.reset()
        self.updateStates()

