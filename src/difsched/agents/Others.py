from src.difsched.env.RewardFuntions import RewardKernel

import numpy as np

class PolicyDemoAdaptiveAlpha:
    def __init__(self, params):
        self.params = params
        self.rewardKernel = RewardKernel(params)
        self.M = 3
        self.alphaList = np.linspace(params['alpha_range'][0], params['alpha_range'][1], params['discrete_alpha_steps'])
    
    def predict(self, u):
        w = self.typeAllocator(u, self.params['LEN_window'])
        JmdpRecord = []
        for alpha in self.alphaList:
            r = np.floor(alpha*self.params['B'])/(np.sum(w)+1e-10) * w 
            Jmdp = self.rewardKernel.getReward(u, w, r, self.M, alpha)
            JmdpRecord.append(Jmdp)
        alpha = self.alphaList[np.argmin(JmdpRecord)]
        r = self.getDependentAction(u, w, alpha, self.params['B'])
        return w, r, self.M, alpha

    def getActionsByGivenAlpha(self, u, alpha):
        w = self.typeAllocator(u, self.params['LEN_window'])
        r = self.getDependentAction(u, w, alpha, self.params['B'])
        return w, r, self.M
    
    def typeAllocator(self, u, lEN_window):
        w = (u>int(lEN_window*0.35)).astype(int)
        return w
    
    def getDependentAction(self, u, w, alpha, B):
        r = np.floor(alpha*B)/(np.sum(w)+1e-10) * w
        return r
    

class RandomPolicy:
    def __init__(self, params):
        self.params = params
        self.N_user = params['N_user']
        self.M_list = range(1, 10)
        self.alphaList = np.linspace(params['alpha_range'][0], params['alpha_range'][1], params['discrete_alpha_steps'])

    def predict(self, u):
        w = np.random.choice([0,1], size=self.N_user)
        r = np.random.randint(0, self.params['B'], self.N_user)
        M = np.random.choice(self.M_list)
        alpha = np.random.choice(self.alphaList)
        return w, r, M, alpha