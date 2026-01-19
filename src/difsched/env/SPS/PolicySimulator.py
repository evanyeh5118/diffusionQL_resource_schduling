from .RewardFuntions import RewardKernel
from tqdm import tqdm
import numpy as np

class PolicySimulator:
    def __init__(self, env, policies=None, userMap=None):
        self.env = env
        self.N_user = self.env.N_user
        self.loadPolicies(policies, userMap)
        
    def reset(self):
        self.env.reset()

    def setupModes(self, obvMode="perfect", mode="test", type="data"):
        self.obvMode = obvMode
        self.mode = mode
        self.type = type
        self.env.selectMode(mode=self.mode, type=self.type)
        
    def loadPolicies(self, policies, userMap):
        self.policies = policies
        self.userMap = userMap
        if self.policies is not None and self.userMap is not None:
            if len(self.userMap) != len(self.policies):
                raise ValueError(f"The sum of userMap is not equal to the number of users: {len(self.userMap)} != {len(self.policies)}")
            userMapTotal = [u for i in range(len(self.userMap)) for u in self.userMap[i]]
            if sorted(userMapTotal) != list(range(self.N_user)):
                raise ValueError(f"userMap does not cover all users exactly once. userMapTotal: {userMapTotal}, expected: {list(range(self.N_user))}")

    def runSimulation(self, 
                      num_windows=1000, 
                      N_episodes=10,
                      verbose=True):
        self.env.reset()
        self.env.selectMode(mode=self.mode, type=self.type)
        self.rewardRecord, self.actionsRecord, self.uRecord, self.uNextRecord = [], [], [], []
        with tqdm(range(num_windows), desc="Simulation Progress", leave=False, disable=not verbose) as window_bar:
            for window in window_bar:
                u, u_predicted = self.env.getStates()
                u_active = self._observeStates(u, u_predicted)
                r = self._merge_actions(u_active)
                reward = self.env.applyActions(r)
                self.env.updateStates()
                u_next, u_next_predicted = self.env.getStates()
                self._recordResults(reward, r, u, u_next, u_predicted, u_next_predicted)
                if window % (num_windows/N_episodes) == 0:
                    self.env.reset()
                    self.env.selectMode(mode=self.mode, type=self.type)
        simResult = {
            "rewardRecord": self.rewardRecord,
            "actionsRecord": self.actionsRecord,
            "uRecord": self.uRecord,
            "uNextRecord": self.uNextRecord
        }
        return simResult
    
    def _observeStates(self, u, u_predicted):
        if self.obvMode == "perfect":
            return u
        elif self.obvMode == "predicted":
            return u_predicted
        else:
            raise ValueError(f"Invalid observation mode: {self.obvMode}")
    
    def _merge_actions(self, u):
        r_list = []
        for i, policy in enumerate(self.policies):
            r = policy.predict(u[self.userMap[i]])
            r_list.append(r)
        r_flat = np.concatenate(r_list).reshape(-1)
        return r_flat
    
    def _recordResults(self, reward, r, u, u_next, u_predicted, u_next_predicted):
        self.rewardRecord.append(reward)
        self.actionsRecord.append(r)
        if self.obvMode == "perfect":
            self.uRecord.append(u)
            self.uNextRecord.append(u_next)
        elif self.obvMode == "predicted":
            self.uRecord.append(u_predicted)
            self.uNextRecord.append(u_next_predicted)

    
