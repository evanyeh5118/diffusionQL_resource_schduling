from src.difsched.env.SPS.RewardFuntions import RewardKernel as SpsRewardKernel
from src.difsched.env.Hybrid.RewardFuntions import RewardKernel as HybridRewardKernel
from .MdpBuilderHelpers import *
from .MdpSolver import *

class SpsEnvActionSpace:
    def __init__(self, params):
        self.params = params
        self.N_user = params['N_user']
        self.N_r = params['N_r']
        self.B = params['B']
        self.r_list = np.linspace(0, self.B, self.N_r)
   
    def getSizeActionSpace(self):
        return self.N_r**self.N_user
    
    def buildActionSpace(self):
        self.actionSpace = []
        for idx_w in range(self.N_r**self.N_user):
            r_idx = index_to_tuple(idx_w, self.N_r, self.N_user)
            r = [self.r_list[r_idx[i]] for i in range(self.N_user)]
            self.actionSpace.append(r)
        return self.actionSpace


class HybridEnvActionSpace:
    def __init__(self, params):
        self.params = params
        self.N_user = params['N_user']
        self.N_kappa = params['N_kappa']
        self.kappa_range = params['kappa_range']
        self.B = params['B']
        self.M_list = params['M_list']
        self.N_M = len(self.M_list)
        self.kappa_list = np.linspace(self.kappa_range[0], self.kappa_range[1], self.N_kappa)
        self.N_actions = (2**self.N_user) * self.N_M * self.N_kappa
        #action: (w, r, M, kappa(alpha)) is one action

    def getSizeActionSpace(self):
        return self.N_actions
    
    def buildActionSpace(self):
        self.actionSpace = []
        for idx_w in range(2**self.N_user*self.N_M):
            w = index_to_tuple(idx_w, 2, self.N_user)
            for kappa in self.kappa_list:
                for M in self.M_list:
                    self.actionSpace.append((w, M, kappa))
        return self.actionSpace

    def getDependentAction(self, w, kappa):
        w = np.array(w)
        r = np.floor(kappa*self.B)/(np.sum(w)+1e-10) * w
        return r

class MdpFormulator:
    def __init__(self, params, M_original):
        self.params = params
        self.N_user = params['N_user']
        self.LEN_window = params['LEN_window']
        self.N_aggregation = params['N_aggregation']
        if params['EnvType'] == 'SPS':
            self.envActionSpace = SpsEnvActionSpace(params)
            self.rewardKernel = SpsRewardKernel(params)
        elif params['EnvType'] == 'HYBRID':
            self.envActionSpace = HybridEnvActionSpace(params)
            self.rewardKernel = HybridRewardKernel(params)
        else:
            raise ValueError(f"Invalid EnvType: {params['EnvType']}")
        #------------------------------------------------------------
        self.M_original = M_original
        self.N_states_original = len(self.M_original) # (=self.LEN_window+1)
        (self.N_states, self.N_actions) = (None, None)
        self.actionSpace = None
        (self.M_aggregationSingle, self.p_aggregationSingle) = (None, None)
        (self.M_aggregation, self.p_aggregation) = (None, None)
        self.initialize()

    def initialize(self):
        #------------------------------------------------------------
        self.N_states = self.N_aggregation ** self.N_user
        self.N_actions = self.envActionSpace.getSizeActionSpace()
        self.actionSpace = self.envActionSpace.buildActionSpace()

    def aggregateModel(self, approximate=False):
        self.buildAggregatedModel()
        if approximate:
            self.buildAggregatedRewardTableApproximate()
        else:
            self.buildAggregatedRewardTable()
        
    def buildAggregatedModel(self):
        #---------- get aggregation mapping ----------
        self.p_original = compute_stationary_distribution(self.M_original)
        self.thresholds = optimal_threshold_binning_uniform_arr(self.p_original, self.N_aggregation)[1:]
        self.aggregationMap = np.searchsorted(self.thresholds, np.arange(self.N_states_original))
        C = vector_to_mapping_matrix(self.aggregationMap)
        #---------- get aggregated transition matrix ----------
        self.M_aggregationSingle = compute_aggregated_transition_matrix(self.M_original, C)
        self.M_aggregation = compute_joint_transition_matrix(self.M_aggregationSingle, self.N_user)
        self.p_aggregation = compute_stationary_distribution(self.M_aggregation)
    
    def buildAggregatedRewardTable(self):  
        N_stateOriginal = (self.LEN_window+1)**self.N_user
        N_stateAggregated = self.N_aggregation**self.N_user
        self.aggregatedRewardTable = np.zeros((N_stateAggregated, self.N_actions))
        for sOrigin in range(N_stateOriginal):
            uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
            sAggregated = self.from_origin_to_aggregated_state(sOrigin)
            for a in range(self.N_actions):
                reward = self.rewardKernel.getReward(np.array(uOrigin), np.array(r))
                self.aggregatedRewardTable[sAggregated, a] += (
                   reward * np.prod(self.p_original[uOrigin])
                )
    def buildAggregatedRewardTableApproximate(self):
        self.aggregatedRewardTable = np.zeros((self.N_states, self.N_actions))
        for s in range(self.N_states):
            uOriginExpectation = self.from_aggregated_to_origin_state_expectation(s)
            for a in range(self.N_actions):
                if self.params['EnvType'] == 'SPS':
                    r = self.actionSpace[a]
                    reward = self.rewardKernel.getReward(np.array(uOriginExpectation), np.array(r))
                elif self.params['EnvType'] == 'HYBRID':
                    w, M, kappa = self.actionSpace[a]
                    r = self.envActionSpace.getDependentAction(w, kappa)
                    reward = self.rewardKernel.getReward(np.array(uOriginExpectation), np.array(w), r, M, kappa)
                self.aggregatedRewardTable[s, a] = reward

    def getMdpKernel(self):
        mdpParams = self.params.copy()
        mdpParams['N_states'] = self.N_states
        mdpParams['N_actions'] = self.N_actions
        mdpParams['aggregationMap'] = self.aggregationMap
        mdpParams['N_aggregation'] = self.N_aggregation
        #-------------------------------------------------
        #actionTable = {i: self.actionSpace[i] for i in range(self.N_actions)}
        actionTable = self.actionSpace
        mdpParams['actionTable'] = actionTable
        #------------------------------------------------
        transitionTable = np.zeros((self.N_states, self.N_states, self.N_actions))
        for i in range(self.N_actions): transitionTable[:,:,i] = self.M_aggregation
        rewardTable = self.aggregatedRewardTable
        kernelParams = mdpParams.copy()
        kernelParams['transitionTable'] = transitionTable
        kernelParams['rewardTable'] = rewardTable
        return MdpKernel(kernelParams), mdpParams
        
    def from_origin_to_aggregated_state(self, sOrigin):
        uOrigin = index_to_tuple(sOrigin, self.LEN_window+1, self.N_user)
        uAggregated = self.aggregationMap[uOrigin]
        sAggregated = tuple_to_index(uAggregated, self.N_aggregation)
        return sAggregated
    
    def from_aggregated_to_origin_state_expectation(self, sAggregated):
        uAggregated = index_to_tuple(sAggregated, self.N_aggregation, self.N_user)
        uOrigiinApproximate = []
        for u in uAggregated:
            uOrigin_within_aggregation = np.where(self.aggregationMap == u)[0]
            uExpWeighted = (self.p_original[uOrigin_within_aggregation]/np.sum(self.p_original[uOrigin_within_aggregation])) * uOrigin_within_aggregation
            uOrigiinApproximate.append(int(np.floor(np.sum(uExpWeighted))))
        return uOrigiinApproximate