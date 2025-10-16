import numpy as np
from scipy.signal import convolve
from math import ceil

class WirelessModel():
    def __init__(self, params):
        self.r_min = 0
        self.r_max = params['B']
        self.r_list = np.arange(self.r_min, self.r_max).astype(int)
        self.sigmoid_k_list = params['sigmoid_k_list']
        self.sigmoid_s_list = params['sigmoid_s_list']
        self.N_r = len(self.r_list)
        self.p_idx = 0
        self.packetTransmissionCDF_list = []
        self.initialize()
        
    def initialize(self):
        for sigmoid_k, sigmoid_s in zip(self.sigmoid_k_list, self.sigmoid_s_list):
            self.packetTransmissionCDF_list.append(sigmoid(self.r_list, sigmoid_k, sigmoid_s))

    def randomlySwitchParameters(self):
        self.p_idx = np.random.randint(0, len(self.sigmoid_k_list))

    def successfulPacketCDF(self, r):
        r_idx = np.clip(np.searchsorted(self.r_list, r), 0, self.N_r-1)
        return self.packetTransmissionCDF_list[self.p_idx][r_idx]

def sigmoid(x, k, s):
    return 1 / (1 + np.exp(-k * (x - s)))