import numpy as np
from scipy.signal import convolve
from math import ceil

class WirelessModel():
    def __init__(self, params):
        self.r_min = 0
        self.r_max = params['B']
        self.r_list = np.arange(self.r_min, self.r_max).astype(int)
        self.N_r = len(self.r_list)
        self.sigma_list = params['sigma_list']
        self.packetTransmissionCDF_list = []
        self.current_sigma_idx = 0
        self.initialize()
        
    def initialize(self):
        self.analyticalModel = WirelessAnalyticalModel()
        for sigma in self.sigma_list:
            self.analyticalModel.params["sigma"] = sigma
            self.packetTransmissionCDF_list.append(self.analyticalModel.computeCDF())
    
    def swichSigma(self, idx):
        self.current_sigma_idx = idx

    def successfulPacketCDF(self, r):
        r_idx = np.clip(np.searchsorted(self.r_list, r), 0, self.N_r-1)
        return self.packetTransmissionCDF_list[self.current_sigma_idx][r_idx]

class WirelessAnalyticalModel:
    def __init__(self, param=None, D_range=None, N_D_points=None):
        self.params = param
        self.N_D_points = N_D_points
        self.cdf = None
        self.initialize()

    def initialize(self):
        if self.params is None:
            self.params = {}
            self.setDefaultParams()
        if  self.N_D_points is None:
            self.N_D_points=4000

    def setDefaultParams(self):
        # Set 1
        self.params["alpha"] = 10 ** 13.6       # Large path loss exponent
        self.params["beta"] = 3.6               # Path loss exponent
        self.params["G"] = 10 ** 0.3            # Antenna gain
        self.params["L"] = 10 ** 0.7            # Losses
        self.params["PRBsize"] = 180000         # 180 kHz
        self.params["noise_dB"] = -174 + 10 * np.log10(self.params["PRBsize"]) + 6  # +6 dB for NoiseRise
        self.params["bruit"] = 10 ** (self.params["noise_dB"] / 10) / 1000           # Noise + interference in watts
        self.params["Pmax"] = 0.5               # Maximum power
        self.params["sigma"] = 0.9
        
        # Set 2
        self.params["packet_size_bits"] = 100 * 8   # 0.1 KB in bits
        self.params["slot_length"] = 0.001          # 1 ms
        self.params["K"] = self.params["packet_size_bits"] / (self.params["PRBsize"]* self.params["slot_length"])  # Transmission efficiency factor
        self.params["a"] = 0.5
        self.params["E_max"] = 8      # dB
        self.params["E_min"] = 0.01  # dB
        self.params["lambda_"] = 1 / 2  # Parameter of the fast fading exponential distribution

    #===========================================================================
    #=========================Analytical Result=================================
    #===========================================================================
    def computeCDF(self):
        under, over = self.rho_bounds(self.params)
        n_min = max(1, int(np.floor(under)) - 2)
        n_max = int(np.ceil(over)) + 2
        grid_int = np.arange(n_min, n_max + 1)
        F_grid = self.analytic_cdf_one_packet_rayleigh(grid_int, self.params, R=1.0)
        pmf = self.pmf_from_integer_grid_cdf(F_grid)
        cdf = np.cumsum(pmf)
        return cdf

    def analytic_cdf_one_packet_rayleigh(self, r_vals, p, R):
        """
        F_rho(r) = ∫_0^R P(rho <= r | d) * (2d/R^2) dd, with Rayleigh(sigma) fading.
        Uses trapezoidal rule over d for proper normalization.
        """
        n_d = self.N_D_points
        r_vals = np.atleast_1d(r_vals).astype(float)
        under, over = self.rho_bounds(p)

        # distance grid on (0,R]; avoid d=0
        d = np.linspace(0.0, R, n_d+1)[1:]
        w = 2.0 * d / (R**2)                         # p_D(d) = 2d/R^2

        # conditional interior CDF for Rayleigh: exp( - h(r;d)^2 / (2 sigma^2) )
        H = self.h_of_r(r_vals[:, None], d[None, :], p)   # (m,n_d)
        F_cond = np.exp( - (H**2) / (2.0 * p["sigma"]**2) )

        # clamp by rows (below underline -> 0, above overline -> 1)
        below = r_vals <= under
        above = r_vals >= over
        if np.any(below):
            F_cond[below, :] = 0.0
        if np.any(above):
            F_cond[above, :] = 1.0

        # Proper integral over d
        F = np.trapz(F_cond * w[None, :], d, axis=1)

        # numerical hygiene
        F = np.clip(F, 0.0, 1.0)
        return F

    def pmf_from_integer_grid_cdf(self, F_on_grid):
        # P(N=n) = F(n) - F(n-1) assuming grid is consecutive integers
        diff = F_on_grid[1:] - F_on_grid[:-1]
        return np.clip(diff, 0.0, 1.0)

    # ===================== Helper Functions =====================
    def c_of_d(self, d, p):
        return (p["G"]*p["Pmax"]) / (p["alpha"] * d**p["beta"] * p["L"] * p["bruit"])

    def rho_bounds(self, p):
        return p["K"]/(p["a"]*p["E_max"]), p["K"]/(p["a"]*p["E_min"])

    def h_of_r(self, r, d, p):
        # r: (m,1), d: (1,n) -> (m,n)
        return (2.0**(p["K"]/(p["a"]*r)) - 1.0) / self.c_of_d(d, p)

    

