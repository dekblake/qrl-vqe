import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 


class ResidualEnv(gym.Env):
    def __init__(self, num_assets, mu_data, var_data, vqe_data, risk_aversion=0.5, fee_pct=0.001):
        super(ResidualEnv, self).__init__()
        self.num_assets = num_assets
        self.risk_aversion = risk_aversion
        self.fee_pct = fee_pct

        self.mu_data = mu_data
        self.var_data = var_data
        self.vqe_data = vqe_data

        self.current_day = 0
        self.max_days = len(mu_data) - 1

        self.action_space = spaces.MultiBinary(num_assets*2)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(num_assets * 4,), dtype=np.float32)

        self.current_weights = np.zeros(num_assets)
        self.current_tiers = np.zeros(num_assets)
        
    def _get_obs(self):

        #ARIMA GARCH + VQE
        mu = self.mu_data[self.current_day]
        var = self.var_data[self.current_day]
        vqe_target = self.vqe_data[self.current_day]
        
        obs = np.concatenate([mu, var, vqe_target, self.current_tiers]).astype(np.float32)

        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # starting at random days in the dataset 
        self.current_day = np.random.randint(0, self.max_days - 30)

        self.current_weights = np.zeros(self.num_assets)
        self.current_tiers = np.zeros(self.num_assets)
        
        return self._get_obs(), {}

    def step(self, action_bits):
        # 1. DECODE ACTIONS (From 20 bits -> 10 integer tiers 0,1,2,3)
        action_pairs = np.reshape(action_bits, (self.num_assets, 2))
        target_tiers = action_pairs[:, 0] * 2 + action_pairs[:, 1]
        
        # 2. CALCULATE WEIGHTS (Normalize tiers so they sum to 1.0)
        total_tiers = np.sum(target_tiers)
        if total_tiers == 0:
            new_weights = np.zeros(self.num_assets) # 100% Cash
        else:
            new_weights = target_tiers / total_tiers
            
        # 3. GET TODAY'S MARKET DATA
        mu = self.mu_data[self.current_day]
        var = self.var_data[self.current_day]

        # diagonal covariance matrix from the variances for the reward calculation 
        Sigma = np.diag(var)
        
        # 4. FINRL MATH: Transaction Costs (Turnover)
        # How much of the portfolio weight had to be shifted?
        turnover = np.sum(np.abs(new_weights - self.current_weights))
        transaction_cost = turnover * self.fee_pct
        
        # 5. FINRL MATH: Portfolio Return & Risk (Markowitz)
        expected_return = np.dot(new_weights, mu)
        portfolio_variance = np.dot(new_weights.T, np.dot(Sigma, new_weights))
        
        # 6. CALCULATE THE REWARD (The Dynamic QUBO)
        # Maximize return, minimize risk, minimize trading fees!
        reward = expected_return - (self.risk_aversion * portfolio_variance) - transaction_cost
        
        # 7. UPDATE STATE FOR TOMORROW
        self.current_weights = new_weights
        self.current_tiers = target_tiers
        self.current_day += 1

        # end of data check
        terminated = self.current_day >= self.max_days
        
        # Create tomorrow's observation with check
        next_obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape)
        
        
        return next_obs, float(reward), terminated, False, {}
