import gymnasium as gym 
from gymnasium import spaces 
import numpy as np 


class ResidualEnv(gym.Env):
    def __init__(self, num_assets=10, risk_aversion=0.5, fee_pct=0.001):
        super(ResidualEnv, self).__init__()
        self.num_assets = num_assets
        self.risk_aversion = risk_aversion
        self.fee_pct = fee_pct

        self.action_space = spaces.Multibinary(num_assets*2)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(num_assets * 4,), dtype=np.float32)

        self.current_weights = np.zeros(num_assets)
        self.current_tiers = np.zeros(num_assets)
        
    def _get_mock_market_data(self):
        #until ARIMA GARCH
        mu = np.random.uniform(-0.02, 0.05, self.num_assets)
        
        # Create a mock Covariance matrix (Sigma)
        A = np.random.rand(self.num_assets, self.num_assets)
        Sigma = np.dot(A, A.T) * 0.01 
        var = np.diag(Sigma) # Extract just the diagonal for the state space

        vqe_target = np.random.randint(0, 4, self.num_assets)

        return mu, Sigma, var, vqe_target
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_weights = np.zeros(self.num_assets)
        self.current_tiers = np.zeros(self.num_assets)
        
        mu, _, var, vqe_target = self._get_mock_market_data()
        
        # Construct the 40-feature state tensor
        obs = np.concatenate([mu, var, vqe_target, self.current_tiers]).astype(np.float32)
        return obs, {}

    def step(self, action_20_bit):
        # 1. DECODE ACTIONS (From 20 bits -> 10 integer tiers 0,1,2,3)
        action_pairs = np.reshape(action_20_bit, (self.num_assets, 2))
        target_tiers = action_pairs[:, 0] * 2 + action_pairs[:, 1]
        
        # 2. CALCULATE WEIGHTS (Normalize tiers so they sum to 1.0)
        total_tiers = np.sum(target_tiers)
        if total_tiers == 0:
            new_weights = np.zeros(self.num_assets) # 100% Cash
        else:
            new_weights = target_tiers / total_tiers
            
        # 3. GET TODAY'S MARKET DATA
        mu, Sigma, var, vqe_target = self._get_mock_market_data()
        
        # 4. FINRL MATH: Transaction Costs (Turnover)
        # How much of the portfolio weight had to be shifted?
        turnover = np.sum(np.abs(new_weights - self.current_weights))
        transaction_cost = turnover * self.transaction_fee_percent
        
        # 5. FINRL MATH: Portfolio Return & Risk (Markowitz)
        expected_return = np.dot(new_weights, mu)
        portfolio_variance = np.dot(new_weights.T, np.dot(Sigma, new_weights))
        
        # 6. CALCULATE THE REWARD (The Dynamic QUBO)
        # Maximize return, minimize risk, minimize trading fees!
        reward = expected_return - (self.risk_aversion * portfolio_variance) - transaction_cost
        
        # 7. UPDATE STATE FOR TOMORROW
        self.current_weights = new_weights
        self.current_tiers = target_tiers
        
        # Create tomorrow's observation
        next_obs = np.concatenate([mu, var, vqe_target, self.current_tiers]).astype(np.float32)
        
        # We'll just say an episode is done after a fixed time in the training loop
        terminated = False 
        
        return next_obs, float(reward), terminated, False, {}
