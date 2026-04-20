import pandas as pd
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from scipy.stats import norm, kurtosis, skew
import datetime

# ==============================================================================
# SECTION 0 — CONFIGURABLE BANK REGISTRY (config.py)
# ==============================================================================

BANK_REGISTRY = {
    "CBZ": {
        "file": "CBZ_Holdings_Cleaned.csv",
        "full_name": "CBZ Holdings",
        "has_precomputed_returns": True,
        "has_log_returns": True,
        "date_range": ("2021-01-04", "2025-12-31"),
        "observations": 1085
    },
    "ZB": {
        "file": "ZB_Holdings_Cleaned.csv",
        "full_name": "ZB Financial Holdings",
        "has_precomputed_returns": True,
        "has_log_returns": True,
        "date_range": ("2021-01-05", "2025-12-31"),
        "observations": 714
    },
    "FBC": {
        "file": "FBC_Holdings_Cleaned.csv",
        "full_name": "FBC Holdings",
        "has_precomputed_returns": False,
        "has_log_returns": False,
        "date_range": ("2021-01-04", "2025-12-31"),
        "observations": None
    }
}

ACTIVE_BANKS = ["CBZ", "ZB"]
MACRO_FILE = "ZSE_Economic_Data.csv"

# ==============================================================================
# SECTION 1 — DATA PIPELINE (data_pipeline.py)
# ==============================================================================

class ZSEDataPipeline:
    def __init__(self, bank_registry, active_banks, macro_file, macro_indicators=None):
        self.registry = bank_registry
        self.active_banks = active_banks
        self.macro_file = macro_file
        self.macro_indicators = macro_indicators or [
            'CPI_Inflation_%', 'ZSE_VFEX_Spread_pts', 'RBZ_Policy_Rate_%',
            'USD_ZiG_Rate', 'Minerals_Export_Index', 'VIX', 'Gold_Price_USD_oz'
        ]
        self.bank_data = {}
        self.macro_data = None
        self.combined_data = None

    def ingest_price_data(self, ticker, custom_df=None):
        if custom_df is not None:
            df = custom_df.copy()
        else:
            config = self.registry.get(ticker, {
                "date_range": ("2021-01-04", "2025-12-31"),
            })
            dates = pd.date_range(start=config['date_range'][0], end=config['date_range'][1], freq='B')
            n = len(dates)
            df = pd.DataFrame(index=dates)
            df['Price'] = 100 * (1 + np.random.normal(0.0005, 0.02, n)).cumprod()
            df['Open'] = df['Price'] * (1 + np.random.normal(0, 0.005, n))
            df['High'] = df[['Price', 'Open']].max(axis=1) * (1 + abs(np.random.normal(0, 0.005, n)))
            df['Low'] = df[['Price', 'Open']].min(axis=1) * (1 - abs(np.random.normal(0, 0.005, n)))
            df['Vol.'] = np.random.lognormal(10, 1, n)

        # Standard Processing
        df = df.rename(columns={'Price': 'Close', 'Vol.': 'Volume', 'Change%': 'Pct_Change'})
        df['Volume_Imputed'] = df['Volume'].isna()
        df['Volume'] = df['Volume'].ffill(limit=3).fillna(0)
        
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change()
        if 'Log_Return' not in df.columns:
            df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        df = df.dropna(subset=['Daily_Return'])
        return df

    def ingest_macro_data(self, custom_df=None):
        if custom_df is not None:
            df = custom_df.copy()
        else:
            dates = pd.date_range(start="2021-01-31", end="2025-12-31", freq='ME')
            n = len(dates)
            df = pd.DataFrame(index=dates)
            for indicator in self.macro_indicators:
                df[indicator] = np.random.uniform(10, 100, n)
        
        daily_dates = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
        df = df.reindex(daily_dates).ffill()
        return df

    def feature_engineering(self, df):
        # Price-based
        df['Rolling_Vol_20d'] = df['Daily_Return'].rolling(20).std()
        df['Rolling_Vol_60d'] = df['Daily_Return'].rolling(60).std()
        df['Vol_Ratio'] = df['Rolling_Vol_20d'] / df['Rolling_Vol_60d']
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['OHLC_Momentum'] = (df['Close'] - df['Open']) / df['Open']
        
        # Technical
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs)) / 100.0
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # BB
        ma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['BB_Position'] = (df['Close'] - ma20) / (2 * std20)
        
        df['EMA_Ratio'] = (df['Close'].ewm(span=12).mean() / df['Close'].ewm(span=26).mean()) - 1
        
        # Sentiment Placeholder
        df['Sent_t'] = 0.0
        
        # Rolling Normalization
        continuous_cols = ['Daily_Return', 'Log_Return', 'Rolling_Vol_20d', 'Rolling_Vol_60d', 
                           'Vol_Ratio', 'MACD_Histogram', 'EMA_Ratio', 'Price_Range', 
                           'OHLC_Momentum', 'Volume_Ratio']
        
        for col in continuous_cols:
            mu = df[col].rolling(60).mean()
            sigma = df[col].rolling(60).std()
            df[f'z_{col}'] = (df[col] - mu) / sigma
            df[f'z_{col}'] = df[f'z_{col}'].clip(-3, 3)
            
        return df

# ==============================================================================
# SECTION 2 — GYMNASIUM ENVIRONMENT (zse_env.py)
# ==============================================================================

class ZSEBankTradingEnv(gym.Env):
    def __init__(self, data, active_banks, macro_indicators, initial_capital=1.0):
        super(ZSEBankTradingEnv, self).__init__()
        self.data = data
        self.active_banks = active_banks
        self.macro_indicators = macro_indicators
        self.n_banks = len(active_banks)
        self.initial_capital = initial_capital
        
        # Observation space: (13 * N_banks) + N_macro + N_banks portfolio + 1 time
        self.obs_dim = (13 * self.n_banks) + len(macro_indicators) + self.n_banks + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        
        # Action space: Continuous portfolio weights (raw, will be softmaxed)
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_banks,), dtype=np.float32)
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 60 # Start after normalization window
        self.portfolio_value = self.initial_capital
        self.weights = np.array([1.0 / self.n_banks] * self.n_banks)
        self.peak_value = self.initial_capital
        self.history = []
        
        return self._get_observation(), {}

    def _get_observation(self):
        # Extract features for current step
        obs = []
        # Bank features
        for bank in self.active_banks:
            # Simulated extraction
            obs.extend([0.0] * 13) 
        
        # Macro features
        obs.extend([0.0] * len(self.macro_indicators))
        
        # Portfolio state
        obs.extend(self.weights.tolist())
        
        # Time
        obs.append(self.current_step / len(self.data))
        
        return np.array(obs, dtype=np.float32)

    def step(self, action):
        # Softmax action to get weights
        exp_action = np.exp(action)
        new_weights = exp_action / np.sum(exp_action)
        
        # Execution Filter & Transaction Costs
        # ... logic ...
        
        # Portfolio Evolution
        # ... logic ...
        
        self.current_step += 1
        terminated = self.current_step >= len(self.data) - 1
        
        # Zero-Ruin Check
        mdd = (self.peak_value - self.portfolio_value) / self.peak_value
        if mdd > 0.15:
            reward = -10.0
            terminated = True
        else:
            reward = 0.0 # Placeholder for DSR
            
        return self._get_observation(), reward, terminated, False, {}

    def get_decision_rule(self, current_weights, proposed_weights):
        decisions = {}
        for i, bank in enumerate(self.active_banks):
            diff = proposed_weights[i] - current_weights[i]
            if diff > 0.02:
                decisions[bank] = "BUY"
            elif diff < -0.02:
                decisions[bank] = "SELL"
            else:
                decisions[bank] = "HOLD"
        return decisions

# ==============================================================================
# SECTION 4 — BAYESIAN PARTICLE FILTER (agent.py)
# ==============================================================================

class ParticleFilter:
    def __init__(self, n_particles=500):
        self.n = n_particles
        # 0: LOW, 1: HIGH, 2: CRISIS
        self.particles = np.random.choice([0, 1, 2], size=n_particles)
        self.weights = np.ones(n_particles) / n_particles
        
    def update(self, obs_vol):
        # Likelihoods
        l_low = norm.pdf(obs_vol, 0.02, 0.01)
        l_high = norm.pdf(obs_vol, 0.06, 0.02)
        l_crisis = norm.pdf(obs_vol, 0.15, 0.05)
        
        likelihoods = np.array([l_low, l_high, l_crisis])
        
        for i in range(self.n):
            self.weights[i] *= likelihoods[self.particles[i]]
            
        self.weights /= np.sum(self.weights)
        
        # Resample if ESS < N/2
        ess = 1.0 / np.sum(self.weights**2)
        if ess < self.n / 2:
            self._resample()
            
    def _resample(self):
        indices = np.random.choice(range(self.n), size=self.n, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.n) / self.n
        
    def get_regime_probs(self):
        probs = [np.sum(self.weights[self.particles == i]) for i in range(3)]
        return np.array(probs)

# ==============================================================================
# MAIN PROTOTYPE EXECUTION (Simulated)
# ==============================================================================

if __name__ == "__main__":
    print("ZSE AI Trading Prototype Initialized.")
    print(f"Active Banks: {ACTIVE_BANKS}")
    # Pipeline execution would go here
