from __future__ import annotations

import gymnasium as gym
import numpy as np


class IndicatorStocksEnv(gym.Env):
    """Simple stocks env with indicator features.
    Observation: flattened window of features (window_size * n_features,)
    Actions: Discrete(2) -> 0: sell/flat, 1: buy/long
    Reward: position * price_change + realized PnL on close
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 5,
        frame_bound=(5, None),
        reward_type: str = "simple",
        risk_factor: float = 0.01,
        risk_free_rate: float = 0.0,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.frame_bound = (frame_bound[0], frame_bound[1] or len(df))
        self.start = self.frame_bound[0]
        self.end = self.frame_bound[1]

        # Use Close + all additional feature columns
        base_cols = ["Close"]
        self.features = base_cols + [
            c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Date", "Index"]
        ]
        self.feature_array = self.df[self.features].values

        # Discrete buy/sell actions (keep consistent with baseline)
        self.action_space = gym.spaces.Discrete(2)

        # Flattened observation
        n_feats = len(self.features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * n_feats,),
            dtype=np.float32,
        )

        # Reward function selection
        self.reward_type = reward_type
        self.risk_factor = risk_factor
        self.risk_free_rate = risk_free_rate
        self.returns_buffer = []
        self.sharpe_window = 20

        self._seed = None
        self.seed(0)
        self.reset()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self._seed = int(seed)
        np.random.seed(self._seed)
        return [self._seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start
        self.position = 0
        self.entry_price = 0.0
        self._total_reward = 0.0
        self.returns_buffer = []
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        idx = self.current_step
        window = self.feature_array[idx - self.window_size : idx]
        return window.astype(np.float32).reshape(-1)

    def step(self, action):
        prev_price = self.df.loc[self.current_step - 1, "Close"]
        cur_price = self.df.loc[self.current_step, "Close"]
        price_change = cur_price - prev_price

        profit = self.position * price_change
        realized_pnl = 0.0

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = cur_price
        elif action == 0 and self.position == 1:
            realized_pnl = cur_price - self.entry_price
            self.position = 0
            self.entry_price = 0.0

        if self.reward_type == "simple":
            reward = profit + realized_pnl
        elif self.reward_type == "risk_adjusted":
            risk_penalty = abs(price_change) * self.risk_factor
            reward = profit + realized_pnl - risk_penalty
        elif self.reward_type == "sharpe":
            step_return = profit + realized_pnl
            self.returns_buffer.append(step_return)
            if len(self.returns_buffer) > self.sharpe_window:
                self.returns_buffer.pop(0)
            mean_return = np.mean(self.returns_buffer) if self.returns_buffer else 0.0
            volatility = np.std(self.returns_buffer) + 1e-9
            reward = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        else:
            reward = profit + realized_pnl

        self._total_reward += reward
        self.current_step += 1

        terminated = False
        if self.current_step >= self.end:
            terminated = True
            if self.position == 1:
                last_price = self.df.loc[self.end - 1, "Close"]
                pnl = last_price - self.entry_price
                self._total_reward += pnl
                self.position = 0

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        truncated = False
        info = {}
        return obs, float(reward), terminated, truncated, info

    @property
    def total_profit(self):
        return float(self._total_reward)


class PositionSizeStocksEnv(gym.Env):
    """Stocks env with continuous position size and transaction costs.
    Observation: flattened window of features (window_size * n_features,)
    Actions: Box(-1, 1) -> target position size (-1 short, 0 flat, 1 long)
    Reward: position * price_change - trading_costs
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        window_size: int = 5,
        frame_bound=(5, None),
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.frame_bound = (frame_bound[0], frame_bound[1] or len(df))
        self.start = self.frame_bound[0]
        self.end = self.frame_bound[1]

        base_cols = ["Close"]
        self.features = base_cols + [
            c for c in df.columns if c not in ["Open", "High", "Low", "Close", "Date", "Index"]
        ]
        self.feature_array = self.df[self.features].values

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        n_feats = len(self.features)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size * n_feats,),
            dtype=np.float32,
        )

        self.transaction_cost = transaction_cost
        self.slippage = slippage

        self._seed = None
        self.seed(0)
        self.reset()

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        self._seed = int(seed)
        np.random.seed(self._seed)
        return [self._seed]

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.start
        self.position = 0.0
        self._total_reward = 0.0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        idx = self.current_step
        window = self.feature_array[idx - self.window_size : idx]
        return window.astype(np.float32).reshape(-1)

    def step(self, action):
        prev_price = self.df.loc[self.current_step - 1, "Close"]
        cur_price = self.df.loc[self.current_step, "Close"]
        price_change = cur_price - prev_price

        if isinstance(action, (np.ndarray, list)):
            action = float(action[0])
        target_position = float(np.clip(action, -1.0, 1.0))

        trade_size = abs(target_position - self.position)
        cost_rate = self.transaction_cost + self.slippage
        trading_cost = trade_size * cost_rate * cur_price

        reward = self.position * price_change - trading_cost
        self.position = target_position

        self._total_reward += reward
        self.current_step += 1

        terminated = False
        if self.current_step >= self.end:
            terminated = True

        obs = (
            self._get_obs()
            if not terminated
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )
        truncated = False
        info = {}
        return obs, float(reward), terminated, truncated, info

    @property
    def total_profit(self):
        return float(self._total_reward)
