from __future__ import annotations

import gym_anytrading
from gym_anytrading.envs import StocksEnv
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv

from .envs import IndicatorStocksEnv


def train_baseline_a2c(
    train_df,
    window_size: int = 5,
    total_timesteps: int = 50_000,
    learning_rate: float = 7e-4,
    gamma: float = 0.99,
    seed: int = 42,
):
    env = DummyVecEnv(
        [
            lambda: StocksEnv(
                df=train_df,
                frame_bound=(window_size, len(train_df)),
                window_size=window_size,
            )
        ]
    )

    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, env


def train_indicator_a2c(
    train_ind_df,
    window_size: int = 5,
    total_timesteps: int = 500_000,
    learning_rate: float = 7e-4,
    gamma: float = 0.9,
    seed: int = 24,
    reward_type: str = "risk_adjusted",
    risk_factor: float = 0.01,
):
    ind_env = DummyVecEnv(
        [
            lambda: IndicatorStocksEnv(
                train_ind_df,
                window_size=window_size,
                frame_bound=(window_size, len(train_ind_df)),
                reward_type=reward_type,
                risk_factor=risk_factor,
            )
        ]
    )

    model = A2C(
        "MlpPolicy",
        ind_env,
        learning_rate=learning_rate,
        gamma=gamma,
        verbose=1,
        seed=seed,
    )
    model.learn(total_timesteps=total_timesteps)
    return model, ind_env
