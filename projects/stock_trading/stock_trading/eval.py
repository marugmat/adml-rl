from __future__ import annotations

import gym_anytrading
from gym_anytrading.envs import StocksEnv

from .envs import IndicatorStocksEnv, PositionSizeStocksEnv


def backtest_baseline(model, test_df, window_size: int = 5):
    test_env = StocksEnv(
        df=test_df,
        frame_bound=(window_size, len(test_df)),
        window_size=window_size,
    )

    obs, _ = test_env.reset()
    done = False
    actions = []
    rewards = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = test_env.step(action)
        done = terminated or truncated
        actions.append(action)
        rewards.append(reward)

    env = test_env.unwrapped
    total_reward = float(sum(rewards))
    total_profit = float(env._total_profit)
    total_return = (total_profit - 1) * 100

    return {
        "env": test_env,
        "actions": actions,
        "rewards": rewards,
        "total_reward": total_reward,
        "total_profit": total_profit,
        "total_return": total_return,
    }


def backtest_indicators(model, test_ind_df, window_size: int = 5):
    test_env = IndicatorStocksEnv(
        test_ind_df, window_size=window_size, frame_bound=(window_size, len(test_ind_df))
    )
    obs, _ = test_env.reset()
    done = False
    actions = []
    rewards = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = test_env.step(action)
        done = terminated or truncated
        actions.append(int(action))
        rewards.append(reward)

    return {
        "env": test_env,
        "actions": actions,
        "rewards": rewards,
        "total_reward": float(sum(rewards)),
        "total_profit": float(test_env.total_profit),
    }


def backtest_position_size(
    model,
    test_ind_df,
    window_size: int = 5,
    transaction_cost: float = 0.001,
    slippage: float = 0.0005,
):
    test_env = PositionSizeStocksEnv(
        test_ind_df,
        window_size=window_size,
        frame_bound=(window_size, len(test_ind_df)),
        transaction_cost=transaction_cost,
        slippage=slippage,
    )
    obs, _ = test_env.reset()
    done = False
    actions = []
    rewards = []

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = test_env.step(action)
        done = terminated or truncated
        if isinstance(action, (list, tuple)):
            actions.append(float(action[0]))
        elif hasattr(action, "__len__"):
            actions.append(float(action[0]))
        else:
            actions.append(float(action))
        rewards.append(reward)

    return {
        "env": test_env,
        "actions": actions,
        "rewards": rewards,
        "total_reward": float(sum(rewards)),
        "total_profit": float(test_env.total_profit),
    }
