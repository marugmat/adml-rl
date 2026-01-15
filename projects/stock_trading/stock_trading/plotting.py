from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_price_history(df, ticker: str):
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"])
    plt.title(f"{ticker} Stock Price History")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_strategy(test_df, actions, rewards, total_return: float):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    prices = test_df["Close"].values[5:]
    dates = test_df.index[5:]

    axes[0].plot(dates, prices, label="Price", alpha=0.7)

    buy_signals = []
    sell_signals = []
    hold_signals = []
    prev_action = None
    for i, a in enumerate(actions):
        if prev_action is not None and a == prev_action:
            hold_signals.append(i)
        elif a == 1:
            buy_signals.append(i)
        elif a == 0:
            sell_signals.append(i)
        prev_action = a

    if buy_signals:
        axes[0].scatter(
            [dates[i] for i in buy_signals],
            [prices[i] for i in buy_signals],
            color="green",
            marker="^",
            s=100,
            label="Buy",
            zorder=5,
        )

    if sell_signals:
        axes[0].scatter(
            [dates[i] for i in sell_signals],
            [prices[i] for i in sell_signals],
            color="red",
            marker="v",
            s=100,
            label="Sell",
            zorder=5,
        )

    if hold_signals:
        axes[0].scatter(
            [dates[i] for i in hold_signals],
            [prices[i] for i in hold_signals],
            color="orange",
            marker="o",
            s=60,
            label="Hold",
            zorder=4,
            alpha=0.7,
        )

    axes[0].set_ylabel("Price ($)")
    axes[0].set_title("Trading Strategy on Test Data")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    cumulative_rewards = np.cumsum(rewards)
    axes[1].plot(dates[1:], cumulative_rewards, label="Agent", color="blue")
    axes[1].axhline(y=0, color="black", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].set_title("Agent Performance Over Time")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    bah_return = ((test_df["Close"].iloc[-1] / test_df["Close"].iloc[0]) - 1) * 100
    return float(bah_return), float(total_return)


def compare_agents(test_df, actions_base, rewards_base, actions_ind, rewards_ind):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    prices = test_df["Close"].values[5:]
    dates = test_df.index[5:]

    axes[0].plot(dates, prices, label="Price", alpha=0.7)

    buy_base = [i for i, a in enumerate(actions_base) if a == 1]
    sell_base = [i for i, a in enumerate(actions_base) if a == 0]

    buy_ind = [i for i, a in enumerate(actions_ind) if a == 1]
    sell_ind = [i for i, a in enumerate(actions_ind) if a == 0]

    if buy_base:
        axes[0].scatter(
            [dates[i] for i in buy_base],
            [prices[i] for i in buy_base],
            color="green",
            marker="^",
            s=80,
            label="Baseline Buy",
            zorder=5,
            alpha=0.6,
        )
    if sell_base:
        axes[0].scatter(
            [dates[i] for i in sell_base],
            [prices[i] for i in sell_base],
            color="red",
            marker="v",
            s=80,
            label="Baseline Sell",
            zorder=5,
            alpha=0.6,
        )

    if buy_ind:
        axes[0].scatter(
            [dates[i] for i in buy_ind],
            [prices[i] for i in buy_ind],
            color="lime",
            marker="P",
            s=60,
            label="Indicators Buy",
            zorder=6,
            alpha=0.9,
        )
    if sell_ind:
        axes[0].scatter(
            [dates[i] for i in sell_ind],
            [prices[i] for i in sell_ind],
            color="magenta",
            marker="X",
            s=60,
            label="Indicators Sell",
            zorder=6,
            alpha=0.9,
        )

    axes[0].set_ylabel("Price ($)")
    axes[0].legend(loc="upper left")

    axes[1].plot(dates[1:], np.cumsum(rewards_base), label="Baseline cumulative reward")
    axes[1].plot(dates, np.cumsum(rewards_ind), label="Indicators cumulative reward")
    axes[1].set_ylabel("Cumulative Reward")
    axes[1].legend()
    plt.show()
