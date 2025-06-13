import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class MarketMakerEnv:
    """Simple environment for learning line moves."""

    def __init__(self, price_step: int = 5, max_steps: int = 20) -> None:
        self.price_step = price_step
        self.max_steps = max_steps
        self.reset_called = False

    def reset(self, odds_df: pd.DataFrame) -> np.ndarray:
        """Initialize the environment state from ``odds_df``."""
        self.odds_df = odds_df.reset_index(drop=True)
        self.t0 = self.odds_df["timestamp"].iloc[0]
        self.tN = self.odds_df["timestamp"].iloc[-1]
        self.gt_closing = self.odds_df["price"].iloc[-1]
        self.step_idx = 0
        self.cur_price = self.odds_df["price"].iloc[0]
        self.cur_time = self.odds_df["timestamp"].iloc[0]
        self.done = False
        self.reset_called = True
        self._update_features()
        return self.observation()

    def _update_features(self) -> None:
        self.time_to_game = (self.tN - self.cur_time).total_seconds()
        idx = self.step_idx
        self.volatility = self.odds_df.get("volatility", pd.Series(0.0)).iloc[idx]
        self.momentum = self.odds_df.get("momentum", pd.Series(0.0)).iloc[idx]

    def step(self, action: int):
        assert self.reset_called, "Call reset first!"
        if self.done:
            raise RuntimeError("Episode is done")
        self.cur_price += self.price_step * action
        self.step_idx += 1
        if self.step_idx >= len(self.odds_df):
            self.done = True
            reward = -abs(self.cur_price - self.gt_closing)
        else:
            self.cur_time = self.odds_df["timestamp"].iloc[self.step_idx]
            self._update_features()
            reward = 0.0
        return self.observation(), reward, self.done, {}

    def observation(self) -> np.ndarray:
        return np.array(
            [
                self.time_to_game / 3600.0,
                self.cur_price / 100.0,
                self.volatility,
                self.momentum,
            ],
            dtype=np.float32,
        )


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(x)


def train_market_maker_rl(
    dataset_path: str,
    model_out: str = "market_maker_rl.pt",
    episodes: int = 500,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-3,
    epsilon_start: float = 1.0,
    epsilon_final: float = 0.1,
    epsilon_decay: float = 0.995,
) -> None:
    """Train a DQN agent to mimic bookmaker line moves."""

    if dataset_path.endswith(".pkl"):
        import pickle
        with open(dataset_path, "rb") as f:
            episodes_data = pickle.load(f)
    else:
        df = pd.read_csv(dataset_path, parse_dates=["timestamp"])
        episodes_data = [g for _, g in df.groupby("event_id")]

    env = MarketMakerEnv()
    state_size = len(env.reset(episodes_data[0]))
    action_space = [-2, -1, 0, 1, 2]
    action_size = len(action_space)
    qnet = QNetwork(state_size, action_size)
    optimizer = optim.Adam(qnet.parameters(), lr=lr)
    memory: deque = deque(maxlen=5000)
    epsilon = epsilon_start

    def select_action(state: np.ndarray, eps: float) -> int:
        if random.random() < eps:
            return random.randint(0, action_size - 1)
        with torch.no_grad():
            qs = qnet(torch.tensor(state).float().unsqueeze(0))
            return int(qs.argmax().item())

    for ep in range(episodes):
        odds_df = random.choice(episodes_data)
        state = env.reset(odds_df)
        total_reward = 0.0
        for _ in range(env.max_steps):
            action_idx = select_action(state, epsilon)
            action = action_space[action_idx]
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action_idx, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break

        if len(memory) >= batch_size:
            batch = random.sample(memory, batch_size)
            states = torch.tensor([b[0] for b in batch], dtype=torch.float32)
            actions = torch.tensor([b[1] for b in batch], dtype=torch.int64)
            rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
            next_states = torch.tensor([b[3] for b in batch], dtype=torch.float32)
            dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

            q_values = qnet(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = qnet(next_states).max(1)[0].detach()
            target = rewards + (1 - dones) * gamma * next_q
            loss = nn.MSELoss()(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if ep % 20 == 0:
            print(f"Episode {ep} | Reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")
        epsilon = max(epsilon * epsilon_decay, epsilon_final)

    torch.save(qnet.state_dict(), model_out)
    print(f"Saved RL policy to {model_out}")


def rl_adjust_price(current_state: np.ndarray, model_path: str = "market_maker_rl.pt") -> int:
    """Load the trained agent and return the recommended price adjustment."""
    action_space = [-2, -1, 0, 1, 2]
    qnet = QNetwork(state_size=len(current_state), action_size=len(action_space))
    qnet.load_state_dict(torch.load(model_path, map_location="cpu"))
    qnet.eval()
    with torch.no_grad():
        qs = qnet(torch.tensor(current_state).float().unsqueeze(0))
        action_idx = int(qs.argmax().item())
    return action_space[action_idx]


def convert_odds_timelines_to_episodes(cache_path: str):
    """Return a list of odds timelines loaded from pickled files."""
    import os
    import pickle
    import glob

    episodes = []
    for pkl_file in glob.glob(os.path.join(cache_path, "*.pkl")):
        with open(pkl_file, "rb") as f:
            odds_df = pickle.load(f)
            episodes.append(odds_df)
    return episodes
