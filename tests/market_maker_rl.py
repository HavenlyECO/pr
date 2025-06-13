import numpy as np
import pandas as pd
import torch
import os
from market_maker_rl import MarketMakerEnv, train_market_maker_rl, rl_adjust_price, QNetwork


def test_env_step_reward():
    # Synthetic odds timeline
    times = pd.date_range("2025-01-01 12:00", periods=5, freq="H")
    prices = [100, 102, 105, 107, 110]
    df = pd.DataFrame({"timestamp": times, "price": prices})
    env = MarketMakerEnv()
    env.reset(df)
    # Try a sequence of "hold" actions
    total_reward = 0
    for _ in range(5):
        obs, reward, done, _ = env.step(0)
        total_reward += reward
        if done:
            break
    # Final reward should be negative abs(predicted closing - true closing)
    assert isinstance(reward, float)
    assert done


def test_training_and_inference(tmp_path):
    # Generate synthetic dataset with 3 episodes
    episodes = []
    for offset in [0, 5, 10]:
        times = pd.date_range("2025-01-01 12:00", periods=6, freq="H")
        prices = [100 + offset + i * 2 for i in range(6)]
        df = pd.DataFrame({"timestamp": times, "price": prices})
        df["volatility"] = np.random.rand(6)
        df["momentum"] = np.random.rand(6)
        episodes.append(df)
    dataset_path = tmp_path / "episodes.pkl"
    import pickle

    with open(dataset_path, "wb") as f:
        pickle.dump(episodes, f)
    # Train RL agent
    model_path = tmp_path / "rl.pt"
    train_market_maker_rl(str(dataset_path), model_out=str(model_path), episodes=20)
    assert os.path.exists(model_path)
    # Test inference
    env = MarketMakerEnv()
    state = env.reset(episodes[0])
    adj = rl_adjust_price(state, model_path=str(model_path))
    assert isinstance(adj, int)
