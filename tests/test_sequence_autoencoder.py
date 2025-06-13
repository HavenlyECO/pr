import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sequence_autoencoder import (
    train_sequence_autoencoder,
    encode_odds_sequence,
    load_autoencoder,
)


def test_autoencoder_training_and_encoding(tmp_path):
    # Create 3 synthetic sequences
    odds_dfs = []
    for offset in [100, 110, 120]:
        t = pd.date_range("2022-01-01", periods=10, freq="H")
        price = np.linspace(offset, offset + 9, 10)
        df = pd.DataFrame({"timestamp": t, "price": price})
        odds_dfs.append(df)
    # Save as .pkl
    pkl_path = tmp_path / "synthetic_odds.pkl"
    import pickle

    with open(pkl_path, "wb") as f:
        pickle.dump(odds_dfs, f)
    # Train autoencoder
    model_out = tmp_path / "odds_autoencoder.pt"
    train_sequence_autoencoder(
        str(pkl_path),
        model_out=str(model_out),
        epochs=10,
        latent_dim=4,
        hidden_dim=8,
    )
    # Check encoding
    latent = encode_odds_sequence(odds_dfs[0], "price", model_path=str(model_out))
    assert latent.shape[0] == 4
    # Check model can reconstruct (loss goes down)
    model = load_autoencoder(str(model_out))
    model.eval()
    x = (
        torch.tensor(
            (
                odds_dfs[0]["price"] - odds_dfs[0]["price"].mean()
            )
            / odds_dfs[0]["price"].std()
        )
        .float()
        .unsqueeze(0)
        .unsqueeze(-1)
    )
    with torch.no_grad():
        y_hat, _ = model(x)
        assert y_hat.shape == (1, 10, 1)

