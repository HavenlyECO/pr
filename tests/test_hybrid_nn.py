import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hybrid_nn import HybridNet, train_hybrid_model, predict_hybrid_probability


def test_hybrid_shapes_and_training(tmp_path):
    # Synthetic data: 10 events, 5 fund features, 12-timestep market sequence (single feature)
    N, D1, T, D2 = 10, 5, 12, 1
    fund_X = np.random.randn(N, D1).astype(np.float32)
    market_X = np.random.randn(N, T, D2).astype(np.float32)
    y = np.random.binomial(1, 0.5, N).astype(np.float32)
    npz_path = tmp_path / "hybrid_train.npz"
    np.savez(npz_path, fund_X=fund_X, market_X=market_X, y=y)
    model_path = tmp_path / "hybrid_model.pt"
    train_hybrid_model(str(npz_path), model_out=str(model_path), epochs=3, batch_size=2)
    assert os.path.exists(model_path)

    # Test predict_hybrid_probability
    out = predict_hybrid_probability(fund_X[0], market_X[0], model_path=str(model_path))
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0

    # Test output shape
    m = HybridNet(fund_in_dim=D1, market_in_dim=D2, market_seq_len=T)
    fund_b = torch.tensor(fund_X[:2])
    market_b = torch.tensor(market_X[:2])
    out_tensor = m(fund_b, market_b)
    assert out_tensor.shape == (2,)
