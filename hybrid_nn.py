import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class FundamentalBranch(nn.Module):
    def __init__(self, input_dim, hidden_dims=(32, 32)):
        super().__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            last_dim = h
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MarketBranch(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.hidden_dim = hidden_dim

    def forward(self, seq):
        # seq: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(seq)
        return h_n[-1]  # (batch, hidden_dim)

class HybridNet(nn.Module):
    def __init__(self, fund_in_dim, market_in_dim, market_seq_len, fund_hidden=(32, 32), market_hidden=32, out_hidden=32):
        super().__init__()
        self.fund_branch = FundamentalBranch(fund_in_dim, fund_hidden)
        self.market_branch = MarketBranch(market_in_dim, market_hidden)
        concat_dim = fund_hidden[-1] + market_hidden
        self.final = nn.Sequential(
            nn.Linear(concat_dim, out_hidden),
            nn.ReLU(),
            nn.Linear(out_hidden, 1)
        )

    def forward(self, fund_x, market_seq):
        fund_feat = self.fund_branch(fund_x)
        market_feat = self.market_branch(market_seq)
        x = torch.cat([fund_feat, market_feat], dim=-1)
        out = self.final(x)
        return torch.sigmoid(out).squeeze(-1)

def train_hybrid_model(dataset_path, model_out="hybrid_model.pt", epochs=30, batch_size=32, lr=1e-3, device=None, val_frac=0.15):
    """
    dataset_path: .npz with arrays: fund_X (N, D1), market_X (N, T, D2), y (N,)
    """
    data = np.load(dataset_path)
    fund_X = data["fund_X"]
    market_X = data["market_X"]
    y = data["y"]
    N = fund_X.shape[0]
    idx = np.arange(N)
    np.random.shuffle(idx)
    split = int(N * (1 - val_frac))
    train_idx, val_idx = idx[:split], idx[split:]
    fund_X_train, fund_X_val = fund_X[train_idx], fund_X[val_idx]
    market_X_train, market_X_val = market_X[train_idx], market_X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridNet(
        fund_in_dim=fund_X.shape[1],
        market_in_dim=market_X.shape[2],
        market_seq_len=market_X.shape[1]
    ).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    def make_batches(X1, X2, y, bs):
        n = X1.shape[0]
        for i in range(0, n, bs):
            yield (
                torch.tensor(X1[i:i+bs]).float().to(device),
                torch.tensor(X2[i:i+bs]).float().to(device),
                torch.tensor(y[i:i+bs]).float().to(device)
            )

    for epoch in range(epochs):
        model.train()
        losses = []
        for fund_b, market_b, y_b in make_batches(fund_X_train, market_X_train, y_train, batch_size):
            opt.zero_grad()
            y_hat = model(fund_b, market_b)
            loss = loss_fn(y_hat, y_b)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_preds = []
            for fund_b, market_b, y_b in make_batches(fund_X_val, market_X_val, y_val, batch_size):
                val_preds.append(model(fund_b, market_b).cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_loss = np.mean((val_preds - y_val)**2)
        print(f"Epoch {epoch+1} | Train loss {np.mean(losses):.4f} | Val MSE {val_loss:.4f}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "fund_in_dim": fund_X.shape[1],
        "market_in_dim": market_X.shape[2],
        "market_seq_len": market_X.shape[1],
    }, model_out)
    print(f"Saved hybrid model to {model_out}")

def load_hybrid_model(model_path, device=None):
    checkpoint = torch.load(model_path, map_location=device or "cpu")
    model = HybridNet(
        fund_in_dim=checkpoint['fund_in_dim'],
        market_in_dim=checkpoint['market_in_dim'],
        market_seq_len=checkpoint['market_seq_len']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def predict_hybrid_probability(fund_features, odds_sequence, model_path="hybrid_model.pt", device=None):
    """
    fund_features: (D1,) array-like
    odds_sequence: (T, D2) array-like (normalized)
    """
    model = load_hybrid_model(model_path, device=device)
    device = device or "cpu"
    with torch.no_grad():
        fund_x = torch.tensor(fund_features).float().unsqueeze(0).to(device)
        market_x = torch.tensor(odds_sequence).float().unsqueeze(0).to(device)
        prob = model(fund_x, market_x).cpu().item()
    return float(prob)
