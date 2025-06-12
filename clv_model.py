import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


def american_odds_to_prob(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)


def remove_vig(imp_prob_1, imp_prob_2):
    """Remove vigorish from a pair of implied probabilities, returning fair probabilities."""
    overround = imp_prob_1 + imp_prob_2
    return imp_prob_1 / overround, imp_prob_2 / overround


class CLVDataset(Dataset):
    """Dataset for CLV training."""

    def __init__(self, df, feature_cols, bet_odds_col, closing_odds_col, outcome_col):
        self.X = df[feature_cols].values.astype(np.float32)
        self.bet_prob = df[bet_odds_col].apply(american_odds_to_prob).values.astype(np.float32)
        self.close_prob = df[closing_odds_col].apply(american_odds_to_prob).values.astype(np.float32)
        self.y = df[outcome_col].values.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.bet_prob[idx]),
            torch.tensor(self.close_prob[idx]),
            torch.tensor(self.y[idx]),
        )

class CLVNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.prob_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.outcome_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x):
        shared = self.shared(x)
        pred_prob = self.prob_head(shared).squeeze(-1)
        pred_outcome = self.outcome_head(shared).squeeze(-1)
        return pred_prob, pred_outcome

def train_clv_model(
    df,
    feature_cols,
    bet_odds_col,
    closing_odds_col,
    outcome_col,
    epochs=40,
    batch_size=128,
    lr=1e-3,
    clv_loss_weight=1.0,
    outcome_loss_weight=0.5,
    val_split=0.2,
    verbose=1,
    device=None,
):
    dataset = CLVDataset(df, feature_cols, bet_odds_col, closing_odds_col, outcome_col)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(feature_cols)
    model = CLVNet(input_dim)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    clv_loss_fn = nn.MSELoss()
    outcome_loss_fn = nn.BCELoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for X, bet_prob, close_prob, y in train_loader:
            X = X.to(device)
            close_prob = close_prob.to(device)
            y = y.float().to(device)
            optimizer.zero_grad()
            pred_prob, pred_outcome = model(X)
            clv_loss = clv_loss_fn(pred_prob, close_prob)
            outcome_loss = outcome_loss_fn(pred_outcome, y)
            loss = clv_loss_weight * clv_loss + outcome_loss_weight * outcome_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")
            evaluate_clv_model(model, val_loader, device)
    return model

def evaluate_clv_model(model, loader, device):
    model.eval()
    pred_probs, close_probs, bet_probs, y_true, pred_outcomes = [], [], [], [], []
    with torch.no_grad():
        for X, bet_prob, close_prob, y in loader:
            X = X.to(device)
            bet_prob = bet_prob.cpu().numpy()
            close_prob = close_prob.cpu().numpy()
            y = y.cpu().numpy()
            pred_prob, pred_outcome = model(X)
            pred_probs.extend(pred_prob.cpu().numpy())
            close_probs.extend(close_prob)
            bet_probs.extend(bet_prob)
            y_true.extend(y)
            pred_outcomes.extend((pred_outcome.cpu().numpy() > 0.5).astype(int))
    pred_probs = np.array(pred_probs)
    close_probs = np.array(close_probs)
    bet_probs = np.array(bet_probs)
    y_true = np.array(y_true)
    pred_outcomes = np.array(pred_outcomes)
    mean_clv = np.mean(pred_probs - close_probs)
    pct_beating_market = np.mean(pred_probs > close_probs)
    acc = np.mean(pred_outcomes == y_true)
    print(
        f"Val: Mean CLV={mean_clv:.4f} | %Beating Market={pct_beating_market:.2%} | Outcome Acc={acc:.3f}"
    )

if __name__ == "__main__":
    df = pd.read_csv("clv_training_data.csv")
    feature_cols = [
        "opening_odds",
        "line_move",
        "volatility",
        "home_rating",
        "away_rating",
    ]
    bet_odds_col = "bet_odds"
    closing_odds_col = "closing_odds"
    outcome_col = "outcome"
    train_clv_model(
        df,
        feature_cols,
        bet_odds_col,
        closing_odds_col,
        outcome_col,
        epochs=40,
        batch_size=128,
        lr=1e-3,
        clv_loss_weight=1.0,
        outcome_loss_weight=0.25,
        val_split=0.2,
        verbose=1,
    )

