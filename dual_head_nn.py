import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


class BettingDataset(Dataset):
    """Dataset for dual-head training."""

    def __init__(self, df, market_features, team_features, closing_line_col, outcome_col):
        self.market_X = df[market_features].values.astype(np.float32)
        self.team_X = df[team_features].values.astype(np.float32) if team_features else None
        self.closing_line = df[closing_line_col].values.astype(np.float32)
        self.outcome = df[outcome_col].values
        if len(self.outcome.shape) == 1:
            self.outcome = self.outcome.astype(np.int64)
        else:
            self.outcome = self.outcome.astype(np.float32)

    def __len__(self):
        return len(self.closing_line)

    def __getitem__(self, idx):
        x_market = self.market_X[idx]
        x_team = self.team_X[idx] if self.team_X is not None else None
        y_closing_line = self.closing_line[idx]
        y_outcome = self.outcome[idx]
        if x_team is not None:
            x = np.concatenate([x_market, x_team])
        else:
            x = x_market
        return torch.tensor(x), torch.tensor(y_closing_line), torch.tensor(y_outcome)


class DualHeadNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, outcome_classes=2):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.market_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.outcome_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, outcome_classes),
        )

    def forward(self, x):
        shared = self.shared(x)
        closing_pred = self.market_head(shared).squeeze(-1)
        outcome_logits = self.outcome_head(shared)
        return closing_pred, outcome_logits


def train_dual_head(
    df,
    market_features,
    team_features,
    closing_line_col,
    outcome_col,
    *,
    outcome_classes=2,
    epochs=50,
    batch_size=128,
    lr=1e-3,
    market_loss_weight=1.0,
    outcome_loss_weight=1.0,
    val_split=0.2,
    verbose=1,
    device=None,
):
    dataset = BettingDataset(df, market_features, team_features, closing_line_col, outcome_col)
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = len(market_features) + (len(team_features) if team_features else 0)
    model = DualHeadNet(input_dim, hidden_dim=128, outcome_classes=outcome_classes)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    market_loss_fn = nn.MSELoss()
    if outcome_classes == 2:
        outcome_loss_fn = nn.BCEWithLogitsLoss()
    else:
        outcome_loss_fn = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y_market, y_outcome in train_loader:
            x = x.to(device)
            y_market = y_market.to(device)
            y_outcome = y_outcome.to(device)
            optimizer.zero_grad()
            pred_market, pred_outcome = model(x)
            market_loss = market_loss_fn(pred_market, y_market)
            if outcome_classes == 2:
                y_outcome = y_outcome.float()
                out_loss = outcome_loss_fn(pred_outcome.squeeze(), y_outcome)
            else:
                y_outcome = y_outcome.long()
                out_loss = outcome_loss_fn(pred_outcome, y_outcome)
            loss = market_loss_weight * market_loss + outcome_loss_weight * out_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose:
            print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")
            evaluate_dual_head(model, val_loader, outcome_classes, device)
    return model


def evaluate_dual_head(model, loader, outcome_classes, device):
    model.eval()
    market_losses = []
    outcome_preds = []
    outcome_targets = []
    market_targets = []
    market_loss_fn = nn.MSELoss(reduction='sum')
    if outcome_classes == 2:
        outcome_loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
    else:
        outcome_loss_fn = nn.CrossEntropyLoss(reduction='sum')
    with torch.no_grad():
        for x, y_market, y_outcome in loader:
            x = x.to(device)
            y_market = y_market.to(device)
            y_outcome = y_outcome.to(device)
            pred_market, pred_outcome = model(x)
            market_loss = market_loss_fn(pred_market, y_market)
            market_losses.append(market_loss.item())
            market_targets.extend(y_market.cpu().numpy())
            if outcome_classes == 2:
                probs = torch.sigmoid(pred_outcome.squeeze()).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                outcome_preds.extend(preds)
                outcome_targets.extend(y_outcome.cpu().numpy())
            else:
                preds = torch.argmax(pred_outcome, dim=-1).cpu().numpy()
                outcome_preds.extend(preds)
                outcome_targets.extend(y_outcome.cpu().numpy())
    mse = np.mean(market_losses) / len(loader.dataset)
    acc = np.mean(np.array(outcome_preds) == np.array(outcome_targets))
    print(f"Validation: Market MSE={mse:.4f} | Outcome Acc={acc:.3f}")


if __name__ == "__main__":
    df = pd.read_csv("dual_head_training_data.csv")
    market_features = [
        "opening_odds",
        "line_move",
        "volatility",
        "timestamp_open",
        "timestamp_close",
    ]
    team_features = ["home_rating", "away_rating"]
    closing_line_col = "closing_odds"
    outcome_col = "outcome"
    outcome_classes = 2

    train_dual_head(
        df,
        market_features,
        team_features,
        closing_line_col,
        outcome_col,
        outcome_classes=outcome_classes,
        epochs=30,
        batch_size=128,
        lr=1e-3,
        market_loss_weight=1.0,
        outcome_loss_weight=1.0,
        val_split=0.2,
        verbose=1,
    )

