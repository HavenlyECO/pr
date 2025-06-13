import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import Dataset, DataLoader

class OddsSequenceDataset(Dataset):
    """Dataset of odds sequences for autoencoder training."""
    def __init__(self, sequences, seq_len=None):
        """
        sequences: list of 1D np.arrays (price/time series, already normalized)
        seq_len: pad/truncate all sequences to this length (if None, use max)
        """
        if seq_len is None:
            seq_len = max(len(seq) for seq in sequences)
        self.seq_len = seq_len
        self.data = []
        for seq in sequences:
            seq = np.asarray(seq, dtype=np.float32)
            if len(seq) < seq_len:
                # Pad with last value
                pad_len = seq_len - len(seq)
                seq = np.concatenate([seq, np.full(pad_len, seq[-1] if len(seq) else 0.0)])
            elif len(seq) > seq_len:
                seq = seq[:seq_len]
            self.data.append(seq)
        self.data = np.stack(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return as shape (seq_len, 1)
        return torch.tensor(self.data[idx]).unsqueeze(-1), torch.tensor(self.data[idx]).unsqueeze(-1)

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len, latent_dim=8, hidden_dim=32):
        super().__init__()
        self.encoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, 1)
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        # x: (batch, seq_len, 1)
        enc_out, (h, c) = self.encoder(x)
        # Use last hidden state as summary
        z = self.enc_to_latent(h[-1])  # (batch, latent_dim)
        # Repeat latent for each timestep to initialize decoder
        dec_init = self.latent_to_dec(z).unsqueeze(0)  # (1, batch, hidden_dim)
        dec_in = torch.zeros(x.size(0), self.seq_len, 1, device=x.device)
        dec_out, _ = self.decoder(dec_in, (dec_init, torch.zeros_like(dec_init)))
        out = self.out_proj(dec_out)  # (batch, seq_len, 1)
        return out, z

    def encode(self, x):
        with torch.no_grad():
            enc_out, (h, c) = self.encoder(x)
            z = self.enc_to_latent(h[-1])
        return z

def train_sequence_autoencoder(dataset_path, model_out="odds_autoencoder.pt", latent_dim=8, hidden_dim=32,
                              epochs=40, batch_size=32, lr=1e-3, device=None):
    # Load dataset: expect pickled list of odds DataFrames, each with 'price' and 'timestamp'
    if dataset_path.endswith(".pkl"):
        with open(dataset_path, "rb") as f:
            odds_dfs = pickle.load(f)
    else:
        raise ValueError("Only .pkl format supported for odds timeline dataset")
    # Prepare normalized price sequences
    sequences = []
    for df in odds_dfs:
        prices = np.asarray(df['price'], dtype=np.float32)
        # Normalize price sequence
        mean, std = prices.mean(), prices.std() if prices.std() > 1e-5 else 1.0
        norm_prices = (prices - mean) / std
        sequences.append(norm_prices)
    seq_len = max(len(seq) for seq in sequences)
    dataset = OddsSequenceDataset(sequences, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMAutoencoder(seq_len=seq_len, latent_dim=latent_dim, hidden_dim=hidden_dim)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            y_hat, _ = model(x)
            loss = loss_fn(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss: {epoch_loss / len(loader):.4f}")
    torch.save({'model_state_dict': model.state_dict(), 'seq_len': seq_len, 'latent_dim': latent_dim, 'hidden_dim': hidden_dim}, model_out)
    print(f"Saved autoencoder to {model_out}")
    return model

def load_autoencoder(model_path, device=None):
    checkpoint = torch.load(model_path, map_location=device or "cpu")
    model = LSTMAutoencoder(seq_len=checkpoint['seq_len'], latent_dim=checkpoint['latent_dim'], hidden_dim=checkpoint['hidden_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def encode_odds_sequence(df, price_col, model_path="odds_autoencoder.pt", device=None):
    """
    Returns a 1D np.array (latent vector) for the given odds timeline.
    """
    model = load_autoencoder(model_path, device=device)
    prices = np.asarray(df[price_col], dtype=np.float32)
    # Normalize as during training
    mean, std = prices.mean(), prices.std() if prices.std() > 1e-5 else 1.0
    norm_prices = (prices - mean) / std
    # Pad/truncate to seq_len
    seq_len = model.seq_len
    if len(norm_prices) < seq_len:
        pad_len = seq_len - len(norm_prices)
        norm_prices = np.concatenate([norm_prices, np.full(pad_len, norm_prices[-1] if len(norm_prices) else 0.0)])
    elif len(norm_prices) > seq_len:
        norm_prices = norm_prices[:seq_len]
    x = torch.tensor(norm_prices).float().unsqueeze(0).unsqueeze(-1)
    z = model.encode(x)
    return z.squeeze(0).cpu().numpy()
