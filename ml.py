import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# MVP: strict minimal feature set, no fallback logic

# 1. TRAINING PHASE

df = pd.read_csv("retrosheet_training_data.csv")
X = df[["price1", "price2"]]
y = df["home_team_win"]
model = LogisticRegression()
model.fit(X, y)

with open("mvp_model.pkl", "wb") as f:
    pickle.dump(model, f)

# 2. PREDICTION PHASE (example)
with open("mvp_model.pkl", "rb") as f:
    model = pickle.load(f)

# Predict for a new game (replace with real data)
new_game = pd.DataFrame([{"price1": -120, "price2": 110}])
win_prob = model.predict_proba(new_game)[:, 1][0]
print("Win probability:", win_prob)
