import numpy as np

def build_training_days(P1_kw, P2_kw, P3_24, P4_24, num_days=260, noise=0.08, seed=42):
    rng = np.random.default_rng(seed)
    base_total = (P1_kw + P2_kw) + P3_24 + P4_24
    days = []
    for _ in range(num_days):
        scale = float(np.clip(rng.normal(1.0, 0.12), 0.75, 1.25))
        seq = base_total * scale
        seq = seq * (1.0 + rng.normal(0.0, noise, size=seq.shape))
        seq = np.clip(seq, 0.0, None)
        days.append(seq.astype(np.float32))
    return np.stack(days, axis=0)  # [D,24]

def try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return True, torch, nn
    except Exception:
        return False, None, None

def train_lstm_forecaster(P1_kw, P2_kw, P3_24, P4_24, epochs=120, lr=2e-3, lookback=6):
    ok, torch, nn = try_import_torch()
    if not ok:
        return None, lookback, "persistence"

    data = build_training_days(P1_kw, P2_kw, P3_24, P4_24)
    X, Y = [], []
    for day in data:
        for t in range(lookback, 23):
            X.append(day[t-lookback:t])
            Y.append(day[t])
    X = np.array(X)[:, :, None]  # [N,lookback,1]
    Y = np.array(Y)[:, None]     # [N,1]

    class LoadLSTM(nn.Module):
        def __init__(self, hidden=48):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, 1)
        def forward(self, x):
            y, _ = self.lstm(x)
            return self.fc(y[:, -1, :])

    model = LoadLSTM()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X, dtype=torch.float32)
    Y_t = torch.tensor(Y, dtype=torch.float32)

    model.train()
    for ep in range(epochs):
        opt.zero_grad()
        pred = model(X_t)
        loss = loss_fn(pred, Y_t)
        loss.backward()
        opt.step()
    return model, lookback, "lstm"

def forecast_next_total_load(hour, P1_kw, P2_kw, P3_24, P4_24, model, lookback):
    total_24 = (P1_kw + P2_kw) + P3_24 + P4_24
    if hour == 0:
        hist = [float(total_24[0])] * lookback
    else:
        start = max(0, hour - lookback)
        hist = list(map(float, total_24[start:hour]))
        if len(hist) < lookback:
            hist = [hist[0]]*(lookback-len(hist)) + hist

    ok, torch, _ = try_import_torch()
    if (model is None) or (not ok):
        return {"yhat_total_kw": float(hist[-1]), "mode": "persistence"}

    with torch.no_grad():
        x = torch.tensor(np.array(hist, dtype=np.float32)[None, :, None])
        yhat = float(model(x).item())
    return {"yhat_total_kw": yhat, "mode": "lstm"}

def make_lstm_baseline_controller(S_base_kva, V0=1.0, k=0.0018):
    def ctrl(obs):
        f = obs["forecast"]
        P_total_kw = float(obs["P_total_pu"] * S_base_kva)
        yhat = float(f["yhat_total_kw"])
        Vref = V0 + k*(yhat - P_total_kw)
        return Vref
    return ctrl
