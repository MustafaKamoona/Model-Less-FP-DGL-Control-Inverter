import numpy as np

def try_import_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        return True, torch, nn, optim
    except Exception:
        return False, None, None, None

class Replay:
    def __init__(self, cap=20000, seed=42):
        self.cap = cap
        self.buf = []
        self.i = 0
        self.rng = np.random.default_rng(seed)
    def push(self, s, a, r, sp, done):
        item = (s, a, r, sp, done)
        if len(self.buf) < self.cap:
            self.buf.append(item)
        else:
            self.buf[self.i] = item
            self.i = (self.i + 1) % self.cap
    def sample(self, n):
        idx = self.rng.choice(len(self.buf), size=n, replace=False)
        batch = [self.buf[i] for i in idx]
        s, a, r, sp, d = map(np.array, zip(*batch))
        return s, a, r, sp, d
    def __len__(self):
        return len(self.buf)

def make_dqn_controller(
    sim,
    P1_kw, P2_kw, P3_24, P4_24,
    actions=None,
    use_forecast_feature=True,
    forecast_mode="next_hour_schedule",
    episodes=80,
    seed=42
):
    """Train a simple DQN and return a controller function.

    RL is model-free: it consumes measurements only.
    Forecast feature (optional) is non-ML (schedule/persistence), not LSTM.
    """
    ok, torch, nn, optim = try_import_torch()
    rng = np.random.default_rng(seed)

    if actions is None:
        actions = np.linspace(-0.008, 0.008, 9)
    actions = np.array(actions, dtype=float)
    nA = len(actions)

    def simple_forecast_scalar(hour):
        total_24 = (P1_kw + P2_kw) + P3_24 + P4_24
        if forecast_mode == "persistence":
            return float(total_24[hour])
        nxt = min(23, hour+1)
        return float(total_24[nxt])

    def obs_to_vec(V_bus, Iinv_pu, P_total_pu, hour):
        vec = [*V_bus.tolist(), float(Iinv_pu), float(P_total_pu), hour/23.0]
        if use_forecast_feature:
            f = simple_forecast_scalar(int(hour))
            vec.append(f / sim.S_base_kva)  # normalize
        return np.array(vec, dtype=np.float32)

    if not ok:
        # fallback rule controller
        def fallback(obs):
            e = 1.0 - obs["V_avg"]
            return float(np.clip(1.0 + 0.8*e, sim.Vref_min, sim.Vref_max))
        return fallback

    class QNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, out_dim)
            )
        def forward(self, x):
            return self.net(x)

    # determine input dim
    test_vec = obs_to_vec(np.ones(4), 1.0, 1.0, 0)
    in_dim = int(test_vec.shape[0])

    q = QNet(in_dim, nA)
    qt = QNet(in_dim, nA)
    qt.load_state_dict(q.state_dict())
    opti = optim.Adam(q.parameters(), lr=2e-3)
    rb = Replay(cap=25000, seed=seed)

    gamma = 0.98
    batch = 256
    eps0, eps1 = 1.0, 0.05

    def reward_fn(V_bus, Iinv_pu, dV):
        v_pen = float(np.mean(np.abs(V_bus - 1.0)))
        i_pen = float(max(0.0, Iinv_pu - sim.I_max_pu))
        u_pen = float(abs(dV))
        return float(-(6.0*v_pen + 8.0*i_pen + 0.6*u_pen))

    base_P3 = np.array(P3_24, dtype=float)
    base_P4 = np.array(P4_24, dtype=float)

    def sample_day_profiles():
        s = float(np.clip(rng.normal(1.0, 0.10), 0.8, 1.2))
        n3 = base_P3 * s * (1.0 + rng.normal(0.0, 0.04, size=24))
        n4 = base_P4 * s * (1.0 + rng.normal(0.0, 0.04, size=24))
        return np.clip(n3, 0, None), np.clip(n4, 0, None)

    for ep in range(episodes):
        eps = eps0 + (eps1-eps0) * (ep / max(1, episodes-1))
        P3_day, P4_day = sample_day_profiles()

        V_bus = np.ones(4, dtype=float) * 1.0
        Vref = 1.0

        for hour in range(24):
            P_kw_vec = np.array([P1_kw, P2_kw, float(P3_day[hour]), float(P4_day[hour])], dtype=float)
            P_pu_vec = sim.kw_to_pu(P_kw_vec)
            P_total_pu = float(np.sum(P_pu_vec))

            for k in range(sim.steps_per_hour):
                V_avg = float(np.mean(V_bus))
                Iinv_pu = float(P_total_pu / max(V_avg, 0.7))

                s = obs_to_vec(V_bus, Iinv_pu, P_total_pu, hour)

                if rng.random() < eps:
                    a_idx = int(rng.integers(0, nA))
                else:
                    with torch.no_grad():
                        a_idx = int(torch.argmax(q(torch.tensor(s)[None, :])).item())

                dV = float(actions[a_idx])
                Vref = float(np.clip(Vref + dV, sim.Vref_min, sim.Vref_max))

                V_source = Vref - sim.k_droop * P_total_pu
                V_targets = V_source - sim.k_drop * P_pu_vec
                V_bus = V_bus + (sim.dt / sim.tau_v) * (V_targets - V_bus)

                V_avg2 = float(np.mean(V_bus))
                Iinv2 = float(P_total_pu / max(V_avg2, 0.7))
                sp = obs_to_vec(V_bus, Iinv2, P_total_pu, hour)

                r = reward_fn(V_bus, Iinv2, dV)
                done = (hour == 23 and k == sim.steps_per_hour-1)
                rb.push(s, a_idx, r, sp, done)

                if len(rb) >= batch:
                    S, A, R, SP, D = rb.sample(batch)
                    S_t = torch.tensor(S)
                    A_t = torch.tensor(A, dtype=torch.int64)[:, None]
                    R_t = torch.tensor(R, dtype=torch.float32)[:, None]
                    SP_t = torch.tensor(SP)
                    D_t = torch.tensor(D.astype(np.float32))[:, None]

                    q_sa = q(S_t).gather(1, A_t)
                    with torch.no_grad():
                        max_qsp = torch.max(qt(SP_t), dim=1, keepdim=True)[0]
                        y = R_t + gamma * (1.0 - D_t) * max_qsp
                    loss = torch.mean((q_sa - y)**2)
                    opti.zero_grad()
                    loss.backward()
                    opti.step()

        if (ep+1) % 10 == 0:
            qt.load_state_dict(q.state_dict())

    # inference controller (maintains internal Vref state)
    Vref_state = 1.0
    def ctrl(obs):
        nonlocal Vref_state
        V_bus = obs["V_bus"]
        Iinv_pu = obs["Iinv_pu"]
        P_total_pu = obs["P_total_pu"]
        hour = obs["hour"]
        s = obs_to_vec(V_bus, Iinv_pu, P_total_pu, hour)
        with torch.no_grad():
            a_idx = int(torch.argmax(q(torch.tensor(s)[None, :])).item())
        dV = float(actions[a_idx])
        Vref_state = float(np.clip(obs["Vref"] + dV, sim.Vref_min, sim.Vref_max))
        return Vref_state

    return ctrl
