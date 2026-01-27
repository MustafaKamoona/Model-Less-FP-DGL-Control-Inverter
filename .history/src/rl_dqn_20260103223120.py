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
    seed=42,
    print_every=5,
    # --- NEW: smoothing / rate limiting knobs ---
    vref_smooth_alpha=0.85,      # closer to 1 => smoother (recommended 0.80–0.95)
    max_delta_per_step=0.004,    # hard limit on |ΔV| per minute step
):
    """
    Train a simple DQN and return:
      - controller_fn(obs)-> Vref (pu)
      - training_logs dict (episode_reward, episode_voltage_deviation)

    Notes:
    - RL is model-free: consumes measurements only.
    - Forecast feature (optional) is non-ML (schedule/persistence), not LSTM.
    - Actions are ΔV adjustments to inverter voltage reference Vref (Option A).
    - Includes smoothing/rate-limiting to reduce aggressive control effort.
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
        nxt = min(23, hour + 1)
        return float(total_24[nxt])

    def obs_to_vec(V_bus, Iinv_pu, P_total_pu, hour):
        vec = [*V_bus.tolist(), float(Iinv_pu), float(P_total_pu), hour / 23.0]
        if use_forecast_feature:
            f = simple_forecast_scalar(int(hour))
            vec.append(f / sim.S_base_kva)  # normalize kW by base
        return np.array(vec, dtype=np.float32)

    if not ok:
        # fallback rule controller
        def fallback(obs):
            e = 1.0 - obs["V_avg"]
            return float(np.clip(1.0 + 0.8 * e, sim.Vref_min, sim.Vref_max))

        training_logs = {
            "episode_reward": np.array([]),
            "episode_voltage_deviation": np.array([]),
            "note": "PyTorch not available; returned fallback controller."
        }
        return fallback, training_logs

    class QNet(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 48),
                nn.ReLU(),
                nn.Linear(48, 48),
                nn.ReLU(),
                nn.Linear(48, out_dim),
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

    def reward_fn(V_bus, Iinv_pu, dV_used):
        # Penalize voltage deviation, overcurrent, and control movement
        v_pen = float(np.mean(np.abs(V_bus - 1.0)))
        i_pen = float(max(0.0, Iinv_pu - sim.I_max_pu))
        u_pen = float(abs(dV_used))
        return float(-(6.0 * v_pen + 8.0 * i_pen + 0.8 * u_pen))

    # Load randomization (domain randomization)
    base_P3 = np.array(P3_24, dtype=float)
    base_P4 = np.array(P4_24, dtype=float)

    def sample_day_profiles():
        s = float(np.clip(rng.normal(1.0, 0.10), 0.8, 1.2))
        n3 = base_P3 * s * (1.0 + rng.normal(0.0, 0.04, size=24))
        n4 = base_P4 * s * (1.0 + rng.normal(0.0, 0.04, size=24))
        return np.clip(n3, 0, None), np.clip(n4, 0, None)

    def apply_vref_update(vref_prev, dv_action):
        """Apply rate-limit + smoothing, then clamp to bounds."""
        # hard rate limit
        dv = float(np.clip(dv_action, -max_delta_per_step, max_delta_per_step))
        vref_raw = vref_prev + dv

        # smoothing: vref_new = alpha*vref_prev + (1-alpha)*vref_raw
        alpha = float(np.clip(vref_smooth_alpha, 0.0, 0.9999))
        vref_new = alpha * vref_prev + (1.0 - alpha) * vref_raw

        # clamp
        vref_new = float(np.clip(vref_new, sim.Vref_min, sim.Vref_max))
        return vref_new, dv  # return dv actually used

    # ---- Training logs ----
    episode_rewards = []
    episode_vdev = []

    for ep in range(episodes):
        eps = eps0 + (eps1 - eps0) * (ep / max(1, episodes - 1))
        P3_day, P4_day = sample_day_profiles()

        V_bus = np.ones(4, dtype=float) * 1.0
        Vref = 1.0

        ep_reward = 0.0
        ep_v_list = []

        for hour in range(24):
            P_kw_vec = np.array([P1_kw, P2_kw, float(P3_day[hour]), float(P4_day[hour])], dtype=float)
            P_pu_vec = sim.kw_to_pu(P_kw_vec)
            P_total_pu = float(np.sum(P_pu_vec))

            for k in range(sim.steps_per_hour):
                V_avg = float(np.mean(V_bus))
                Iinv_pu = float(P_total_pu / max(V_avg, 0.7))

                svec = obs_to_vec(V_bus, Iinv_pu, P_total_pu, hour)

                # epsilon-greedy
                if rng.random() < eps:
                    a_idx = int(rng.integers(0, nA))
                else:
                    with torch.no_grad():
                        a_idx = int(torch.argmax(q(torch.tensor(svec)[None, :])).item())

                dv_action = float(actions[a_idx])

                # Apply improved Vref update (rate-limit + smoothing)
                Vref, dv_used = apply_vref_update(Vref, dv_action)

                # Environment update
                V_source = Vref - sim.k_droop * P_total_pu
                V_targets = V_source - sim.k_drop * P_pu_vec
                V_bus = V_bus + (sim.dt / sim.tau_v) * (V_targets - V_bus)

                V_avg2 = float(np.mean(V_bus))
                Iinv2 = float(P_total_pu / max(V_avg2, 0.7))
                sp = obs_to_vec(V_bus, Iinv2, P_total_pu, hour)

                r = reward_fn(V_bus, Iinv2, dv_used)
                ep_reward += r
                ep_v_list.append(float(np.mean(np.abs(V_bus - 1.0))))

                done = (hour == 23 and k == sim.steps_per_hour - 1)
                rb.push(svec, a_idx, r, sp, done)

                # Learn
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

                    loss = torch.mean((q_sa - y) ** 2)
                    opti.zero_grad()
                    loss.backward()
                    opti.step()

        episode_rewards.append(float(ep_reward))
        episode_vdev.append(float(np.mean(ep_v_list)) if len(ep_v_list) else float("nan"))

        # Target update + progress print
        if (ep + 1) % 20 == 0:
            qt.load_state_dict(q.state_dict())

        if (ep + 1) % max(1, print_every) == 0:
            print(
                f"Episode {ep+1:4d}/{episodes} | eps={eps:.3f} | "
                f"Reward={episode_rewards[-1]:.2f} | Mean|V-1|={episode_vdev[-1]:.4f}"
            )

    training_logs = {
        "episode_reward": np.array(episode_rewards, dtype=float),
        "episode_voltage_deviation": np.array(episode_vdev, dtype=float),
        "actions": actions.copy(),
        "use_forecast_feature": bool(use_forecast_feature),
        "forecast_mode": str(forecast_mode),
        "vref_smooth_alpha": float(vref_smooth_alpha),
        "max_delta_per_step": float(max_delta_per_step),
    }

    # ---- Inference controller ----
    Vref_state = 1.0

    def ctrl(obs):
        nonlocal Vref_state
        V_bus = obs["V_bus"]
        Iinv_pu = obs["Iinv_pu"]
        P_total_pu = obs["P_total_pu"]
        hour = obs["hour"]

        svec = obs_to_vec(V_bus, Iinv_pu, P_total_pu, hour)

        with torch.no_grad():
            a_idx = int(torch.argmax(q(torch.tensor(svec)[None, :])).item())

        dv_action = float(actions[a_idx])

        # Apply the same update logic used in training
        Vref_state, _dv_used = apply_vref_update(Vref_state, dv_action)

        return Vref_state

    return ctrl, training_logs
