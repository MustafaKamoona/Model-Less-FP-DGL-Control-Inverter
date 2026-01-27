import os
import json
import numpy as np
import matplotlib.pyplot as plt

from data.load_profiles import P1_KW, P2_KW, P3_KW_24, P4_KW_24
from src.microgrid_sim import MicrogridSim
from src.controllers_pid import make_pid_controller
from src.lstm_baseline import train_lstm_forecaster, forecast_next_total_load, make_lstm_baseline_controller
from src.rl_dqn import make_dqn_controller
from src.metrics import compute_metrics

def ensure_dirs(base="results"):
    os.makedirs(os.path.join(base, "figures"), exist_ok=True)
    os.makedirs(os.path.join(base, "metrics"), exist_ok=True)

def plot_and_save(sim_pid, sim_lstm, sim_rl, outdir="results/figures"):
    sims = [("PID", sim_pid), ("LSTM-baseline", sim_lstm), ("Proposed RL", sim_rl)]

    # Average voltage
    plt.figure(figsize=(10,5))
    for name, s in sims:
        plt.plot(s["t"], np.mean(s["V"], axis=1), label=f"{name} (Vavg)")
    plt.axhline(1.05, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Time (h)")
    plt.ylabel("Average Bus Voltage (pu)")
    plt.title("Average AC Voltage over 24h")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Vavg_24h.png"), dpi=200)
    plt.close()

    # Per-bus voltages for RL
    plt.figure(figsize=(10,5))
    for i in range(4):
        plt.plot(sim_rl["t"], sim_rl["V"][:, i], label=f"Bus {i+1}")
    plt.axhline(1.05, linestyle="--")
    plt.axhline(0.95, linestyle="--")
    plt.xlabel("Time (h)")
    plt.ylabel("Voltage (pu)")
    plt.title("Proposed RL: Per-Bus Voltages")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "RL_bus_voltages.png"), dpi=200)
    plt.close()

    # Inverter current
    plt.figure(figsize=(10,5))
    for name, s in sims:
        plt.plot(s["t"], s["Iinv_pu"], label=name)
    plt.axhline(1.20, linestyle="--")
    plt.xlabel("Time (h)")
    plt.ylabel("Inverter Current (pu)")
    plt.title("Inverter Current")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Iinv_24h.png"), dpi=200)
    plt.close()

    # Vref
    plt.figure(figsize=(10,5))
    for name, s in sims:
        plt.plot(s["t"], s["Vref"], label=name)
    plt.xlabel("Time (h)")
    plt.ylabel("Vref (pu)")
    plt.title("Controller Output (Vref)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "Vref_24h.png"), dpi=200)
    plt.close()

def main():
    ensure_dirs()

    sim = MicrogridSim()

    # --- PID ---
    pid = make_pid_controller()
    sim_pid = sim.run_day(pid, P1_KW, P2_KW, P3_KW_24, P4_KW_24)

    # --- LSTM baseline ---
    lstm_model, lookback, mode = train_lstm_forecaster(P1_KW, P2_KW, P3_KW_24, P4_KW_24)
    def fcast_fn(hour, P3, P4):
        return forecast_next_total_load(hour, P1_KW, P2_KW, P3, P4, lstm_model, lookback)
    lstm_ctrl = make_lstm_baseline_controller(sim.S_base_kva)
    sim_lstm = sim.run_day(lstm_ctrl, P1_KW, P2_KW, P3_KW_24, P4_KW_24, forecast_fn=fcast_fn)

    # --- Proposed RL: model-free DQN; forecast feature uses next-hour schedule (NOT LSTM) ---
    rl_ctrl = make_dqn_controller(
        sim,
        P1_KW, P2_KW, P3_KW_24, P4_KW_24,
        use_forecast_feature=True,
        forecast_mode="next_hour_schedule",
        episodes=80
    )
    sim_rl = sim.run_day(rl_ctrl, P1_KW, P2_KW, P3_KW_24, P4_KW_24)

    # Save raw arrays
    np.savez("results/sim_pid.npz", **sim_pid)
    np.savez("results/sim_lstm.npz", **sim_lstm)
    np.savez("results/sim_rl.npz", **sim_rl)

    # Metrics
    m_pid = compute_metrics(sim_pid, sim.I_max_pu, P3_KW_24, P4_KW_24)
    m_lstm = compute_metrics(sim_lstm, sim.I_max_pu, P3_KW_24, P4_KW_24)
    m_rl = compute_metrics(sim_rl, sim.I_max_pu, P3_KW_24, P4_KW_24)

    metrics = {"PID": m_pid, "LSTM-baseline": m_lstm, "Proposed RL": m_rl}
    with open("results/metrics/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    plot_and_save(sim_pid, sim_lstm, sim_rl)

    print("Done. Metrics saved to results/metrics/metrics.json and figures saved to results/figures/")

if __name__ == "__main__":
    main()
