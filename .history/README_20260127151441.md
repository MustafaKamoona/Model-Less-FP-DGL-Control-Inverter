# Microgrid Forecast-Proactive Model-Free RL (Project)

This project implements a **4-bus inverter-dominated microgrid** with:
- DC source: 800 V (fixed)
- One inverter feeding 4 AC buses
- Loads:
  - Bus1: 10 kW (fixed)
  - Bus2: 12 kW (fixed)
  - Bus3: 24-hour profile (peak 33 kW at 16:00)
  - Bus4: 24-hour profile (peak 50 kW at 20:00)

Controllers compared:
1) PID baseline
2) LSTM-based proactive baseline (LSTM used ONLY here)
3) Proposed Model-Free DQN controller (Option A: adjusts Vref), using a **non-ML** forecast feature:
   - `FORECAST_MODE = 'next_hour_schedule'`

## Quick start
1) Open `notebooks/main.ipynb` and run cells top-to-bottom.

Or run from terminal:
```bash
python -m src.run_experiment
```

Outputs:
- `results/sim_*.npz` raw signals
- `results/metrics/metrics.json`
- figures in `results/figures/`

## Notes
- Control is *model-free*: it uses measurements only (voltages, current proxy, power).
- The simulator is a lightweight surrogate (averaged inverter + voltage-drop model) for benchmarking.
"# Model-Less_FP-DGL_Control_Inverter" 
