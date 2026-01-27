import numpy as np

def compute_metrics(sim_out, I_max_pu, P3_24=None, P4_24=None, peak_hours=(16, 20)):
    """
    Metrics for 24h simulation output dict:
      sim_out["t"]       time array in hours
      sim_out["V"]       shape (N,4) bus voltages pu
      sim_out["Vref"]    shape (N,) inverter voltage reference
      sim_out["Iinv_pu"] shape (N,) inverter current pu
      sim_out["overI"]   shape (N,) amount over current limit (>=0)

    Returns dict of scalar metrics + recovery times.
    """

    t = np.asarray(sim_out["t"], dtype=float)
    V = np.asarray(sim_out["V"], dtype=float)
    Vref = np.asarray(sim_out["Vref"], dtype=float)
    Iinv = np.asarray(sim_out["Iinv_pu"], dtype=float)
    overI = np.asarray(sim_out.get("overI", np.maximum(0.0, Iinv - float(I_max_pu))), dtype=float)

    # Voltage deviation stats (over all buses and all time steps)
    vdev_all = np.abs(V - 1.0)  # (N,4)
    mean_vdev = float(np.mean(vdev_all))
    max_vdev = float(np.max(vdev_all))

    # Over-current events: count time steps where current exceeds limit
    overI_events = int(np.sum(overI > 0.0))

    # ---- FAIR control effort ----
    # Use mean absolute step change (not sum), so update-rate differences don't dominate.
    dVref = np.diff(Vref, prepend=Vref[0])
    control_effort_mean = float(np.mean(np.abs(dVref)))
    control_effort_rms = float(np.sqrt(np.mean(dVref**2)))

    # Recovery time helper:
    # time after the start of hour h until all buses are within ±tol band.
    def recovery_time_after(hour_idx, tol):
        start_t = float(hour_idx)
        mask = t >= start_t
        if not np.any(mask):
            return float("nan")
        Vseg = V[mask]
        tseg = t[mask]

        # all buses within tol at a time index
        ok = np.all(np.abs(Vseg - 1.0) <= tol, axis=1)
        if not np.any(ok):
            return float("nan")

        first_ok_idx = int(np.argmax(ok))
        return float(tseg[first_ok_idx] - start_t)

    rec_5pct = {int(h): recovery_time_after(h, 0.05) for h in peak_hours}
    rec_1pct = {int(h): recovery_time_after(h, 0.01) for h in peak_hours}

    return {
        "max_vdev_pu": max_vdev,
        "mean_vdev_pu": mean_vdev,
        "overI_events": overI_events,
        "control_effort_mean": control_effort_mean,
        "control_effort_rms": control_effort_rms,
        "peak_hours": list(map(int, peak_hours)),
        "recovery_5pct_hours": rec_5pct,
        "recovery_1pct_hours": rec_1pct,
    }
