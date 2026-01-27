import numpy as np

def compute_metrics(sim: dict, I_max_pu: float, P3_24: np.ndarray, P4_24: np.ndarray, tol: float = 0.05):
    V = sim["V"]
    t = sim["t"]
    Vref = sim["Vref"]
    Iinv = sim["Iinv_pu"]

    vdev = np.abs(V - 1.0)
    max_vdev = float(np.max(vdev))
    mean_vdev = float(np.mean(vdev))

    overI_events = int(np.sum(Iinv > I_max_pu))

    dVref = np.diff(Vref, prepend=Vref[0])
    effort = float(np.sum(np.abs(dVref)))

    peak3 = int(np.argmax(P3_24))
    peak4 = int(np.argmax(P4_24))
    peak_hours = sorted(list(set([peak3, peak4])))

    def recovery_time_after(hour_idx, window_hours=3):
        start_t = hour_idx
        end_t = min(24.0, hour_idx + window_hours)
        mask = (t >= start_t) & (t <= end_t)
        if not np.any(mask):
            return float("nan")
        Vseg = V[mask]
        tseg = t[mask]
        ok = np.all(np.abs(Vseg - 1.0) <= tol, axis=1)
        if not np.any(ok):
            return float("nan")
        return float(tseg[np.argmax(ok)] - start_t)

    rec_times = {int(h): recovery_time_after(h) for h in peak_hours}

    return {
        "max_vdev_pu": max_vdev,
        "mean_vdev_pu": mean_vdev,
        "overI_events": overI_events,
        "control_effort": effort,
        "peak_hours": peak_hours,
        "recovery_times_hours": rec_times,
    }
