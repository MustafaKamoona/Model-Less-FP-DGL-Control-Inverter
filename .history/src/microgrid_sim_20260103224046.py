import numpy as np

class MicrogridSim:
    """A lightweight surrogate microgrid simulator (averaged inverter + feeder voltage drop).

    - Hourly loads are piecewise-constant.
    - Internal time-step is 1 minute by default.
    - Control action is Option A: set inverter AC voltage reference Vref (pu).

    This is meant as a control benchmark (not EMT-accurate).
    """

    def __init__(
        self,
        S_base_kva: float = 100.0,
        V_nom_ll: float = 400.0,
        tau_v_hours: float = 0.6,
        dt_hours: float = 1/60,
        k_drop=None,
        k_droop: float = 0.015,
        I_max_pu: float = 1.20,
        Vref_min: float = 0.95,
        Vref_max: float = 1.05,
        enable_reactive_model: bool = True,
        kq: float = 0.30,          # reactive sensitivity (pu Q per pu voltage error)
        Q_max_pu: float = 0.40,    # reactive limit (pu)
        seed: int = 42
    ):
        self.S_base_kva = float(S_base_kva)
        self.V_nom_ll = float(V_nom_ll)
        self.tau_v = float(tau_v_hours)
        self.dt = float(dt_hours)
        self.steps_per_hour = int(round(1 / self.dt))
        self.k_drop = np.array(k_drop if k_drop is not None else [0.020, 0.022, 0.030, 0.034], dtype=float)
        assert self.k_drop.shape == (4,)
        self.k_droop = float(k_droop)
        self.I_max_pu = float(I_max_pu)
        self.Vref_min = float(Vref_min)
        self.Vref_max = float(Vref_max)

        self.enable_reactive_model = bool(enable_reactive_model)
        self.kq = float(kq)
        self.Q_max_pu = float(Q_max_pu)

        self.rng = np.random.default_rng(seed)

    def kw_to_pu(self, P_kw: np.ndarray) -> np.ndarray:
        return np.asarray(P_kw, dtype=float) / self.S_base_kva

    def run_day(
        self,
        controller_fn,
        P1_kw: float,
        P2_kw: float,
        P3_24: np.ndarray,
        P4_24: np.ndarray,
        forecast_fn=None,
        Vdc=800.0
    ):
        """Simulate 24h with 1-minute internal steps.

        controller_fn(obs)-> Vref (pu)
        forecast_fn(hour, P3_24, P4_24)-> any (used by LSTM baseline only)
        """
        V_bus = np.ones(4, dtype=float) * 1.0
        Vref = 1.0

        T = []
        V_log = []
        Vref_log = []
        Vdc_log = []
        Iinv_log = []
        P_log = []
        Q_log = []
        overI_log = []

        for hour in range(24):
            P_kw_vec = np.array([P1_kw, P2_kw, float(P3_24[hour]), float(P4_24[hour])], dtype=float)
            P_pu_vec = self.kw_to_pu(P_kw_vec)
            P_total_pu = float(np.sum(P_pu_vec))

            fcast = forecast_fn(hour, P3_24, P4_24) if forecast_fn is not None else None

            for k in range(self.steps_per_hour):
                t = hour + k * self.dt
                V_avg = float(np.mean(V_bus))

                # Simple reactive support model (optional):
                # If voltage is low -> inject +Q to support voltage. If voltage is high -> absorb Q.
                if self.enable_reactive_model:
                    Q_total_pu = float(np.clip(self.kq * (1.0 - V_avg), -self.Q_max_pu, self.Q_max_pu))
                else:
                    Q_total_pu = 0.0

                # Approx inverter current magnitude (pu) from apparent power / voltage
                S_pu = float(np.sqrt(P_total_pu**2 + Q_total_pu**2))
                Iinv_pu = float(S_pu / max(V_avg, 0.7))
                overI = max(0.0, Iinv_pu - self.I_max_pu)

                obs = {
                    "t": t,
                    "hour": hour,
                    "V_bus": V_bus.copy(),
                    "V_avg": V_avg,
                    "Vref": Vref,
                    "Vdc": float(Vdc),
                    "Iinv_pu": Iinv_pu,
                    "P_pu_vec": P_pu_vec.copy(),
                    "P_total_pu": P_total_pu,
                    "Q_total_pu": Q_total_pu,
                    "forecast": fcast,
                }

                # Controller chooses Vref
                Vref = float(controller_fn(obs))
                Vref = float(np.clip(Vref, self.Vref_min, self.Vref_max))

                # Source voltage with droop vs loading
                V_source = Vref - self.k_droop * P_total_pu

                # Voltage targets include load drop; Q support reduces drop slightly (simple coupling)
                # (This is a surrogate; it just makes Q visibly meaningful for plots.)
                q_support = 0.010 * Q_total_pu  # small coupling gain
                V_targets = (V_source - self.k_drop * P_pu_vec) + q_support

                # First-order voltage dynamics
                V_bus = V_bus + (self.dt / self.tau_v) * (V_targets - V_bus)

                # Logs
                T.append(t)
                V_log.append(V_bus.copy())
                Vref_log.append(Vref)
                Vdc_log.append(float(Vdc))
                Iinv_log.append(Iinv_pu)
                P_log.append(P_total_pu)
                Q_log.append(Q_total_pu)
                overI_log.append(overI)

        return {
            "t": np.array(T),
            "V": np.array(V_log),
            "Vref": np.array(Vref_log),
            "Vdc": np.array(Vdc_log),
            "Iinv_pu": np.array(Iinv_log),
            "P_total_pu": np.array(P_log),
            "Q_total_pu": np.array(Q_log),
            "overI": np.array(overI_log),
        }
