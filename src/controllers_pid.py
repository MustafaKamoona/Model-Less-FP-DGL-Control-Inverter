def make_pid_controller(Vset=1.0, V0=1.0, Kp=0.95, Ki=0.40):
    integ = 0.0
    last_t = None

    def ctrl(obs):
        nonlocal integ, last_t
        t = obs["t"]
        if last_t is None:
            last_t = t
        dt = max(1e-6, t - last_t)
        last_t = t

        e = Vset - obs["V_avg"]
        integ += e * dt
        Vref = V0 + Kp*e + Ki*integ
        return Vref

    return ctrl
