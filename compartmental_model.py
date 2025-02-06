import numpy as np
import casadi as cs

def dydt_numpy(t, x, u):
    # Paraméterek tömbje, amely a modell egyes konstansait tartalmazza
    param = np.array([1/3, 0.75, 1, 1/2.5, 1/3, 0.6, 1/4, 1/4, 0.076, 1/10, 0.145])
    
    # Az állapotváltozók kibontása a bemeneti vektorból
    S, L, P, I, A, H, R, D = x
    
    # A járvány dinamikája
    dSdt = -param[0] * (1 - u) * (P + I + A * param[1]) * S / param[2]
    dLdt = param[0] * (1 - u) * (P + I + A * param[1]) * S / param[2] - param[3] * L
    dPdt = param[3] * L - param[4] * P
    dIdt = param[4] * param[5] * P - param[6] * I
    dAdt = (1 - param[5]) * param[4] * P - param[7] * A
    dHdt = param[6] * param[8] * I - param[9] * H
    dRdt = param[6] * (1 - param[8]) * I + param[7] * A + (1 - param[10]) * param[9] * H
    dDdt = param[10] * param[9] * H
    
    return np.array([dSdt, dLdt, dPdt, dIdt, dAdt, dHdt, dRdt, dDdt])

def dydt_casadi(t, x, u):
    # Paraméterek deklarálása Casadi MX szimbolikus változóként
    param = cs.MX([1/3, 0.75, 1, 1/2.5, 1/3, 0.6, 1/4, 1/4, 0.076, 1/10, 0.145])
    
    # Az állapotváltozók külön-külön történő kibontása
    S, L, P, I, A, H, R, D = x
    
    # A járvány dinamikája
    dSdt = -param[0] * (1 - u) * (P + I + A * param[1]) * S / param[2]
    dLdt = param[0] * (1 - u) * (P + I + A * param[1]) * S / param[2] - param[3] * L
    dPdt = param[3] * L - param[4] * P
    dIdt = param[4] * param[5] * P - param[6] * I
    dAdt = (1 - param[5]) * param[4] * P - param[7] * A
    dHdt = param[6] * param[8] * I - param[9] * H
    dRdt = param[6] * (1 - param[8]) * I + param[7] * A + (1 - param[10]) * param[9] * H
    dDdt = param[10] * param[9] * H
    
    return cs.vertcat(dSdt, dLdt, dPdt, dIdt, dAdt, dHdt, dRdt, dDdt).T

def runge_kutta_4_step(dydt, x, u, dt=1):
    # Kezdeti időpont 
    t = 0
    
    # Runge-Kutta 4. rendű lépései
    k1 = dydt(t, x, u)
    k2 = dydt(t + dt / 2, x + (dt / 2) * k1, u)
    k3 = dydt(t + dt / 2, x + (dt / 2) * k2, u)
    k4 = dydt(t + dt, x + dt * k3, u)
    
    # Súlyozott összegzés
    K = k1 + 2 * k2 + 2 * k3 + k4
    
    # Következő állapot kiszámítása
    res = x + (dt / 6) * K
    return res

def compartmental_model_mapping(x):
    # A kimeneti függvény, amely a kórházi esetek számát adja vissza
    return x[5]
