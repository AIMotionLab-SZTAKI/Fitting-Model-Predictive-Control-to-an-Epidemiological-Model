from scipy.integrate import solve_ivp
import casadi as cs
import numpy as np
from parameters import real_population,dt
from compartmental_model import dydt_numpy
def real_system_step(u,t_span,y0):
    times=np.linspace(t_span[0],t_span[1],1000)
    soln = solve_ivp(lambda t, x: dydt_numpy(t, x, u), t_span, y0, t_eval=times)
    return soln
def real_model_simulation(u_values,time_horizont,x0):
    
    real_system=[]
    t0_step=0
    t_end_step=t0_step+dt
    t_span=np.array([t0_step,t_end_step])
    x=x0
    hospital=[]
    hospital.append(x[5]*real_population)
        

    while(t0_step < time_horizont-1):
        sol=real_system_step(u_values[int(t0_step/dt)],t_span,x)
        real_system.append(sol)
        x=sol.y[:,-1]
        t0_step=t0_step+dt
        t_end_step=t_end_step+dt
        t_span=np.array([t0_step,t_end_step])
        hospital.append(x[5]*real_population)    
    return hospital