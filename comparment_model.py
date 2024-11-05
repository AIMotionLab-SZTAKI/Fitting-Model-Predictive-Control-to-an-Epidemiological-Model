import numpy as np
from scipy.integrate import solve_ivp
from parameters_for_compartment import *
import casadi as cs
import numpy as np
import matplotlib.pyplot as plt

def dydt_numpy(t, x,u):
   
                    #0->beta   #1->delta    #2->N          #3->alpha    #4->p    #5->q     #6->ro_1   #7->ro_a  #8->eta  #9->h    #10->mikro
    param=np.array([1/3 ,     0.75 ,       1,    1/2.5 ,      1/3 ,    0.6 ,      1/4 ,     1/4 ,     0.076 ,  1/10 ,    0.145])    
    S = x[0]
    L = x[1]
    P = x[2]
    I = x[3]
    A = x[4]
    H = x[5]
    R = x[6]
    D = x[7]
    dSdt = -param[0]*(1-u)*(P+I+A*param[1])*S/param[2]
    dLdt=param[0]*(1-u)*(P+I+A*param[1])*S/param[2]-param[3]*L
    dPdt=param[3]*L-param[4]*P
    dIdt=param[4]*param[5]*P-param[6]*I
    dAdt=(1-param[5])*param[4]*P-param[7]*A
    dHdt=param[6]*param[8]*I-param[9]*H
    dRdt=param[6]*(1-param[8])*I+param[7]*A+(1-param[10])*param[9]*H
    dDdt=param[10]*param[9]*H
    return [dSdt, dLdt,dPdt,dIdt,dAdt,dHdt,dRdt,dDdt]
def dydt_casadi(t, x, u):
    param = cs.MX([1/3, 0.75, 1, 1/2.5, 1/3, 0.6, 1/4, 1/4, 0.076, 1/10, 0.145])
    S, L, P, I, A, H, R, D = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]

    dSdt = -param[0] * (1 - u) * (P + I + A * param[1]) * S / param[2]
    dLdt = param[0] * (1 - u) * (P + I + A * param[1]) * S / param[2] - param[3] * L
    dPdt = param[3] * L - param[4] * P
    dIdt = param[4] * param[5] * P - param[6] * I
    dAdt = (1 - param[5]) * param[4] * P - param[7] * A
    dHdt = param[6] * param[8] * I - param[9] * H
    dRdt = param[6] * (1 - param[8]) * I + param[7] * A + (1 - param[10]) * param[9] * H
    dDdt = param[10] * param[9] * H
    
    return cs.vertcat(dSdt, dLdt, dPdt, dIdt, dAdt, dHdt, dRdt, dDdt)

def runge_kutta_4_step(x, u,dydt):
    t = 0
    dt = 1
    k1 = dydt(t, x, u)
    k2 = dydt(t + dt/2, x + (dt/2 * k1), u)
    k3 = dydt(t + dt/2, x + (dt/2 * k2), u)
    k4 = dydt(t + dt, x + (dt * k3), u)
    
    K = k1 + 2 * k2 + 2 * k3 + k4
    res = x + (dt / 6) * K
    return res

def real_system_step(u,t_span,y0,dydt):
    times=np.linspace(t_span[0],t_span[1],1000)
    soln = solve_ivp(lambda t, x: dydt(t, x, u), t_span, y0, t_eval=times)
    return soln
def real_model_simulation(u_values,dydt):
    
    real_system=[]
    t0_step=0
    t_end_step=t0_step+dt
    t_span=np.array([t0_step,t_end_step])
    x=x0
    hospital=[]
    hospital.append(x[5]*real_population)
        

    while(t0_step < time_horizont-1):
        sol=real_system_step(u_values[int(t0_step/dt)],t_span,x,dydt)
        real_system.append(sol)
        x=sol.y[:,-1]
        t0_step=t0_step+dt
        t_end_step=t_end_step+dt
        t_span=np.array([t0_step,t_end_step])
        hospital.append(x[5]*real_population)    
    return hospital
def solution_prcessing(solution,time_horizon):
    solution_x_u_y = solution[ 'x' ]  
    x_opt =np.squeeze( solution_x_u_y[ :8*time_horizon].reshape(( 8, time_horizon ))) *real_population 
    u_opt = np.squeeze(solution_x_u_y[ 8*time_horizon:8*time_horizon + time_horizon ].reshape(( 1, time_horizon )))   
    y_opt =np.squeeze (solution_x_u_y[ 8*time_horizon+time_horizon: ].reshape((1, time_horizon)))*real_population
    return [x_opt,u_opt,y_opt]
def norm_round(vector):
    max=np.max(vector)
    return np.round((vector/max)*max_control_value)/(max_control_value/max)
def u_extended(U,horizont):
    U_result=np.ones((1,horizont))
    for i in range (horizont):
        U_result[:,i]=U[int(i/holding_time)] 
    return np.squeeze(U_result)
def visualize (x_opt,u_opt):
    u_opt=u_extended(u_opt,time_horizont)
    u_q=norm_round(u_opt)
    y_real=real_model_simulation(u_q,dydt_numpy)
    plt.grid()
    plt.plot( y_real,color="k",linestyle="-",marker="")
    plt.plot( x_opt[5],color="m",linestyle="-",marker="")
    plt.legend(['The real system respond ','The predicted respond' ])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()
    plt.grid()
    plt.plot(u_q,color="r",linestyle="",marker=".")
    plt.plot(u_opt,color="b",linestyle="",marker=".")
    plt.legend(['Control signal with round','Control signal without round'])
    plt.xlabel("Time [days]")
    plt.ylabel("Control scenarios")
    plt.show()
    plt.grid()
    plt.plot(x_opt[0],color="b",linestyle="-",marker="")
    plt.plot(x_opt[6],color="g",linestyle="-",marker="")
    plt.legend(['Susceptibles','Recover'])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()
    plt.grid()
    plt.plot(x_opt[1],color="b",linestyle="-",marker="")
    plt.plot(x_opt[2],color="g",linestyle="-",marker="")
    plt.plot(x_opt[3],color="r",linestyle="-",marker="")
    plt.plot(x_opt[4],color="c",linestyle="-",marker="")
    plt.plot(x_opt[5],color="m",linestyle="-",marker="")
    plt.plot(x_opt[7],color="k",linestyle="-",marker="")
    plt.legend(['Latent','Pre-symptomatic ','Symptomatic infected','Symptomatic infected but will recover','Hospital','Died'])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()