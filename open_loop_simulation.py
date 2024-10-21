from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import visualize_sol,simulate
import numpy as np
from parameters import *
U_system=np.empty((total_time_horizont))
Y_system=np.empty((total_time_horizont))
noise=1
noise_MPC=0
delta_time=holding_time
time_horizont=total_time_horizont
x=x0
x_init=np.zeros((17*time_horizont+int((time_horizont-1)/holding_time)+1,1)) 
for i in range(int(np.ceil(total_time_horizont/delta_time))):
    if time_horizont>=holding_time:
        [Y,U,X]=shrinking_MPC(noise_MPC,time_horizont,x,grace_time,holding_time,x_init)
        x_init=X
    else:
        U=np.zeros(time_horizont)
    if i==0:
        x=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
        0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
        [Y_sim,x_next]=simulate(U[0:delta_time],x,noise)
    else:
        [Y_sim,x_next]=simulate(U[0:delta_time],x.T,noise)
    x=x_next.T
    if i==int(np.ceil(total_time_horizont/delta_time))-1:
        U_system[-time_horizont:]=U[0:time_horizont]
        Y_system[-time_horizont:]=Y_sim[0:time_horizont]
    else:
        U_system[i*delta_time:(i+1)*delta_time]=U[0:delta_time]
        Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
    time_horizont=time_horizont-delta_time

visualize_sol(Y_system,U_system)

x_init=np.zeros((17*total_time_horizont+int((total_time_horizont-1)/holding_time)+1,1)) 
[Y,U,X]=shrinking_MPC(noise_MPC,total_time_horizont,x0,grace_time,holding_time,x_init)
visualize_sol(Y,U)

