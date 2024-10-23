from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import simulate
import numpy as np
from parameters import *
def simualte_with_open_loop (noise,noise_MPC,MPC_type):
    U_system=np.empty((total_time_horizont))
    Y_system=np.empty((total_time_horizont))
    
    delta_time=holding_time
    time_horizont=total_time_horizont
    x=x0
    
    if MPC_type==shrinking_MPC:
        x_init=np.zeros((17*total_time_horizont+int((total_time_horizont-1)/holding_time)+1,1))
        grace_time_param=grace_time
    if MPC_type==rolling_MPC:
        x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
        grace_time_param=transfomed_grace_time
    for i in range(int(np.ceil(total_time_horizont/delta_time))):
        if MPC_type==shrinking_MPC:
            if time_horizont>=holding_time:
                print(time_horizont)
                [Y,U,X]=MPC_type(noise_MPC,time_horizont,x,grace_time_param,holding_time,x_init)
                
                x_init=X
                U_disc=np.round(U)
                U_disc[U_disc<0]=0
            else:
                U=np.zeros(time_horizont)
                U_disc=U
        if MPC_type==rolling_MPC:
            if time_horizont-grace_time>=holding_time:
                print(time_horizont)
                [Y,U,X]=MPC_type(noise_MPC,time_horizont,x,grace_time_param,holding_time,x_init)
                
                x_init=X
                U_disc=np.round(U)
                U_disc[U_disc<0]=0
            else:
                U=np.zeros(time_horizont)
                U_disc=U
        if i==0:
            x=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
            0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
            [Y_sim,x_next]=simulate(U_disc[0:delta_time],x,noise)
        else:
            [Y_sim,x_next]=simulate(U_disc[0:delta_time],x.T,noise)
        x=x_next.T
        if i==int(np.ceil(total_time_horizont/delta_time))-1:
            U_system[-time_horizont:]=U_disc[0:time_horizont]
            Y_system[-time_horizont:]=Y_sim[0:time_horizont]
        else:
            U_system[i*delta_time:(i+1)*delta_time]=U_disc[0:delta_time]
            Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
        time_horizont=time_horizont-delta_time
    return [Y_system,U_system]



