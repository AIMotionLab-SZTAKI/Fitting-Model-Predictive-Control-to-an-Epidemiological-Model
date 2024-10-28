from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import simulate
import numpy as np
from parameters import *
def simualte_with_close_loop (noise,noise_MPC,MPC_type,disc):
    U_system=np.empty((total_time_horizont_extended))
    Y_system=np.empty((total_time_horizont_extended))
    
    delta_time=holding_time
    time_horizont=total_time_horizont_extended
    x=x0
    if MPC_type==shrinking_MPC:
        x_init=np.zeros((17*time_horizont+int((time_horizont-1)/holding_time)+1,1))
        for i in range(int(np.ceil(total_time_horizont_extended/delta_time))):
                
                if time_horizont>holding_time:
                    [Y,U,X]=MPC_type(noise_MPC,time_horizont,x,grace_time,holding_time,x_init)
                    
                    x_init=X
                    if disc==1:
                        U_calculated=np.round(U)
                        U_calculated[U_calculated<0]=0
                    else:
                        U_calculated=U
                else:
                    U=np.zeros(time_horizont)
                    U_calculated=U
                if i==0:
                    x=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                    0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
                    [Y_sim,x_next]=simulate(U_calculated[0:delta_time],x,noise)
                else:
                    [Y_sim,x_next]=simulate(U_calculated[0:delta_time],x.T,noise)
                x=x_next.T
                U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
                Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
                time_horizont=time_horizont-delta_time
        return [Y_system,U_system]
    
    if MPC_type==rolling_MPC:
        x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
        i=0
        while (time_horizont>2*grace_time):
                
            if time_horizont>holding_time*2:
                [Y,U,X]=MPC_type(noise_MPC,time_horizont,x,grace_time,holding_time,x_init)
                x_init=X
                if disc==1:
                    U_calculated=np.round(U)
                    U_calculated[U_calculated<0]=0
                else:
                    U_calculated=U
        
               
            if i==0:
                x=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
                [Y_sim,x_next]=simulate(U_calculated[0:delta_time],x,noise)
            else:
                [Y_sim,x_next]=simulate(U_calculated[0:delta_time],x.T,noise)
            x=x_next.T
            U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
            Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
            time_horizont=time_horizont-delta_time
            i=i+1
        [Y_sim,x_next]=simulate(U_calculated[2*-grace_time:],x.T,noise)
        U_system[2*-grace_time:]=U_calculated[2*-grace_time:]
        Y_system[2*-grace_time:]=Y_sim
        return [Y_system,U_system]

        
  