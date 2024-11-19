
import numpy as np
from parameters import *
from models import*
from strategies import *
def open_loop(noise,noise_MPC,MPC_type,time_horizont,grace_time,x0,disc,model,plant,x_init):
    [Y,U,X]=MPC_type(noise_MPC,time_horizont,x0,grace_time,x_init,model)
    if disc==1:
        U_calculated=np.round(U)
        U_calculated[U_calculated<0]=0
    else:
        U_calculated=U
               
    [Y_sim,x_next]=plant.response(U_calculated,x0.T,noise)
    return [Y_sim,U_calculated,X]

def closed_loop(noise,noise_MPC,MPC_type,x0,disc,model,plant):
    if MPC_type==shrinking_MPC:
        time_horizont=total_time_horizont_extended
        grace_time=grace_time_extended
    if MPC_type==rolling_MPC:
        time_horizont=total_time_horizont_extended-grace_time_extended
        grace_time=0
    U_system=np.empty((total_time_horizont_extended))
    Y_system=np.empty((total_time_horizont_extended))
    delta_time=holding_time
    x=x0
    x_init=np.zeros((model.dim+1)*time_horizont+int(np.ceil((time_horizont)/holding_time)))
    for i in range(int(np.ceil(total_time_horizont_extended/delta_time))):
        if total_time_horizont_extended-i*delta_time>grace_time_extended:
            [Y,U_calculated,X]=open_loop(noise,noise_MPC,MPC_type,time_horizont,grace_time,x.T,disc,model,x_init)
        else:
            U_calculated=np.zeros(delta_time)
        [Y_sim,x_next]=plant.response(U_calculated[0:delta_time],x,noise)
        U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
        Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
        x=x_next
        x_init=X
        if MPC_type==rolling_MPC:
            grace_time=grace_time+delta_time
        if MPC_type==shrinking_MPC:
            time_horizont=time_horizont-delta_time
    return [Y_system,U_system]
def closed_loop_with_PanSim(noise_MPC,MPC_type,pansim,model,U_init):
    
    U_system=np.empty((total_time_horizont_extended))
    Y_model=np.empty((total_time_horizont_extended))
    Y_real=np.empty((total_time_horizont_extended))
    x=pansim.get_initial_state(U_init)
    delta_time=holding_time
    if MPC_type==shrinking_MPC:
        time_horizont=total_time_horizont_extended
        grace_time=grace_time_extended
    if MPC_type==rolling_MPC:
        time_horizont=total_time_horizont_extended-grace_time_extended
        grace_time=0
    x_init=np.zeros((model.dim+1)*time_horizont+int(np.ceil((time_horizont)/holding_time)))
    for i in range(int(np.ceil(total_time_horizont_extended/delta_time))):
        if total_time_horizont_extended-i*delta_time>grace_time_extended:
            [Y,U,X]=MPC_type(noise_MPC,time_horizont,x.T,grace_time,x_init,model)    
            U_calculated=np.round(U)
            U_calculated[U_calculated<0]=0
        else:
            U_calculated=np.zeros(delta_time)
        Y_sim=pansim.response(U_calculated[0:delta_time],delta_time)
        x=pansim.get_next_state()
        U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
        Y_real[i*delta_time:(i+1)*delta_time]=Y_sim
        Y_model[i*delta_time:(i+1)*delta_time]=Y[0:delta_time]
        x_init=X
        
        if MPC_type==rolling_MPC:
            grace_time=grace_time+delta_time
        if MPC_type==shrinking_MPC:
            time_horizont=time_horizont-delta_time
    return [Y_model,Y_real,U_system]