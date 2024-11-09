from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
import numpy as np
from parameters_for_nerual import *
from simulate import simulate_with_nerual_network,simualte_with_PanSim
from support import get_init_state
from torch_nets import get_encoder
import pyPanSim as sp
def open_loop_shr_neural(noise,noise_MPC,MPC_type,x0,disc):
    time_horizont=total_time_horizont_extended
    grace_time=grace_time_extended
    x_init=np.zeros((17*time_horizont+int((time_horizont-1)/holding_time)+1,1))
    [Y,U,X]=MPC_type(noise_MPC,time_horizont,x0,grace_time,holding_time,x_init)
    if disc==1:
        U_calculated=np.round(U)
        U_calculated[U_calculated<0]=0
    else:
        U_calculated=U
               
    [Y_sim,x_next]=simulate_with_nerual_network(U_calculated,x0.T,noise)
    return [Y_sim,U_calculated,X]
def open_loop_roll_neural(noise,noise_MPC,MPC_type,x0,disc):
    time_horizont=total_time_horizont_extended
    grace_time=grace_time_extended
    x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
    [Y,U,X]=MPC_type(noise_MPC,time_horizont,x0,grace_time,holding_time,x_init)
    if disc==1:
        U_calculated=np.round(U)
        U_calculated[U_calculated<0]=0
    else:
        U_calculated=U
               
    [Y_sim,x_next]=simulate_with_nerual_network(U_calculated,x0.T,noise)
    return [Y_sim,U_calculated,X]
def open_loop_shr_PanSim(MPC_type,noise_MPC,U_init):
    time_horizont=total_time_horizont_extended
    simulator = sp.SimulatorInterface()
    simulator.initSimulation(init_options)
    encoder=get_encoder()
    [x,U_pre,Y_pre]=get_init_state(simulator,U_init,encoder)
    x_init=np.zeros((17*time_horizont+int((time_horizont-1)/holding_time)+1,1))
    [Y,U,X]=open_loop_shr_neural(0,noise_MPC,MPC_type,x.T,1)
    [New_Input,Y_real]=simualte_with_PanSim(simulator,U)
    return [Y,Y_real,U]
def open_loop_roll_PanSim(MPC_type,noise_MPC,U_init):
    time_horizont=total_time_horizont_extended
    simulator = sp.SimulatorInterface()
    simulator.initSimulation(init_options)
    encoder=get_encoder()
    [x,U_pre,Y_pre]=get_init_state(simulator,U_init,encoder)
    x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
    [Y,U,X]=open_loop_roll_neural(0,noise_MPC,MPC_type,x.T,1)
    [New_Input,Y_real]=simualte_with_PanSim(simulator,U)
    return [Y,Y_real,U]