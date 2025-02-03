from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC
import numpy as np
from torch_nets import system_step_neural,output_mapping_neural,get_encoder
from compartmental_model import runge_kutta_4_step,compartmental_model_mapping
# import pyPanSim as sp
from support import visualize_Y_quess_vs_Y_real,visualize_comapartmental,visualize_execution_time
from parameters import *
# U_init=np.ones(30)*0
# encoder=get_encoder()
# simulator = sp.SimulatorInterface()
# pansim=PanSim(simulator,encoder)
# subnet=Plant(16,system_step_neural,output_mapping_neural)
# plant=Plant(16,system_step_neural,output_mapping_neural)
# x0=pansim.get_initial_state(U_init)
comparmental_plant=Plant(8,runge_kutta_4_step,compartmental_model_mapping)
comparmental_model=Model(8,runge_kutta_4_step,compartmental_model_mapping)
# U=read_array_from_txt('control.txt')
# U=np.ones(total_time_horizont_extended)*17
[Y_real,Y_model,U,X,time]=shrinking_MPC(0,0,total_time_horizont_extended,x0_comparmental,grace_time_extended,comparmental_model,comparmental_plant,1)
# # [Y_real,Y_model,errors]=constant_U_values(U,pansim,subnet,total_time_horizont_extended,x0)
# visualize_comapartmental(X*real_population,U)
print(np.average(time))
visualize_Y_quess_vs_Y_real(Y_model*real_population,Y_real*real_population,U)
visualize_execution_time(time)
print(np.sum(time))