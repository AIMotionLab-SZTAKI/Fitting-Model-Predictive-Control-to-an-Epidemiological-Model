from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC,constant_U_values_closed_loop,constant_U_values_open_loop
import numpy as np
from torch_nets import system_step_neural,output_mapping_neural,get_encoder
from compartmental_model import runge_kutta_4_step,compartmental_model_mapping,dydt_casadi
import pyPanSim as sp
from support import write_array_to_txt,visualize_Y_vs_U,read_array_from_txt,visualize_comapartmental,from_solution_to_x_u_y,norm_round,u_extended,visualize_execution_time
from parameters import *
from functools import partial
from opti_problem import Problem_With_Grace_time
import casadi as cs
import itertools
U_init=np.ones(30)*0
encoder=get_encoder()
simulator = sp.SimulatorInterface()
pansim=PanSim(simulator,encoder)
subnet=Plant(16,system_step_neural,output_mapping_neural)
plant=Plant(16,system_step_neural,output_mapping_neural)
x0=pansim.get_initial_state(U_init)
U=np.ones(total_time_horizont_extended)*0
[Y_real,Y_model,U,time]=shrinking_MPC(0,total_time_horizont_extended,x0,grace_time_extended,subnet,pansim,1)

visualize_Y_quess_vs_Y_real(Y_model*real_population,Y_real*real_population,U)
visualize_execution_time(time)


