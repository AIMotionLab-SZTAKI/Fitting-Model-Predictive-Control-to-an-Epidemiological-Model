from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC
import numpy as np
from torch_nets import system_step_neural,output_mapping_neural,get_encoder
from compartmental_model import runge_kutta_4_step,compartmental_model_mapping,dydt_casadi
from control_loops import open_loop,closed_loop,closed_loop_with_PanSim
# import pyPanSim as sp
from support import write_array_to_txt,visualize_Y_vs_U,read_array_from_txt,visualize_comapartmental,from_solution_to_x_u_y,norm_round,u_extended,visualize_execution_time
from parameters import *
from simulate_neural_and_PanSim import simulate_with_nerual_network
from functools import partial
from opti_problem import Problem_With_Grace_time
import casadi as cs
import itertools
# U_init=np.zeros(30)
# encoder=get_encoder()
# simulator = sp.SimulatorInterface()
# Pan=PanSim(simulator,encoder)
model=Model(16,system_step_neural,output_mapping_neural)
plant=Plant(16,system_step_neural,output_mapping_neural)
# model=Model(8,runge_kutta_4_step,compartmental_model_mapping)
# plant=Plant(8,runge_kutta_4_step,compartmental_model_mapping)
x0=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                    0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
# u=read_array_from_txt('Results with neural/Closed/Shr/control_shr_closed.txt')
[Y_model,U_system,execution_times]=closed_loop(0,0,rolling_MPC,x0,1,model,plant)
flattened_vector=list(itertools.chain(*execution_times))
visualize_Y_vs_U(Y_model,U_system)
visualize_execution_time(flattened_vector)
# MyProblem=Problem_With_Grace_time(x0, total_time_horizont_extended, grace_time, holding_time, cs.sumsqr, model )
# sol=MyProblem.get_soultion('ipopt',x_init)
# [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(sol,total_time_horizont_extended,model.dim)
# u_opt=u_extended(u_opt,total_time_horizont_extended)
# uq=norm_round(u_opt)
# [y_real,x_next]=plant.response(uq,x0,0)
# print(y_real.shape)
# print(u_opt.shape)
# print(x_opt.shape)
# visualize_comapartmental(x_opt,u_opt,y_real)




# x_init=np.zeros((model.dim+1)*total_time_horizont_extended+int(np.ceil((total_time_horizont_extended)/holding_time)))
# [Y_model,U_system,X]=open_loop(0,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x_max.T,1,model,plant,x_init)
# visualize_Y_vs_U(Y_model,U_system)
# [Y_model,U_system,X]=open_loop(1,1,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
# write_array_to_txt('neuralplant_roll_plant_model.txt',Y_model)
# write_array_to_txt('control_roll_plant_model.txt',U_system)

# [Y_model,U_system]=closed_loop(0,0,shrinking_MPC,x0,1,model,plant)
# visualize_Y_vs_U(Y_model,U_system)

# [Y_model,U_system]=closed_loop(1,0,rolling_MPC,x0,1,model,plant)
# write_array_to_txt('control_roll_closed.txt',U_system)
# write_array_to_txt('neuralplant_roll_closed.txt',Y_model)


# [Y_model,U_system,X]=open_loop(1,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
# write_array_to_txt('neuralplant_shr_cont.txt',Y_model)
# write_array_to_txt('control_shr_cont.txt',U_system)

# [Y_model,U_system,X]=open_loop(0,0,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,0,model,plant,x_init)
# write_array_to_txt('neuralplant_roll_cont.txt',Y_model)
# write_array_to_txt('control_roll_cont.txt',U_system)

# [Y_model,U_system,X]=open_loop(0,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
# write_array_to_txt('neuralplant_shr_disr.txt',Y_model)
# write_array_to_txt('control_shr_disr.txt',U_system)

# [Y_model,U_system,X]=open_loop(0,0,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
# write_array_to_txt('neuralplant_roll_disr.txt',Y_model)
# write_array_to_txt('control_roll_disr.txt',U_system)

# [Y_model,U_system,X]=open_loop(1,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
# write_array_to_txt('neuralplant_shr_plant_no_model.txt',Y_model)
# write_array_to_txt('control_shr_plant_no_model.txt',U_system)

# [Y_model,U_system,X]=open_loop(1,0,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
# write_array_to_txt('neuralplant_roll_plant_no_model.txt',Y_model)
# write_array_to_txt('control_roll_plant_no_model.txt',U_system)





