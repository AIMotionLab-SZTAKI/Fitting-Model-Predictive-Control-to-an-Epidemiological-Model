from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC
import numpy as np
from torch_nets import system_step_neural,output_mapping_neural,get_encoder
from control_loops import open_loop,closed_loop,closed_loop_with_PanSim
#import pyPanSim as sp
from support import write_array_to_txt,visualize_Y_vs_U
from parameters import *
from simulate_neural_and_PanSim import simulate_with_nerual_network
U_init=np.zeros(30)
encoder=get_encoder()
#simulator = sp.SimulatorInterface()
#Pan=PanSim(simulator,encoder)
model=Model(16,system_step_neural,output_mapping_neural)
plant=Plant(16,system_step_neural,output_mapping_neural)
x0=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                    0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
x_init=np.zeros((model.dim+1)*total_time_horizont_extended+int(np.ceil((total_time_horizont_extended)/holding_time)))

[Y_model,U_system,X]=open_loop(0,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,0,model,plant,x_init)
write_array_to_txt('neuralplant_shr_cont.txt',Y_model)
write_array_to_txt('control_shr_cont.txt',U_system)

[Y_model,U_system,X]=open_loop(0,0,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,0,model,plant,x_init)
write_array_to_txt('neuralplant_roll_cont.txt',Y_model)
write_array_to_txt('control_roll_cont.txt',U_system)

[Y_model,U_system,X]=open_loop(0,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
write_array_to_txt('neuralplant_shr_disr.txt',Y_model)
write_array_to_txt('control_shr_disr.txt',U_system)

[Y_model,U_system,X]=open_loop(0,0,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
write_array_to_txt('neuralplant_roll_disr.txt',Y_model)
write_array_to_txt('control_roll_disr.txt',U_system)

[Y_model,U_system,X]=open_loop(1,0,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
write_array_to_txt('neuralplant_shr_plant_no_model.txt',Y_model)
write_array_to_txt('control_shr_plant_no_model.txt',U_system)

[Y_model,U_system,X]=open_loop(1,0,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
write_array_to_txt('neuralplant_roll_plant_no_model.txt',Y_model)
write_array_to_txt('control_roll_plant_no_model.txt',U_system)

[Y_model,U_system,X]=open_loop(1,1,shrinking_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
write_array_to_txt('neuralplant_shr_plant_model',Y_model)
write_array_to_txt('control_shr_plant_model',U_system)


[Y_model,U_system,X]=open_loop(1,1,rolling_MPC,total_time_horizont_extended,grace_time_extended,x0.T,1,model,plant,x_init)
write_array_to_txt('neuralplant_roll_plant_model',Y_model)
write_array_to_txt('control_roll_plant_model',U_system)




