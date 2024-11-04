import pyPanSim as sp
from datetime import datetime
from closed_loop_simulation import close_loop_shr_PanSim,closed_loop_shr_neural,closed_loop_roll_neural
from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import write_array_to_txt,get_init_state
from open_loop_simulation import open_loop_shr_PanSim,open_loop_roll_PanSim
import numpy as np
from parameters import init_options
from torch_nets import get_encoder
from simulate import simualte_with_PanSim
U_init=np.ones(30)*5
# simulator = sp.SimulatorInterface()
# simulator.initSimulation(init_options)
# encoder=get_encoder()
# [x,Raw_InputData,Raw_OutputData]=get_init_state(simulator,U_init,encoder)
# [Y_model,U_system]=closed_loop_shr_neural(0,1,shrinking_MPC,x.T,1)
# [inputs_agg,Y_real]=simualte_with_PanSim(simulator,U_system)
# [Y_model,Y_real,U_system]=open_loop_shr_PanSim(shrinking_MPC,1,U_init)
[Y_model,Y_real,U_system]=open_loop_shr_PanSim(shrinking_MPC,1,U_init)
# [Y_model,Y_real,U_system]=close_loop_shr_PanSim(shrinking_MPC,1)
write_array_to_txt('pred.txt',Y_model)
write_array_to_txt('real.txt',Y_real)
write_array_to_txt('control.txt',U_system)