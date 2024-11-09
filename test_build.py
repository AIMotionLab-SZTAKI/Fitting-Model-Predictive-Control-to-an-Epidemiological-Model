
from closed_loop_simulation import closed_loop_shr_PanSim,closed_loop_shr_neural,closed_loop_roll_neural,closed_loop_roll_PanSim
from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import write_array_to_txt,get_init_state
import numpy as np
U_init=np.zeros(30)

[Y_model,Y_real,U_system]=closed_loop_roll_PanSim(rolling_MPC,1,U_init)

write_array_to_txt('pred.txt',Y_model)
write_array_to_txt('real.txt',Y_real)
write_array_to_txt('control.txt',U_system)