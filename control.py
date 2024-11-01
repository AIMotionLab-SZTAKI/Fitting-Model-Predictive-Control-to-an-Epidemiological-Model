from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import write_array_to_txt,visualize_Y_vs_U
import numpy as np
from parameters import *
from closed_loop_simulation import closed_loop_shr_neural,closed_loop_roll_neural
[Y,U]=closed_loop_shr_neural(1,1,shrinking_MPC,x0,1)
visualize_Y_vs_U(Y,U)
[Y,U]=closed_loop_roll_neural(1,1,rolling_MPC,x0,1)
visualize_Y_vs_U(Y,U)
# write_array_to_txt('input_1_control.txt',U)
# write_array_to_txt('input_1_plant_quess.txt',Y)