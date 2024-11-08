from support import read_array_from_txt,visualize_Y_quess_vs_Y_real,visualize_Y_vs_U,write_array_to_txt
from shrinking_horizont import shrinking_MPC
from rolling_horizont import rolling_MPC
from closed_loop_simulation import closed_loop_roll_neural,closed_loop_shr_neural
from open_loop_simulation import open_loop_roll_neural,open_loop_shr_neural
import numpy as np
from opti_problem_for_neural import *
x=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                    0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
[Y,U]=closed_loop_roll_neural(1,1,rolling_MPC,x.T,1)
write_array_to_txt('neuralplant.txt',Y)
write_array_to_txt('control.txt',U)
visualize_Y_vs_U(Y,U)

[Y,U,X]=open_loop_roll_neural(1,1,rolling_MPC,x.T,1)
write_array_to_txt('neuralplant.txt',Y)
write_array_to_txt('control.txt',U)
visualize_Y_vs_U(Y,U)
[Y,U]=closed_loop_shr_neural(1,1,shrinking_MPC,x.T,1)
write_array_to_txt('neuralplant.txt',Y)
write_array_to_txt('control.txt',U)
visualize_Y_vs_U(Y,U)
[Y,U,X]=open_loop_shr_neural(1,1,shrinking_MPC,x.T,1)
write_array_to_txt('neuralplant.txt',Y)
write_array_to_txt('control.txt',U)
visualize_Y_vs_U(Y,U)