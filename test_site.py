from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import visualize_sol
import numpy as np
from parameters import *
from closed_loop_simulation import simualte_with_closed_loop
# x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
# [Y,U,X]=rolling_MPC(0,total_time_horizont_extended,x0,grace_time,holding_time,x_init)
# visualize_sol(Y,U)

[Y,U]=simualte_with_closed_loop(1,1,rolling_MPC,1)
visualize_sol(Y,U)
# [Y,U]=simualte_with_open_loop(0,0,shrinking_MPC,1)
# visualize_sol(Y,U)
