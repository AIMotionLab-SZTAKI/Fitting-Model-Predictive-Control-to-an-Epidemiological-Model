from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import visualize_sol,simulate
import numpy as np
noise =1
[Y,U]=shrinking_MPC(noise)
visualize_sol(Y,U)
noise =0
[Y,U]=shrinking_MPC(noise)
visualize_sol(Y,U)
# U_sim=np.floor(U)
# for i in range (len(U_sim)):
#     if (U_sim[i]<0):
#         U_sim[i]=0
# Y_sim=simulate(U_sim)
# visualize_sol(Y_sim,U_sim)
