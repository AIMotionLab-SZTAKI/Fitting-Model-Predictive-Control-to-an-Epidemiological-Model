from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import visualize_sol,simulate
import numpy as np

noise =0
[Y,U]=rolling_MPC(noise)
visualize_sol(Y,U)
# U_sim=np.round(U)
# for i in range (len(U_sim)):
#     if (U_sim[i]<0):
#         U_sim[i]=0
Y_sim=simulate(U)
visualize_sol(Y_sim,U)
