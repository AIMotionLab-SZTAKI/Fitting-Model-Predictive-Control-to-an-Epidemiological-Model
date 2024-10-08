from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import visualize_sol,simulate
[Y,U]=shrinking_MPC()
visualize_sol(Y,U)
Y_sum=simulate(U)
visualize_sol(Y_sum,U)
