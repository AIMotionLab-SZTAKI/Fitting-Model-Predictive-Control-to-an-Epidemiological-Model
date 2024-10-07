from matplotlib import pyplot as plt
from shrinking_horizont import shrinking_MPC
from rolling_horizont import rolling_MPC
def visualize_sol(Y,U):        
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(Y)
    plt.subplot(2,1,2)
    plt.plot(U,'.')
    plt.grid()
    plt.show()
[Y,U]=shrinking_MPC()
visualize_sol(Y,U)
# [Y,U]=rolling_MPC()
# visualize_sol(Y,U)
