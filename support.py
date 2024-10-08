from parameters import *
import numpy as np
from torch_nets import get_net_models
from torch_nets import system_step
import matplotlib.pyplot as plt


def simulate(U):
    Y = []
    x0=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
    x=x0
    nerual_models=get_net_models()

    for i in range(total_time_horizont):
        [x_next, y_out] = system_step( nerual_models['f'],nerual_models['h'], x, U[i])
        x=x_next
        Y.append(np.squeeze(y_out))   

    Y = np.array(Y)
    return Y

def visualize_sol(Y,U):        
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(Y)
    plt.subplot(2,1,2)
    plt.plot(U,'.')
    plt.grid()
    plt.show()
def from_solution_to_x_u_y(solution,time_horizont):
    solution_x_u_y = solution[ 'x' ]  
    x_opt = solution_x_u_y[ :16*time_horizont].reshape(( 16, time_horizont ))  
    u_opt = solution_x_u_y[ 16*time_horizont:16*time_horizont + time_horizont ].reshape(( 1, time_horizont ))   
    y_opt = solution_x_u_y[ 16*time_horizont+time_horizont: ].reshape((1, time_horizont))
    return [x_opt,u_opt,y_opt]
def from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_horizont):
    x_faltten=cs.reshape(x_opt,time_horizont*16,1)
    u_faltten=cs.reshape(u_opt,time_horizont,1)
    y_faltten=cs.reshape(y_opt,time_horizont,1)
    result=cs.vertcat(x_faltten,u_faltten,y_faltten)
    return result