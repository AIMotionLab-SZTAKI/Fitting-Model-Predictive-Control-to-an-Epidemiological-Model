from parameters import *
import numpy as np
from torch_nets import get_net_models
from torch_nets import system_step
import matplotlib.pyplot as plt

def write_array_to_txt(name,array_input):
    np.savetxt(name,array_input)    

def read_array_from_txt(name):
    array_loaded=np.loadtxt(name)
    return array_loaded

def simulate(U,x0,noise_dec):
    Y = []

    x=x0
    nerual_models=get_net_models()
    
    
    for i in range(len(U)):
        [x_next, y_out] = system_step( nerual_models['f'],nerual_models['h'], x, U[i])
        
        if noise_dec==1 :
            noise = np.random.rand() * 0.025
            x=x_next+noise
        else:
            x=x_next
        Y.append(np.squeeze(y_out))   

    Y = np.array(Y)
    return [Y,x_next]

def visualize_sol(Y,U):
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(Y)
    plt.subplot(2,1,2)
    plt.plot(U,'.')
    plt.grid()
    plt.show()
def from_solution_to_x_u_y(solution,time_horizon):
    control_time=int((time_horizon-1)/holding_time)+1
    solution_x_u_y = solution[ 'x' ]  
    x_opt = solution_x_u_y[ :16*time_horizon].reshape(( 16, time_horizon ))  
    u_opt = solution_x_u_y[ 16*time_horizon:16*time_horizon + control_time ].reshape(( 1, control_time ))   
    y_opt = solution_x_u_y[ 16*time_horizon+control_time: ].reshape((1, time_horizon))
    return [x_opt,u_opt,y_opt]
def from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_horizon):
    control_time=int((time_horizon-1)/holding_time)+1
    x_faltten=cs.reshape(x_opt,time_horizon*16,1)
    u_faltten=cs.reshape(u_opt,control_time,1)
    y_faltten=cs.reshape(y_opt,time_horizon,1)
    result=cs.vertcat(x_faltten,u_faltten,y_faltten)
    return result
def u_extended(U,horizont):
    U_result=np.ones((1,horizont))
    for i in range (horizont):
        U_result[:,i]=U[int(i/holding_time)] 
    return U_result