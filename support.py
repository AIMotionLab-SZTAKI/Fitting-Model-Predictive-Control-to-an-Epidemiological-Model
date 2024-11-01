from parameters import *
import numpy as np
import torch
from torch_nets import get_net_models,system_step,get_encoder,unorm,ynorm,system_step


def get_results(resultArray):
    nHospitalied = []
    for result in resultArray:
        hosp = result[6] + result[7]  
        nHospitalied.append(hosp)
    return nHospitalied


def write_array_to_txt(name,array_input):
    np.savetxt(name,array_input)    

def read_array_from_txt(name):
    array_loaded=np.loadtxt(name)
    return array_loaded
def visualize_simple(Y):
    plt.plot(Y)
    plt.grid()
    plt.show()
def visualize_Y_vs_U(Y,U):
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(Y)
    plt.subplot(2,1,2)
    plt.plot(U,'.')
    plt.grid()
    plt.show()
def visualize_Y_quess_vs_Y_real(Y_quess,Y_real,U):
    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(Y_quess)
    plt.plot(Y_real)
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
def get_init_state(simulator,U_init,encoder):
    results_agg = []
    inputs_agg = []
    run_options_agg = []
    for i in range (30):
        input_idx,run_options=U_init[i],input_sets[int(U_init[i])]
        results = simulator.runForDay(run_options)
        results_agg.append(results)
        inputs_agg.append(input_idx)
        run_options_agg.append(run_options)
    hospitalized_agg=get_results(results_agg)
    [uhist,yhist]=norm_and_unsqueeze(inputs_agg,hospitalized_agg)
    x0 = encoder(uhist, yhist)
    x = x0
    return [x0.detach().numpy(),inputs_agg,hospitalized_agg]
def norm_and_unsqueeze(inputs_agg,hospitalized_agg):
    InputData=torch.Tensor(unorm(inputs_agg))
    OutputData = torch.Tensor(ynorm(hospitalized_agg))
    uhist = InputData[:30].unsqueeze(0).unsqueeze(2)
    yhist = OutputData[:30].unsqueeze(0).unsqueeze(2)
    return [uhist,yhist]