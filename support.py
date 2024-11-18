from parameters import *
import numpy as np
import casadi as cs
import torch
from torch_nets import unorm,ynorm
import matplotlib.pyplot as plt

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
    plt.xlabel("Time [Days]")
    plt.ylabel("Patients")
    plt.plot(Y,label='Predicted process by the neural network')
    plt.legend()
    plt.subplot(2,1,2)
    plt.grid()
    plt.plot(U,'.')
    plt.xlabel("Time [Days]")
    plt.ylabel("Control input index")
    plt.show()
def visualize_Y_quess_vs_Y_real(Y_quess,Y_real,U):
    plt.subplot(2,1,1)
    plt.grid()
    plt.xlabel("Time [Days]")
    plt.ylabel("Patients")

    plt.plot(Y_quess,label='Predicted process by the neural network')
    plt.plot(Y_real,label='Real process by the PanSim')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(U,'.')
    plt.xlabel("Time [Days]")
    plt.ylabel("Control input index")
    plt.grid()
    plt.show()


def from_solution_to_x_u_y(solution,time_horizon,dim):
    control_time=int(np.ceil(time_horizon/holding_time))
    solution_x_u_y = solution[ 'x' ]  
    x_opt = solution_x_u_y[ :dim*time_horizon].reshape(( dim, time_horizon ))  
    u_opt = solution_x_u_y[ dim*time_horizon:dim*time_horizon + control_time ].reshape(( 1, control_time ))   
    y_opt = solution_x_u_y[ dim*time_horizon+control_time: ].reshape((1, time_horizon))
    return [x_opt,u_opt,y_opt]

def from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_horizon,dim):
    control_time=int(np.ceil(time_horizon/holding_time))
    x_faltten=cs.reshape(x_opt,time_horizon*dim,1)
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
def norm_round(vector):
    max=np.max(vector)
    return np.round((vector/max)*max_control)/(max_control/max)
def visualize_comapartmental (x_opt,u_opt,y_real):
    u_q=norm_round(u_opt)
    plt.grid()
    plt.plot( y_real,color="k",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[5,:]),color="m",linestyle="-",marker="")
    plt.legend(['The real system respond ', 'The predicted respond'])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()
    plt.grid()
    plt.plot(np.squeeze(u_q),color="r",linestyle="",marker=".")
    plt.plot(np.squeeze(u_opt),color="b",linestyle="",marker=".")
    plt.legend(['Control signal with round','Control signal without round'])
    plt.xlabel("Time [days]")
    plt.ylabel("Control scenarios")
    plt.show()
    plt.grid()
    plt.plot(np.squeeze (x_opt[0,:]),color="b",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[6,:]),color="g",linestyle="-",marker="")
    plt.legend(['Susceptibles','Recover'])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()
    plt.grid()
    plt.plot(np.squeeze (x_opt[1,:]),color="b",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[2,:]),color="g",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[3,:]),color="r",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[4,:]),color="c",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[5,:]),color="m",linestyle="-",marker="")
    plt.plot(np.squeeze (x_opt[7,:]),color="k",linestyle="-",marker="")
    plt.legend(['Latent','Pre-symptomatic ','Symptomatic infected','Symptomatic infected but will recover','Hospital','Died'])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()
def visualize_the_system_comaprtmental(Y):
    plt.grid()
    plt.plot(Y,color="m",linestyle="-",marker="")
    plt.legend(['The real system respond '])
    plt.xlabel("Time [days]")
    plt.ylabel("Cardinality of the set [sample]")
    plt.show()
