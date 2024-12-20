import numpy as np
from torch_nets import get_net_models
from torch_nets import system_step
from parameters_for_nerual import *
from support import get_results
def simulate_with_nerual_network(U,x0,noise_dec):
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
def simualte_with_PanSim(simulator,U):
    results_agg = []
    inputs_agg=[]
    run_options_agg = []
    for i in range (len(U)):
        input_idx,run_options=U[i],input_sets[int(U[i])]
        results = simulator.runForDay(run_options)
        results_agg.append(results)
        inputs_agg.append(input_idx)
        run_options_agg.append(run_options)
    hospitalized_agg=get_results(results_agg)
    return [inputs_agg,hospitalized_agg]