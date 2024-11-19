
from functools import partial
from torch_nets import system_step_neural,get_net_models
from compartmental_model import runge_kutta_4_step,dydt_casadi
import numpy as np
from parameters import init_options,input_sets
from support import get_results,norm_and_unsqueeze
class Model:
    def __init__(self,dimension, dynamic_for_one_step, output_mapping):
        self.dim = dimension
        self.dynamic = dynamic_for_one_step
        self.map = output_mapping
        
        if self.dynamic==system_step_neural:
            net_models = get_net_models()  
            self.dynamic = partial(dynamic_for_one_step, net_models['f'])
            self.map = partial(output_mapping, net_models['h'])
        if self.dynamic==runge_kutta_4_step:
            self.dynamic = partial(dydt_casadi,dynamic_for_one_step )
class Plant(Model):
    def __init__( self, dimension, dynamic_for_one_step, output_mapping):
        super().__init__( dimension, dynamic_for_one_step, output_mapping)
    def response(self,U,x,noise):
        Y=[]
        for i in range(len(U)):
            x_next = self.dynamic( x, U[i])
            y_out= self.map(x)
            x=x_next
            if noise==1 :
                noise = np.random.rand() * 0.025
                x=x_next+noise
            else:
                x=x_next
            Y.append(np.squeeze(y_out))   

        Y = np.array(Y)
        return [Y,x_next]
class PanSim:
    def __init__(self,simualtor,encoder):
        self.simualtor=simualtor
        self.encoder=encoder
        self.Input=None
        self.Output=None
    def get_initial_state(self,U_init):
        results_agg = []
        inputs_agg = []
        run_options_agg = []
        for i in range (30):
            input_idx,run_options=U_init[i],input_sets[int(U_init[i])]
            results = self.simulator.runForDay(run_options)
            results_agg.append(results)
            inputs_agg.append(input_idx)
            run_options_agg.append(run_options)
        hospitalized_agg=get_results(results_agg)
        [uhist,yhist]=norm_and_unsqueeze(inputs_agg,hospitalized_agg)
        x0 = self.encoder(uhist, yhist)
        x = x0
        self.Input=inputs_agg
        self.Output=hospitalized_agg
        return [x0.detach().numpy()]
    def response(self,U,delta_time):
        results_agg = []
        inputs_agg = []
        run_options_agg = []
        for i in range (len(U)):
            input_idx,run_options=U[i],input_sets[int(U[i])]
            results = self.simulator.runForDay(run_options)
            results_agg.append(results)
            inputs_agg.append(input_idx)
            run_options_agg.append(run_options)
        hospitalized_agg=get_results(results_agg)
        self.Input[-delta_time:]=inputs_agg
        self.Output[-delta_time:]=hospitalized_agg
        return hospitalized_agg
    def get_next_state(self):
        [uhist,yhist]=norm_and_unsqueeze(self.Input,self.Output)
        x0 = self.encoder(uhist, yhist)
        return x0
    