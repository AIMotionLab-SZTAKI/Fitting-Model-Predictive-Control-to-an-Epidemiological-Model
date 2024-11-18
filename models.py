
from functools import partial
from torch_nets import system_step_neural,get_net_models
from compartmental_model import runge_kutta_4_step,dydt_casadi
import numpy as np
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
