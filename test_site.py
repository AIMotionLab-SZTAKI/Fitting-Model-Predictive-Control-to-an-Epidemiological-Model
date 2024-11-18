from models import Model
from strategies import rolling_MPC,shrinking_MPC
import numpy as np
from opti_problem import *
from compartmental_model import runge_kutta_4_step,compartmental_model_mapping
from torch_nets import system_step_neural,output_mapping_neural
from simualte_with_compartmental import real_model_simulation
from functools import partial
from compartmental_model import dydt_numpy, dydt_casadi, runge_kutta_4_step, compartmental_model_mapping
x0=np.array([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
                    0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])
model=Model(16,system_step_neural,output_mapping_neural)
# x0 = np.array([(real_population-real_latent)/real_population, real_latent/real_population,0.,0.,0.,0.,0.,0.])
# model = Model(
#     dimension=8,
#     dynamic_for_one_step=partial(runge_kutta_4_step, dydt_casadi),
#     output_mapping=compartmental_model_mapping
# )
x_init=np.zeros((model.dim+1)*total_time_horizont_extended+int(np.ceil((total_time_horizont_extended)/holding_time)))


[Y,U,X_final]=shrinking_MPC(0,total_time_horizont_extended,x0.T,grace_time_extended,x_init,model)
visualize_Y_vs_U(Y,U)
# u_opt=u_extended(u,total_time_horizont_extended)
# u_sub_opt=norm_round(u_opt)
# x_opt=x_opt*real_population
# y_real=real_model_simulation(np.squeeze(u_sub_opt),total_time_horizont_extended,x0)
# visualize_comapartmental(x_opt,u_opt,y_real)    