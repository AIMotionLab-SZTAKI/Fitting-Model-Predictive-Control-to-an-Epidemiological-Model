from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC
import numpy as np
from torch_nets import system_step_neural,output_mapping_neural,get_encoder
import pyPanSim as sp
from support import visualize_Y_quess_vs_Y_real, visualize_execution_time
from parameters import *

U_init=np.ones(30)*0
encoder=get_encoder()
simulator = sp.SimulatorInterface()
pansim=PanSim(simulator,encoder)
subnet=Model(16,system_step_neural,output_mapping_neural)
x0=pansim.get_initial_state(U_init)
[Y_real,Y_model,U,X,time]=shrinking_MPC(noise_MPC=0, noise_plant=0, time_horizont=total_time_horizont_extended, x=x0,
                                        grace_time=grace_time_extended, model=subnet, plant=pansim, discr=1)
visualize_Y_quess_vs_Y_real(Y_model,Y_real,U)
print(np.average(time))
visualize_Y_quess_vs_Y_real(Y_model*real_population,Y_real*real_population,U)
visualize_execution_time(time)
print(np.sum(time))
