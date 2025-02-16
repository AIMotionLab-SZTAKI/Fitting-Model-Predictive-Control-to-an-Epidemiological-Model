from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC
import numpy as np
from torch_nets import system_step_neural,output_mapping_neural,get_encoder
import pyPanSim as sp
from support import visualize_Y_quess_vs_Y_real
from parameters import *

U_init=np.ones(30)*0
encoder=get_encoder()
simulator = sp.SimulatorInterface()
pansim=PanSim(simulator,encoder)
subnet=Model(16,system_step_neural,output_mapping_neural)
x0=pansim.get_initial_state(U_init)
[Y_real,Y_model,U,time]=shrinking_MPC(0,total_time_horizont_extended,x0,grace_time_extended,subnet,pansim,1)
visualize_Y_quess_vs_Y_real(Y_model,Y_real,U)
