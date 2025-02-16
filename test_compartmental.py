from models import Model,Plant,PanSim
from strategies import rolling_MPC,shrinking_MPC
from compartmental_model import runge_kutta_4_step,compartmental_model_mapping

from support import visualize_Y_quess_vs_Y_real,visualize_execution_time
from parameters import *

comparmental_plant=Plant(8,runge_kutta_4_step,compartmental_model_mapping)
comparmental_model=Model(8,runge_kutta_4_step,compartmental_model_mapping)

[Y_real,Y_model,U,X,time]=shrinking_MPC(0,0,total_time_horizont_extended,x0_comparmental,grace_time_extended,comparmental_model,comparmental_plant,1)

visualize_Y_quess_vs_Y_real(Y_model*real_population,Y_real*real_population,U)
visualize_execution_time(time)
