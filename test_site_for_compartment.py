from opti_problem_for_compartment import *
from parameters_for_compartment import *
from comparment_model import *
import numpy as np
MyProblem=Problem_With_Grace_time(x0,time_horizont,grace_time,holding_time,cs.sumsqr,runge_kutta_4_step)
x=np.zeros(time_horizont*10)
MySolution=MyProblem.get_soultion('ipopt',x)
[x_opt,u_opt,y_opt]=solution_prcessing(MySolution,time_horizont)
visualize(x_opt,u_opt)


# u_opt=u_extended(u_opt,time_horizont)
# U_extended=np.zeros(time_extended)
# U_extended[0:len(u_opt)]=norm_round(u_opt)

# time_horizont=time_extended
# Y_extended=real_model_simulation(U_extended,dydt_numpy,time_extended)
# visualize_the_system(Y_extended)
