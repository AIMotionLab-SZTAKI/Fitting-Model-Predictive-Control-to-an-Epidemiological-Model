from opti_problem import *
from parameters import x0,total_time_horizont,grace_time
from support import *


def rolling_MPC():
    nerual_models=get_net_models()
    x_first=x0
    rolling_horizont=total_time_horizont-grace_time
    x_init=np.zeros((18*(rolling_horizont),1))
    U=np.empty((1,total_time_horizont))
    Y=np.empty((1,total_time_horizont))
    grace_time_step=0
    time_step=10
    while grace_time_step<grace_time+time_step:
        MyProblem=Problem_With_Grace_time(nerual_models['f'],nerual_models['h'],x_first,rolling_horizont,grace_time_step,cs.sumsqr,system_step)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,rolling_horizont)
        
        U[:,grace_time_step:rolling_horizont+grace_time_step]=u_opt[:,0:rolling_horizont]
        Y[:,grace_time_step:rolling_horizont+grace_time_step]=y_opt[:,0:rolling_horizont]
  
  
        x_first=x_opt[:,time_step]
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont)
        grace_time_step=grace_time_step+time_step
        

    Y=np.squeeze(Y)
    U=np.squeeze(U)
    return [Y,U]


