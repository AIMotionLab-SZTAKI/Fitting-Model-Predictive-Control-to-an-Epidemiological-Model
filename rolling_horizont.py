from opti_problem import *
from parameters import x0,total_time_horizont,grace_time,holding_time
from support import *
def rolling_MPC(noise):
    rolling_horizont=(total_time_horizont-grace_time)-((total_time_horizont-grace_time)%holding_time)
    control_time=int((rolling_horizont-1)/holding_time)+1
    nerual_models=get_net_models()
    x_first=x0
    grace_time_step=0
    time_step=holding_time    
    U=np.empty((1,total_time_horizont))
    Y=np.empty((1,total_time_horizont))

    x_init=np.zeros((17*rolling_horizont+control_time,1)) 
    while grace_time_step <= grace_time+time_step:
        MyProblem=Problem_With_Grace_time(nerual_models['f'],nerual_models['h'],x_first,rolling_horizont,grace_time_step,holding_time, cs.sumsqr,system_step)
        if noise:
            MyProblem.add_noise(rolling_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,rolling_horizont)
        u_opt_extended=u_extended(u_opt,rolling_horizont)
        if grace_time_step>grace_time:
            U[:,-grace_time_step:]=u_opt_extended[:,-grace_time_step:]
            Y[:,-grace_time_step:]=y_opt[:,-grace_time_step:]
        else:
            U[:,grace_time_step:rolling_horizont+grace_time_step]=u_opt_extended[:,0:rolling_horizont]
            Y[:,grace_time_step:rolling_horizont+grace_time_step]=y_opt[:,0:rolling_horizont]
  
  
        x_first=x_opt[:,time_step]
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont)
        grace_time_step=grace_time_step+time_step


    Y=np.squeeze(Y)
    U=np.squeeze(U)
    return [Y,U]


