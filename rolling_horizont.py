from opti_problem_for_neural import *
from support import *
from parameters_for_nerual import rolling_horizont,grace_time_extended
def rolling_MPC(noise,time_horizont,x,grace_time_param, holding_time,x_init):
    nerual_models = get_net_models()
    x_first = x
    time_step = holding_time
    grace_time_step=grace_time_param-grace_time_extended
    index = 0
    U = np.empty((1,time_horizont))
    Y = np.empty((1,time_horizont))
    while grace_time_step <= grace_time_param:
        MyProblem=Problem_With_Grace_time(nerual_models['f'],nerual_models['h'],x_first,rolling_horizont,grace_time_step,holding_time, cs.sumsqr,system_step)
        if noise:
            MyProblem.add_noise(rolling_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,rolling_horizont)
        u_opt_extended=u_extended(u_opt,rolling_horizont)
        U[:,index*time_step:rolling_horizont+time_step*(index)]=u_opt_extended[:,:rolling_horizont]
        Y[:,index*time_step:rolling_horizont+time_step*(index)]=y_opt[:,:rolling_horizont]
        
        x_first=x_opt[:,time_step]
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont)
       
        grace_time_step=grace_time_step+time_step
        index=index+1
        
        X_final=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont)
    
  
    
    Y=np.squeeze(Y)
    U=np.squeeze(U)

 
    return [Y,U,X_final]


