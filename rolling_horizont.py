from opti_problem import *
from support import *
def rolling_MPC(noise,time_horizont,x,grace_time, holding_time,x_init):
    rolling_horizont = (time_horizont-grace_time) - ((time_horizont-grace_time)%holding_time)
    nerual_models = get_net_models()
    x_first = x
    grace_time_step = 0
    index = 0
    time_step = holding_time
    U = np.empty((1,time_horizont))
    Y = np.empty((1,time_horizont))
    toggle = 0
    while grace_time_step <= grace_time:
        MyProblem=Problem_With_Grace_time(nerual_models['f'],nerual_models['h'],x_first,rolling_horizont,grace_time_step,holding_time, cs.sumsqr,system_step)
        if noise:
            MyProblem.add_noise(rolling_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,rolling_horizont)
        u_opt_extended=u_extended(u_opt,rolling_horizont)
        
        U[:,index:rolling_horizont+index]=u_opt_extended[:,0:rolling_horizont]
        Y[:,index:rolling_horizont+index]=y_opt[:,0:rolling_horizont]
        
        x_first=x_opt[:,time_step]
        print(grace_time_step, grace_time + time_step)
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont)
        if toggle==0 and grace_time%holding_time!=0:
            grace_time_step=grace_time%holding_time
        else:
            grace_time_step=grace_time_step+time_step
        index=index+time_step
        toggle=1
    X_final=from_x_u_y_to_solution(x_opt[:,time_step:],u_opt[:,1:],y_opt[:,time_step:],rolling_horizont-time_step)
    
  
    
    Y=np.squeeze(Y)
    U=np.squeeze(U)
  
 
    return [Y,U,X_final]


