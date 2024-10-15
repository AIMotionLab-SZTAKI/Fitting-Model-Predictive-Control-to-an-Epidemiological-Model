from opti_problem import *
from parameters import x0,total_time_horizont,grace_time, holding_time
from support import *

def shrinking_MPC(noise):
    control_time=int((total_time_horizont-1)/holding_time)+1
    shrinking_time_horizont=total_time_horizont
    shrinking_time_jump=holding_time*1
    nerual_models=get_net_models()
    x_first=x0
    U=np.empty((1,total_time_horizont))
    Y=np.empty((1,total_time_horizont))
    index=0
    u_index=int(np.ceil(shrinking_time_jump/holding_time))
    x_init=np.zeros((17*shrinking_time_horizont+control_time,1)) 
    
    while shrinking_time_horizont>holding_time:
        MyProblem=Problem_With_Grace_time(nerual_models['f'],nerual_models['h'],x_first,shrinking_time_horizont,grace_time,holding_time,cs.sumsqr,system_step)
        if noise:
            MyProblem.add_noise(shrinking_time_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,shrinking_time_horizont)
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,shrinking_time_horizont)
        u_opt_extended=u_extended(u_opt,shrinking_time_horizont)
        U[:,index:]=u_opt_extended
        Y[:,index:]=y_opt
        shrinking_time_horizont=shrinking_time_horizont-shrinking_time_jump
        index=index+shrinking_time_jump
        if shrinking_time_horizont>=holding_time:
            x_first=x_opt[:,holding_time]
            x_init=from_x_u_y_to_solution(x_opt[:,shrinking_time_jump:],u_opt[:,u_index:],y_opt[:,shrinking_time_jump:],shrinking_time_horizont)        
    Y=np.squeeze(Y)
    U=np.squeeze(U)
    return [Y,U]

