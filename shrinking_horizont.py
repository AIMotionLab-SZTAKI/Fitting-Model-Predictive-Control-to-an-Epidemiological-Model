from opti_problem_for_neural import *
from support import *

def shrinking_MPC(noise,time_horizont,x,grace_time, holding_time,x_init) :
    shrinking_time_horizont=time_horizont
    shrinking_time_jump=holding_time*1
    nerual_models=get_net_models()
    x_first=x
    U=np.empty((1,time_horizont))
    Y=np.empty((1,time_horizont))
    X=np.empty((17*total_time_horizont+int((time_horizont-1)/holding_time)+1,1)) 
    index=0
    u_index=int(np.ceil(shrinking_time_jump/holding_time))
    
    
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
        if shrinking_time_horizont+shrinking_time_jump==time_horizont:
            X=x_init
                   
    Y=np.squeeze(Y)
    U=np.squeeze(U)
    return [Y,U,X]

