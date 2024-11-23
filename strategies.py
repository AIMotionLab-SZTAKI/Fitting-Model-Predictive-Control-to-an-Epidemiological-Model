from opti_problem import *
from support import *
def shrinking_MPC(noise,time_horizont,x,grace_time,x_init,model) :
    shrinking_time_horizont=time_horizont
    shrinking_time_jump=holding_time*1
    x_first=x
    execution_times = []
    U=np.empty((1,time_horizont))
    Y=np.empty((1,time_horizont))
    X=np.empty(((model.dim+1)*time_horizont+int((time_horizont-1)/holding_time)+1,1)) 
    index=0
    u_index=int(np.ceil(shrinking_time_jump/holding_time))
    
    
    while shrinking_time_horizont>holding_time:
        MyProblem=Problem_With_Grace_time(x_first,shrinking_time_horizont,grace_time,holding_time,cs.sumsqr,model)
        if noise:
            MyProblem.add_noise(shrinking_time_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        execution_times.append(MyProblem.time)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,shrinking_time_horizont,model.dim)
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,shrinking_time_horizont,model.dim)
        u_opt_extended=u_extended(u_opt,shrinking_time_horizont)
        U[:,index:]=u_opt_extended
        Y[:,index:]=y_opt
        shrinking_time_horizont=shrinking_time_horizont-shrinking_time_jump
        index=index+shrinking_time_jump
        if shrinking_time_horizont>=holding_time:
            x_first=x_opt[:,holding_time]
            x_init=from_x_u_y_to_solution(x_opt[:,shrinking_time_jump:],u_opt[:,u_index:],y_opt[:,shrinking_time_jump:],shrinking_time_horizont,model.dim) 
        if shrinking_time_horizont+shrinking_time_jump==time_horizont:
            X=x_init
    Y=np.squeeze(Y)
    U=np.squeeze(U)
    return [Y,U,X,execution_times]
def rolling_MPC(noise,time_horizont,x,grace_time_param,x_init,model):
   
    x_first = x
    time_step = holding_time*1
    grace_time_step=grace_time_param
    U = np.empty((1,time_horizont))
    Y = np.empty((1,time_horizont))
    execution_times = []
    X=np.empty(((model.dim+1)*time_horizont+int((time_horizont-1)/holding_time)+1,1)) 
    for i in range(int(np.ceil(time_horizont/holding_time))):
        MyProblem=Problem_With_Grace_time(x_first,time_horizont,grace_time_step,holding_time,cs.sumsqr,model)
        if noise:
            MyProblem.add_noise(time_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        execution_times.append(MyProblem.time)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,time_horizont,model.dim)
        u_opt_extended=u_extended(u_opt,time_horizont)
        U[:,i*time_step:+time_step*(i+1)]=u_opt_extended[:,:time_step]
        Y[:,i*time_step:time_step*(i+1)]=y_opt[:,:time_step]
        
        x_first=x_opt[:,time_step]
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_horizont,model.dim)
       
        grace_time_step=grace_time_step+time_step
        
        X=from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_horizont,model.dim)
    Y=np.squeeze(Y)
    U=np.squeeze(U)
    return [Y,U,X,execution_times]


