from opti_problem import *
from support import *
from models import Plant,PanSim
import itertools
def shrinking_MPC(noise,time_horizont,x,grace_time,model,plant,discr) :
    shrinking_time_horizont=time_horizont
    x_first=x
    time_step=holding_time
    U=np.empty((1,time_horizont))
    Y_model=np.empty((1,time_horizont))
    Y_real=np.empty((1,time_horizont))
    execution_times = []
    x_init=np.zeros((model.dim+1)*time_horizont+int(np.ceil((time_horizont)/holding_time)))
    for i in range(int(np.ceil(time_horizont/holding_time))):
        MyProblem=Problem_With_Grace_time(x_first.T,shrinking_time_horizont,grace_time,holding_time,model)
        if noise:
            MyProblem.add_noise(shrinking_time_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        execution_times.append(MyProblem.time)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,shrinking_time_horizont,model.dim)
      
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,shrinking_time_horizont,model.dim)
        u_opt_extended=u_extended(u_opt,shrinking_time_horizont)
        if discr==1:
            u_opt_extended=np.round(u_opt_extended)
        
        U[:,i*time_step:+time_step*(i+1)]=u_opt_extended[:,:time_step]
        Y_model[:,i*time_step:time_step*(i+1)]=y_opt[:,:time_step]
        shrinking_time_horizont=shrinking_time_horizont-time_step
        x_init=from_x_u_y_to_solution(x_opt[:,time_step:],u_opt[:,1:],y_opt[:,time_step:],shrinking_time_horizont,model.dim) 
        if isinstance(plant, Plant):
            [Y,x_next]=plant.response(np.squeeze(u_opt_extended[:,:time_step]),x_first,0)
                  
            x_first=x_next
        if isinstance(plant, PanSim):
            Y=plant.response(np.squeeze(u_opt_extended[:,:time_step]),time_step)
            x_next=plant.get_next_state()
            x_first=x_next            
            
        Y_real[:,i*time_step:time_step*(i+1)]=Y[:time_step]
        
            
    Y_real=np.squeeze(Y_real)
    Y_model=np.squeeze(Y_model)
    U=np.squeeze(U)
    return [Y_real,Y_model,U,execution_times]
def rolling_MPC(noise,time_horizont,rolling_horizont,x,grace_time,model,plant,discr):
   
    x_init=np.zeros((model.dim+1)*rolling_horizont+int(np.ceil((rolling_horizont)/holding_time)))

    if time_horizont-grace_time>rolling_horizont:
        grace_time_step=0
    else:
        grace_time_step=rolling_horizont-(time_horizont-grace_time)
    x_first=x
    time_step=holding_time
    U=np.empty((1,time_horizont))
    Y_model=np.empty((1,time_horizont))
    Y_real=np.empty((1,time_horizont))
    execution_times = []
    for i in range(int(np.ceil(time_horizont/holding_time))):
        MyProblem=Problem_With_Grace_time(x_first.T,rolling_horizont,grace_time_step,holding_time,model)
        if noise:
            MyProblem.add_noise(rolling_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        execution_times.append(MyProblem.time)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,rolling_horizont,model.dim)
        u_opt_extended=u_extended(u_opt,rolling_horizont)
        if discr==1:
            u_opt_extended=np.round(u_opt_extended)
        
        U[:,i*time_step:+time_step*(i+1)]=u_opt_extended[:,:time_step]
        Y_model[:,i*time_step:time_step*(i+1)]=y_opt[:,:time_step]
        [Y,x_next]=plant.response(np.squeeze(u_opt_extended[:,:time_step]),x_first,0)
        
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont,model.dim)        
        x_first=x_next
        Y_real[:,i*time_step:time_step*(i+1)]=Y[:time_step]
        if rolling_horizont+i*time_step<time_horizont-grace_time:
            grace_time_step=0
        else:
            grace_time_step=rolling_horizont+(i+1)*time_step-(time_horizont-grace_time)
            
    flattened_vector=list(itertools.chain(*execution_times))
    Y_real=np.squeeze(Y_real)
    Y_model=np.squeeze(Y_model)
    U=np.squeeze(U)
    return [Y_real,Y_model,U]


