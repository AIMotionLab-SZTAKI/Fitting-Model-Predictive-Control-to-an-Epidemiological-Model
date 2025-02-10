from opti_problem import *
from support import *
from models import Plant,PanSim
import itertools
from casadi import vertcat
from parameters import margin


# A függvény shrinking horizon stratégiát valósít meg.
# Az egyes iterációkat az előző iterációban kapott megoldásokkal inicializáljuk, így jobb futási időket érhetünk el.
# A függvény eltárolja az egyes iterációk futási idejét.
def shrinking_MPC(noise_MPC,noise_plant,time_horizont,x,grace_time,model,plant,discr) :
    shrinking_time_horizont=time_horizont
    x_first=x
    time_step=holding_time
    U=np.empty((1,time_horizont))
    X_model=np.empty((model.dim,time_horizont))
    Y_model=np.empty((1,time_horizont))
    Y_real=np.empty((1,time_horizont))
    execution_times = []
    x_init=np.zeros((model.dim+1)*time_horizont+int(np.ceil((time_horizont)/holding_time)))
    
    for i in range(int(np.ceil(time_horizont/holding_time))):
        MyProblem=Problem_With_Grace_time(x_first.T,shrinking_time_horizont,grace_time,holding_time,model,margin)
        if noise_MPC:
            MyProblem.add_noise(shrinking_time_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        execution_times.append(MyProblem.time)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,shrinking_time_horizont,model.dim)
      
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,shrinking_time_horizont,model.dim)
        u_opt_extended=u_extended(u_opt,shrinking_time_horizont)
        if discr==1:
            u_opt_extended=model.rounding(u_opt_extended)
        
        U[:,i*time_step:+time_step*(i+1)]=u_opt_extended[:,:time_step]
        Y_model[:,i*time_step:time_step*(i+1)]=y_opt[:,:time_step]
        X_model[:,i*time_step:time_step*(i+1)]=x_opt[:,:time_step]
        shrinking_time_horizont=shrinking_time_horizont-time_step
        x_init=from_x_u_y_to_solution(x_opt[:,time_step:],u_opt[:,1:],y_opt[:,time_step:],shrinking_time_horizont,model.dim) 
        if isinstance(plant, Plant):
            [Y,x_next]=plant.response(np.squeeze(u_opt_extended[:,:time_step]),x_first,noise_plant)
                  
            x_first=x_next
        if isinstance(plant, PanSim):
            Y=plant.response(np.squeeze(u_opt_extended[:,:time_step]),time_step)
            x_next=plant.get_next_state()
            x_first=x_next            
        Y_real[:,i*time_step:time_step*(i+1)]=Y[:time_step]
        
            
    Y_real=np.squeeze(Y_real)
    Y_model=np.squeeze(Y_model)
    U=np.squeeze(U)
    return [Y_real,Y_model,U,X_model,execution_times]

# A függvény rolling horizon stratégiát valósít meg, hasonlóan az előzőhöz.
# A következő iterációban az előző megoldásokkal inicializáljuk a problémát.
def rolling_MPC(noise_MPC,noise_plant,time_horizont,rolling_horizont,x,grace_time,model,plant,discr):
   
    x_init=np.zeros((model.dim+1)*rolling_horizont+int(np.ceil((rolling_horizont)/holding_time)))
 
    if time_horizont-grace_time>rolling_horizont:
        grace_time_step=0
    else:
        grace_time_step=rolling_horizont-(time_horizont-grace_time)
    x_first=x
    time_step=holding_time
    U=np.empty((1,time_horizont))
    X_model=np.empty((model.dim,time_horizont))
    Y_model=np.empty((1,time_horizont))
    Y_real=np.empty((1,time_horizont))
    execution_times = []
    for i in range(int(np.ceil(time_horizont/holding_time))):

        MyProblem=Problem_With_Grace_time(x_first.T,rolling_horizont,grace_time_step,holding_time,model,margin)
        if noise_MPC:
            MyProblem.add_noise(rolling_horizont)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        execution_times.append(MyProblem.time)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,rolling_horizont,model.dim)
        u_opt_extended=u_extended(u_opt,rolling_horizont)
        if discr==1:
            u_opt_extended=model.rounding(u_opt_extended)
        U[:,i*time_step:+time_step*(i+1)]=u_opt_extended[:,:time_step]
        Y_model[:,i*time_step:time_step*(i+1)]=y_opt[:,:time_step]
        X_model[:,i*time_step:time_step*(i+1)]=x_opt[:,:time_step]
        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,rolling_horizont,model.dim)        
        if isinstance(plant, Plant):
            [Y,x_next]=plant.response(np.squeeze(u_opt_extended[:,:time_step]),x_first,noise_plant)
                  
            x_first=x_next
        if isinstance(plant, PanSim):
            Y=plant.response(np.squeeze(u_opt_extended[:,:time_step]),time_step)
            x_next=plant.get_next_state()
            x_first=x_next            
        
        
        Y_real[:,i*time_step:time_step*(i+1)]=Y[:time_step]
        if rolling_horizont+i*time_step<time_horizont-grace_time:
            grace_time_step=0
        else:
            grace_time_step=rolling_horizont+(i+1)*time_step-(time_horizont-grace_time)
            
    
    Y_real=np.squeeze(Y_real)
    Y_model=np.squeeze(Y_model)
    U=np.squeeze(U)
    return [Y_real,Y_model,U,X_model,execution_times]
# Lehetőségünk van állandó bemenetet adni a rendszerhez visszacsatolással.
# Ez a PanSim és a SUBNET identifikáció minőségének vizsgálatára alkalmas.
def constant_U_values_closed_loop(U,pansim,subnet,time_horizont,x_first):
    time_step=holding_time
    Y_model=np.empty((1,time_horizont))
    Y_real=np.empty((1,time_horizont))
    errors=[]
    for i in range(int(np.ceil(time_horizont/holding_time))):
        [Y_sub_model,x_next_model]=subnet.response(U[:time_step],x_first,0)
        Y_sub_real=pansim.response(U[i*time_step:time_step*(i+1)],time_step)
        x_next_real=pansim.get_next_state()
        x_first=x_next_real 
        error=subnet.map(x_next_real)-subnet.map(x_next_model)
        errors.append(error)           
        Y_real[:,i*time_step:time_step*(i+1)]=Y_sub_real[:time_step]
        Y_model[:,i*time_step:time_step*(i+1)]=Y_sub_model[:time_step]
    final_errors=vertcat(*errors)
    return [Y_real,Y_model,final_errors]
#Visszacsatolás nélküli eset is lehetséges.
def constant_U_values_open_loop(U,pansim,subnet,time_horizont,x_first):
    time_step=holding_time
    Y_model=np.empty((1,time_horizont))
    Y_real=np.empty((1,time_horizont))
   
    for i in range(int(np.ceil(time_horizont/holding_time))):
        [Y_sub_model,x_next_model]=subnet.response(U[:time_step],x_first,0)
        Y_sub_real=pansim.response(U[i*time_step:time_step*(i+1)],time_step)
        x_first=x_next_model        
        Y_real[:,i*time_step:time_step*(i+1)]=Y_sub_real[:time_step]
        Y_model[:,i*time_step:time_step*(i+1)]=Y_sub_model[:time_step]
    return [Y_real,Y_model]
