import numpy as np
import pyPanSim as sp
from parameters import *
from simulate import simulate_with_nerual_network,simualte_with_PanSim
from support import *
def closed_loop_shr_neural(noise,noise_MPC,MPC_type,x0,disc):
    U_system=np.empty((total_time_horizont_extended))
    Y_system=np.empty((total_time_horizont_extended))
    
    delta_time=holding_time
    time_horizont=total_time_horizont_extended
    x=x0
    x_init=np.zeros((17*time_horizont+int((time_horizont-1)/holding_time)+1,1))
    for i in range(int(np.ceil(total_time_horizont_extended/delta_time))):
                
            if time_horizont>holding_time:
                [Y,U,X]=MPC_type(noise_MPC,time_horizont,x,grace_time,holding_time,x_init)
                
                x_init=X
                if disc==1:
                    U_calculated=np.round(U)
                    U_calculated[U_calculated<0]=0
                else:
                     U_calculated=U
            else:
                U=np.zeros(time_horizont)
                U_calculated=U
       
            [Y_sim,x_next]=simulate_with_nerual_network(U_calculated[0:delta_time],x.T,noise)
            x=x_next.T
            U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
            Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
            time_horizont=time_horizont-delta_time
    return [Y_system,U_system]
    


def closed_loop_roll_neural(noise,noise_MPC,MPC_type,x0,disc):
    U_system=np.empty((total_time_horizont_extended))
    Y_system=np.empty((total_time_horizont_extended))
    
    delta_time=holding_time
    time_horizont=total_time_horizont_extended
    x=x0
    
    i=0
    x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
    i=0
    while (time_horizont>2*grace_time):
                
            if time_horizont>holding_time*2:
                [Y,U,X]=MPC_type(noise_MPC,time_horizont,x,grace_time,holding_time,x_init)
                x_init=X
                if disc==1:
                    U_calculated=np.round(U)
                    U_calculated[U_calculated<0]=0
                else:
                    U_calculated=U
        
               
         
            [Y_sim,x_next]=simulate_with_nerual_network(U_calculated[0:delta_time],x.T,noise)
            x=x_next.T
            U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
            Y_system[i*delta_time:(i+1)*delta_time]=Y_sim
            time_horizont=time_horizont-delta_time
            i=i+1
    [Y_sim,x_next]=simulate_with_nerual_network(U_calculated[2*-grace_time:],x.T,noise)
    U_system[2*-grace_time:]=U_calculated[2*-grace_time:]
    Y_system[2*-grace_time:]=Y_sim
    return [Y_system,U_system]

def closed_loop_shr_PanSim(MPC_type,noise_MPC,U_init):
    U_system=np.empty((total_time_horizont_extended))
    Y_model=np.empty((total_time_horizont_extended))
    Y_real=np.empty((total_time_horizont_extended))
    Raw_InputData=np.empty((total_time_horizont_extended+30))
    Raw_OutputData=np.empty((total_time_horizont_extended+30))
    delta_time=holding_time
    time_horizont=total_time_horizont_extended
    x_init=np.zeros((17*time_horizont+int((time_horizont-1)/holding_time)+1,1))
    simulator = sp.SimulatorInterface()
    simulator.initSimulation(init_options)
    encoder=get_encoder()
    [x,Raw_InputData[0:30],Raw_OutputData[0:30]]=get_init_state(simulator,U_init,encoder)
    for i in range(int(np.ceil(total_time_horizont_extended/delta_time))):
                
            if time_horizont>holding_time:
                [Y,U,X]=MPC_type(noise_MPC,time_horizont,x.T,grace_time,holding_time,x_init)
                
                x_init=X
               
                U_calculated=np.round(U)
                U_calculated[U_calculated<0]=0
               
            else:
                U=np.zeros(time_horizont)
                U_calculated=U
            [New_Input,New_Output]=simualte_with_PanSim(simulator,U_calculated[0:delta_time])
            Raw_InputData[30+i*delta_time:30+(i+1)*delta_time]=New_Input
            Raw_OutputData[30+i*delta_time:30+(i+1)*delta_time]=New_Output
            [uhist,yhist]=norm_and_unsqueeze(Raw_InputData[(i+1)*delta_time:30+(i+1)*delta_time],Raw_OutputData[(i+1)*delta_time:30+(i+1)*delta_time])
            x0 = encoder(uhist, yhist)
            x=x0.detach().numpy()
            U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
            Y_model[i*delta_time:(i+1)*delta_time]=Y[0:delta_time]
            Y_real[i*delta_time:(i+1)*delta_time]=Raw_OutputData[i*delta_time+30:(i+1)*delta_time+30]
            time_horizont=time_horizont-delta_time
    return [Y_model,Y_real,U_system]
def closed_loop_roll_PanSim(MPC_type,noise_MPC,U_init):
        U_system=np.empty((total_time_horizont_extended))
        Y_model=np.empty((total_time_horizont_extended))
        Y_real=np.empty((total_time_horizont_extended))
        Raw_InputData=np.empty((total_time_horizont_extended+30))
        Raw_OutputData=np.empty((total_time_horizont_extended+30))
        delta_time=holding_time
        time_horizont=total_time_horizont_extended
        x_init=np.zeros((17*rolling_horizont+int((rolling_horizont-1)/holding_time)+1,1))
        simulator = sp.SimulatorInterface()
        simulator.initSimulation(init_options)
        encoder=get_encoder()
        i=0
        [x,Raw_InputData[0:30],Raw_OutputData[0:30]]=get_init_state(simulator,U_init,encoder)
        while (time_horizont>2*grace_time):
                    
                if time_horizont>holding_time*2:
                    [Y,U,X]=MPC_type(noise_MPC,time_horizont,x.T,grace_time,holding_time,x_init)
                    x_init=X
                    U_calculated=np.round(U)
                    U_calculated[U_calculated<0]=0
            
                [New_Input,New_Output]=simualte_with_PanSim(simulator,U_calculated[0:delta_time])
                Raw_InputData[30+i*delta_time:30+(i+1)*delta_time]=New_Input
                Raw_OutputData[30+i*delta_time:30+(i+1)*delta_time]=New_Output
                [uhist,yhist]=norm_and_unsqueeze(Raw_InputData[(i+1)*delta_time:30+(i+1)*delta_time],Raw_OutputData[(i+1)*delta_time:30+(i+1)*delta_time])
                x0 = encoder(uhist, yhist)
                x=x0.detach().numpy()
                U_system[i*delta_time:(i+1)*delta_time]=U_calculated[0:delta_time]
                Y_model[i*delta_time:(i+1)*delta_time]=Y[0:delta_time]
                Y_real[i*delta_time:(i+1)*delta_time]=Raw_OutputData[i*delta_time+30:(i+1)*delta_time+30]
                time_horizont=time_horizont-delta_time
                i=i+1
        [New_Input,New_Output]=simualte_with_PanSim(simulator,U_calculated[2*-grace_time:])
        U_system[2*-grace_time:]=U_calculated[2*-grace_time:]
        Y_model[2*-grace_time:]=Y[2*-grace_time:]
        Y_real[2*-grace_time:]=New_Output
        return [Y_model,Y_real,U_system]