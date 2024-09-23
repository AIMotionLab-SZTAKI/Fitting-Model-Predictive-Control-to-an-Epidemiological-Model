import casadi as cs
import numpy as np
from torch_nets import system_step
from torch_nets import casadi_res_net
from ann_utils import default_state_net,default_output_net
import torch
from matplotlib import pyplot as plt
# Paraméterek
time_horizont = 180
hospital_capacity = 400
min_control_value = 0
max_control_value = 16 

def get_net_models():
    f = default_state_net(nu = 1, nx = 16)
    f.load_state_dict(state_dict=torch.load("f_dict.pt"))
    f.eval()
    h = default_output_net(nx=16, ny=1)
    h.load_state_dict(state_dict=torch.load("h_dict.pt"))
    h.eval()
    casadi_net_f = casadi_res_net(f.net)
    casadi_net_h = casadi_res_net(h.net)
    return [casadi_net_f,casadi_net_h]


def get_controller(casadi_net_f, casadi_net_h):
    x = cs.SX.sym('x', 16,time_horizont)
    u = cs.SX.sym('u', 1,time_horizont)
    y = cs.SX.sym('y', 1,time_horizont)

    objective = cs.sumsqr(u)

    x0=cs.SX([[-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
            0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506]])

    constraints_for_step_state = []
    constraints_for_output=[]
    constraints_for_step_state.append( x[:, 0] -x0.T )
    

    for i in range(time_horizont):
        if i==time_horizont-1:
            [x_next, y_out] = system_step( casadi_net_f,casadi_net_h, x[:,i].T, u[:, i].T )
            constraints_for_output.append( y[:,i]-y_out )
        else:
            [x_next, y_out] = system_step( casadi_net_f,casadi_net_h, x[:,i].T, u[:, i].T )
            constraints_for_step_state.append( x[:, i+1] - x_next.T )
            constraints_for_output.append( y[:,i]-y_out )
        
        
        
    step_state = cs.vertcat( *constraints_for_step_state )
    step_output = cs.vertcat( *constraints_for_output )
    g = cs.vertcat( step_state, step_output )

    for i in range(time_horizont):
        g = cs.vertcat(g, u[:, i]) 
        g = cs.vertcat(g, y[:, i])  
            
    nlp = { 'x': cs.vertcat(cs.vec(x), cs.vec(u),cs.vec(y)), 
            'f': objective,  
            'g': g}  

    lbg=np.zeros(17*time_horizont+2*time_horizont)
    ubg=np.zeros(17*time_horizont+2*time_horizont)
    for i in range (17*time_horizont+2*time_horizont):
        if i<17*time_horizont:
            lbg[i]=0
            ubg[i]=0
        else:
            if(i%2==0):
                lbg[i]=min_control_value
                ubg[i]=max_control_value
            else:
                lbg[i]=0
                ubg[i]=hospital_capacity

    return [nlp,lbg,ubg]

def visualize(solution):
        
    solution_x_u_y = solution[ 'x' ]  
    x_opt = solution_x_u_y[ :16*time_horizont].reshape(( 16, time_horizont ))  
    u_opt = solution_x_u_y[ 16*time_horizont:16*time_horizont + time_horizont ].reshape(( 1, time_horizont ))   
    y_opt = solution_x_u_y[ 16*time_horizont+time_horizont: ].reshape((1, time_horizont))
    y_opt = np.array(y_opt)
    u_opt = np.array(u_opt)
    y_opt=np.squeeze(y_opt)
    u_opt=np.squeeze(u_opt)


    plt.subplot(2,1,1)
    plt.grid()
    plt.plot(y_opt)
    plt.xlabel("Time [days]")
    plt.ylabel("Hospitalized people [patients]")
    plt.subplot(2,1,2)
    plt.plot(u_opt,'.')
    plt.grid()
    plt.xlabel("Time [days]")
    plt.ylabel("Control input")
    plt.show()

    

casadi_net_f,casadi_net_h=get_net_models()

nlp,lbg,ubg=get_controller(casadi_net_f,casadi_net_h)

solver = cs.nlpsol( 'solver', 'ipopt', nlp )

solution = solver( lbg=lbg, ubg=ubg )  

visualize(solution)
