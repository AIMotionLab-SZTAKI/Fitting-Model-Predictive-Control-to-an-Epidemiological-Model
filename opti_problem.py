import casadi as cs
import numpy as np
from torch_nets import system_step,get_net_models
from matplotlib import pyplot as plt
from parameters import min_control_value,max_control_value,hospital_capacity,x0

class Problem:
    def __init__(self,casadi_net_f, casadi_net_h,x0,time_horizont,objective_function,dynamic_for_one_step):
    
        self.x = cs.MX.sym('x', 16,time_horizont)
        self.u = cs.MX.sym('u', 1,time_horizont)
        self.y = cs.MX.sym('y', 1,time_horizont)

        self.objective = objective_function(self.u)
        self.system_step=dynamic_for_one_step
        constraints_for_step_state = []
        constraints_for_output=[]
        constraints_for_step_state.append( self.x[:, 0] -x0)
        
    
        for i in range(time_horizont):
            if i==time_horizont-1:
                [x_next, y_out] = self.system_step( casadi_net_f,casadi_net_h, self.x[:,i].T, self.u[:, i].T )
                constraints_for_output.append( self.y[:,i]-y_out )
            else:
                [x_next, y_out] = self.system_step( casadi_net_f,casadi_net_h, self.x[:,i].T, self.u[:, i].T )
                constraints_for_step_state.append( self.x[:, i+1] - x_next.T )
                constraints_for_output.append( self.y[:,i]-y_out )
            
            
            
        step_state = cs.vertcat( *constraints_for_step_state )
        step_output = cs.vertcat( *constraints_for_output )
        g = cs.vertcat( step_state, step_output )

        for i in range(time_horizont):
            g = cs.vertcat(g, self.u[:, i]) 
            g = cs.vertcat(g, self.y[:, i])  
                
        nlp = { 'x': cs.vertcat(cs.vec(self.x), cs.vec(self.u),cs.vec(self.y)), 
                'f': self.objective,  
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

        self.nlp=nlp
        self.floor_constraints=lbg
        self.ceilloing_constraints=ubg
        
        
    def get_soultion(self,solver_type,x_init):
        solver = cs.nlpsol( 'solver', solver_type, self.nlp )
        solution = solver( lbg=self.floor_constraints, ubg=self.ceilloing_constraints,x0=x_init)  
        return solution


def visualize(solution,time_horizont):
        
    [x_opt,y_opt,u_opt]=from_solution_to_x_u_y(solution,time_horizont)
    x_opt = np.array(x_opt)
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
    
def from_solution_to_x_u_y(solution,time_horizont):
    solution_x_u_y = solution[ 'x' ]  
    x_opt = solution_x_u_y[ :16*time_horizont].reshape(( 16, time_horizont ))  
    u_opt = solution_x_u_y[ 16*time_horizont:16*time_horizont + time_horizont ].reshape(( 1, time_horizont ))   
    y_opt = solution_x_u_y[ 16*time_horizont+time_horizont: ].reshape((1, time_horizont))
    return [x_opt,u_opt,y_opt]
def from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_horizont):
    x_faltten=cs.reshape(x_opt,time_horizont*16,1)
    u_faltten=cs.reshape(u_opt,time_horizont,1)
    y_faltten=cs.reshape(y_opt,time_horizont,1)
    result=cs.vertcat(x_faltten,u_faltten,y_faltten)
    return result