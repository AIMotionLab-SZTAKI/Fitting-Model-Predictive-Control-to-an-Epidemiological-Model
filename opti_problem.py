import casadi as cs
import numpy as np
from support import *
from parameters import min_control_value, max_control_value, hospital_capacity
from support import *

class Problem:
    def __init__( self, casadi_net_f, casadi_net_h, x0, time_horizont, holding_time, objective_function, dynamic_for_one_step ):
        control_time=int((time_horizont-1)/holding_time)+1
        self.x = cs.MX.sym( 'x', 16,time_horizont )
        self.u = cs.MX.sym( 'u', 1,control_time )
        self.y = cs.MX.sym( 'y', 1,time_horizont )
        self.objective = objective_function( self.u )
        self.system_step = dynamic_for_one_step
        constraints_for_step_state = [ ]
        constraints_for_output = [ ]
        constraints_for_step_state.append( self.x[:, 0] - x0 )
        
    
      
        for i in range( time_horizont ):
            if i == time_horizont - 1:
                [ x_next, y_out ] = self.system_step( casadi_net_f,casadi_net_h, self.x[ :, i ].T, self.u[ :, int(i/holding_time) ].T ) 
                constraints_for_output.append( self.y[ :, i ] - y_out )
            else:
                [ x_next, y_out ] = self.system_step( casadi_net_f,casadi_net_h, self.x[ :, i ].T, self.u[ :, int(i/holding_time) ].T ) 
                constraints_for_step_state.append( self.x[ : , i + 1 ] - x_next.T )
                constraints_for_output.append( self.y[ :, i ] - y_out)
            
            
        step_state = cs.vertcat( *constraints_for_step_state )
        step_output = cs.vertcat( *constraints_for_output )
        g = cs.vertcat( step_state, step_output )

        for i in range( control_time ):
            g = cs.vertcat( g, self.u[ :, i ])
        for i in range(time_horizont): 
            g = cs.vertcat( g, self.y[ :, i ])  
                
        nlp = { 'x': cs.vertcat( cs.vec( self.x ), cs.vec( self.u ), cs.vec( self.y )), 
                'f': self.objective,  
                'g': g}  
        
        lbg = np.zeros( 17*time_horizont + control_time + time_horizont )
        ubg=np.zeros( 17 * time_horizont + control_time + time_horizont )
        for i in range ( 17 * time_horizont +  control_time + time_horizont ):
            if i < 17 * time_horizont:
                lbg[ i ] = 0
                ubg[ i ] = 0
            else:
                if i< 17 * time_horizont + control_time:
                    lbg[ i ] = min_control_value
                    ubg[ i ] = max_control_value
                
                else:
                    lbg[ i ] = 0
                    ubg[ i ] = hospital_capacity

        self.nlp = nlp
        self.floor_constraints = lbg
        self.ceilloing_constraints = ubg
        
    def add_noise (self,time_horizont):
        for i in range (17*time_horizont):
            noise = np.random.rand() * 0.025
            self.nlp['g'][i]=self.nlp['g'][i]-noise

    def get_soultion( self, solver_type, x_init ):
        solver = cs.nlpsol( 'solver', solver_type, self.nlp )
        solution = solver( lbg = self.floor_constraints, ubg = self.ceilloing_constraints,x0 = x_init )  
        return solution
class Problem_With_Grace_time(Problem):
      def __init__( self, casadi_net_f, casadi_net_h, x0, time_horizont, grace_time, holding_time, objective_function, dynamic_for_one_step ):
        super().__init__( casadi_net_f, casadi_net_h, x0, time_horizont, holding_time, objective_function, dynamic_for_one_step )
        control_time=int((time_horizont-1)/holding_time)+1
        for i in range ( control_time ):
            if i < control_time - np.ceil(grace_time/holding_time):
                self.floor_constraints[ i + 17 * time_horizont ] = min_control_value
                self.ceilloing_constraints[ i + 17 * time_horizont ] = max_control_value
            else:
                
                self.floor_constraints[ i + 17 * time_horizont ] = 0
                self.ceilloing_constraints[ i + 17 * time_horizont ] = 0
                

    
# nerual_models=get_net_models()
# x_first=x0
# time_shrinking=10
# index=0
# control_time=int((total_time_horizont-1)/holding_time)+1
# x_init=np.zeros((17*total_time_horizont+control_time,1)) 
# MyProblem=Problem_With_Grace_time(nerual_models['f'],nerual_models['h'],x_first,total_time_horizont,grace_time,holding_time,cs.sumsqr,system_step)
# MySolution=MyProblem.get_soultion('ipopt',x_init)
# [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,total_time_horizont)
# u_opt_numpy=np.squeeze(np.array(u_opt))
# y_opt_numpy=np.squeeze(np.array(y_opt))

# visualize_sol(y_opt_numpy,u_opt_numpy)     
