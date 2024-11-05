import casadi as cs
import numpy as np
from support import *
from parameters_for_compartment import min_control_value, max_control_value, hospital_capacity,real_population 
from comparment_model import dydt_casadi
class Problem:
    def __init__( self, x0, time_horizont, holding_time, objective_function, dynamic_for_one_step ):
        self.x = cs.MX.sym( 'x', 8,time_horizont )
        self.u = cs.MX.sym( 'u', 1,time_horizont )
        self.y= cs.MX.sym( 'y', 1,time_horizont )
        self.objective = objective_function( self.u )
        self.system_step = dynamic_for_one_step
        constraints_for_step_state = [ ]
        constraints_for_step_state.append( self.x[:, 0] - x0 )
        control_time=int((time_horizont-1)/holding_time)+1
    
      
        for i in range( time_horizont ):
            if i == time_horizont - 1:
                x_next = self.system_step( self.x[ :, i ], self.u[ :, int(i/holding_time) ] ,dydt_casadi) 
            else:
                x_next = self.system_step(  self.x[ :, i ], self.u[ :, int(i/holding_time) ],dydt_casadi) 
                constraints_for_step_state.append( self.x[ : ,i+1 ] - x_next )
            
            
        step_state = cs.vertcat( *constraints_for_step_state )
        g = step_state

        for i in range( control_time ):
            g = cs.vertcat( g, self.u[ :,i ])
        for i in range(time_horizont): 
            g = cs.vertcat( g, self.y[ :,i ]-self.x[5,i]) 
        for i in range(time_horizont): 
            g = cs.vertcat( g, self.y[ :,i ])
                
        nlp = { 'x': cs.vertcat( cs.vec( self.x ), cs.vec( self.u ), cs.vec( self.y )), 
                'f': self.objective,  
                'g': g}  
        
        lbg = np.zeros( 10*time_horizont+control_time  )
        ubg=np.zeros( 10*time_horizont+control_time  )
        for i in range ( 10 * time_horizont+control_time ):
            if i < 8 * time_horizont:
                lbg[ i ] = 0
                ubg[ i ] = 0
            else:
                if i< 8 * time_horizont+control_time:
                    lbg[ i ] = 0
                    ubg[ i ] = 1
                
                else:
                    if i<9*time_horizont+control_time:
                        lbg[ i ] = 0
                        ubg[ i ] = 0
                    else:
                        lbg[ i ] = 0
                        ubg[ i ] = hospital_capacity/real_population

        self.nlp = nlp
        self.floor_constraints = lbg
        self.ceilloing_constraints = ubg
        

    def get_soultion( self, solver_type, x_init ):
        solver = cs.nlpsol( 'solver', solver_type, self.nlp )
        solution = solver( lbg = self.floor_constraints, ubg = self.ceilloing_constraints,x0 = x_init )  
        return solution 
class Problem_With_Grace_time(Problem):
      def __init__( self,  x0, time_horizont, grace_time, holding_time, objective_function, dynamic_for_one_step ):
        super().__init__(  x0, time_horizont, holding_time, objective_function, dynamic_for_one_step )
        control_time=int((time_horizont-1)/holding_time)+1
        for i in range ( control_time ):
            if i < control_time - np.ceil(grace_time/holding_time):
                self.floor_constraints[ i + 8 * time_horizont ] = min_control_value
                self.ceilloing_constraints[ i + 8 * time_horizont ] = max_control_value
            else:
                
                self.floor_constraints[ i + 8 * time_horizont ] = 0
                self.ceilloing_constraints[ i + 8 * time_horizont ] = 0
                

    
  
