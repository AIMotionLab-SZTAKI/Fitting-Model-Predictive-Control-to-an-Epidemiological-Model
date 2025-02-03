import casadi as cs
import numpy as np
from support import *
from parameters import min_control_value, max_control_value, hospital_capacity
import time


class Problem:
    def __init__( self, x0, time_horizont, holding_time, Model ,margin):
        self.control_time=int(np.ceil(time_horizont/holding_time))
        self.dim=Model.dim
        self.x = cs.MX.sym( 'x', self.dim,time_horizont )
        self.u = cs.MX.sym( 'u', 1,self.control_time )
        self.y = cs.MX.sym( 'y', 1,time_horizont )
        
        self.system_step = Model.dynamic
        self.mapping=Model.map
        constraints_for_step_state = [ ]
        constraints_for_output = [ ]
        constraints_for_step_state.append( self.x[:, 0] - x0 )
        penalty_weight = 1e8  
        self.time=0
        
        penalty = penalty_weight * cs.sumsqr(cs.fmax(self.y - hospital_capacity, 0))

        
        self.objective = cs.sumsqr(self.u) +penalty
        for i in range( time_horizont ):
            if i == time_horizont - 1:
                x_next = self.system_step( self.x[ :, i ].T, self.u[ :, int(i/holding_time) ].T ) 
                y_out=self.mapping(self.x[ :, i ].T)
                constraints_for_output.append( self.y[ :, i ] - y_out)
            else:
                x_next  = self.system_step( self.x[ :, i ].T, self.u[ :, int(i/holding_time) ].T ) 
                y_out=self.mapping(self.x[ :, i ].T)
                constraints_for_step_state.append( self.x[ : , i + 1 ] - x_next.T )
                constraints_for_output.append( self.y[ :, i ] - y_out)
            
            
        step_state = cs.vertcat( *constraints_for_step_state )
        step_output = cs.vertcat( *constraints_for_output )
        g = cs.vertcat( step_state, step_output )

        for i in range( self.control_time ):
            g = cs.vertcat( g, self.u[ :, i ])
        for i in range(time_horizont): 
            g = cs.vertcat( g, self.y[ :, i ])  
                
        nlp = { 'x': cs.vertcat( cs.vec( self.x ), cs.vec( self.u ), cs.vec( self.y )), 
                'f': self.objective,  
                'g': g}  
        
        lbg = np.zeros( (self.dim+1)*time_horizont + self.control_time + time_horizont )
        ubg=np.zeros( (self.dim+1) * time_horizont + self.control_time + time_horizont )
        for i in range ( (self.dim+1) * time_horizont + self.control_time + time_horizont ):
            if i < (self.dim+1) * time_horizont:
                lbg[ i ] = 0
                ubg[ i ] = 0
            else:
                if i< (self.dim+1) * time_horizont + self.control_time:
                    lbg[ i ] = min_control_value
                    ubg[ i ] = max_control_value
                
                else:
                    lbg[ i ] = -1000/real_population
                    ubg[ i ] = hospital_capacity+margin

        self.nlp = nlp
        self.floor_constraints = lbg
        self.ceilloing_constraints = ubg
        
    def add_noise (self,time_horizont):
        for i in range ((self.dim+1)*time_horizont):
            noise = np.random.rand() * 0.025
            self.nlp['g'][i]=self.nlp['g'][i]-noise

    def get_soultion( self, solver_type, x_init ):
        solver = cs.nlpsol( 'solver', solver_type, self.nlp )
        start=time.time()
        solution = solver( lbg = self.floor_constraints, ubg = self.ceilloing_constraints,x0 = x_init ) 
        end=time.time()
        self.time=end-start
        print(solver.stats()['return_status'])
        return solution
class Problem_With_Grace_time(Problem):
      def __init__( self,  x0, time_horizont, grace_time, holding_time, model,margin ):
        super().__init__(  x0, time_horizont, holding_time, model,margin )

        for i in range ( self.control_time ):
            if i < self.control_time - np.ceil(grace_time/holding_time):
                self.floor_constraints[ i + (self.dim+1) * time_horizont ] = min_control_value
                self.ceilloing_constraints[ i + (self.dim+1) * time_horizont ] = max_control_value
            else:
                
                self.floor_constraints[ i + (self.dim+1) * time_horizont ] = 0
                self.ceilloing_constraints[ i + (self.dim+1) * time_horizont ] = 0
                

    
  
