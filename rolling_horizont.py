from opti_problem import *
from parameters import x0,total_time_horizont,grace_time


def rolling_MPC():
    nerual_models=get_net_models()
    x_first=x0
    time_rolling=10
    U=np.empty((1,total_time_horizont+time_rolling+int(total_time_horizont/time_rolling)))
    Y=np.empty((1,total_time_horizont+time_rolling+int(total_time_horizont/time_rolling)))
    rollinng_time_horizont=0
    x_init=np.zeros((18*(time_rolling),1))
  
    while rollinng_time_horizont<total_time_horizont+time_rolling+time_rolling+int(total_time_horizont/time_rolling):
        MyProblem=Problem(nerual_models['f'],nerual_models['h'],x_first,time_rolling,cs.sumsqr,system_step)
        MySolution=MyProblem.get_soultion('ipopt',x_init)
        [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,time_rolling)
        
        if(U[:,rollinng_time_horizont:rollinng_time_horizont+u_opt.shape[1]].shape[1]==10):
            U[:,rollinng_time_horizont:rollinng_time_horizont+u_opt.shape[1]]=u_opt
            Y[:,rollinng_time_horizont:rollinng_time_horizont+y_opt.shape[1]]=y_opt
        else:
            U[:,rollinng_time_horizont:-1]=u_opt[:,0:U[:,rollinng_time_horizont:-1].shape[1]]
            Y[:,rollinng_time_horizont:-1]=y_opt[:,0:U[:,rollinng_time_horizont:-1].shape[1]]
        rollinng_time_horizont=rollinng_time_horizont+time_rolling
        x_first=x_opt[:,-1]

        x_init=from_x_u_y_to_solution(x_opt,u_opt,y_opt,time_rolling)

    Y=np.squeeze(Y)
    U=np.squeeze(U)
    Y = np.delete(Y, np.arange(time_rolling-1, len(Y), time_rolling))
    U = np.delete(U, np.arange(time_rolling-1, len(U), time_rolling))
    Y=Y[0:total_time_horizont]
    U=U[0:total_time_horizont]
    return [Y,U]


