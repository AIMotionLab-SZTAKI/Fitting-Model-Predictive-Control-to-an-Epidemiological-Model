from opti_problem import *
from parameters import x0,total_time_horizont
from matplotlib import pyplot as plt

shrinking_time_horizont=total_time_horizont
nerual_models=get_net_models()
x_first=x0
time_shrinking=10
U=np.empty((1,total_time_horizont))
Y=np.empty((1,total_time_horizont))
index=0
x_init=np.zeros((18*total_time_horizont,1))
while shrinking_time_horizont>0:
    MyProblem=Problem(nerual_models['f'],nerual_models['h'],x_first,shrinking_time_horizont,cs.sumsqr,system_step)
    MySolution=MyProblem.get_soultion('ipopt',x_init)
    [x_opt,u_opt,y_opt]=from_solution_to_x_u_y(MySolution,shrinking_time_horizont)
    U[:,index:]=u_opt
    Y[:,index:]=y_opt
    shrinking_time_horizont=shrinking_time_horizont-time_shrinking
    index=index+time_shrinking
    if shrinking_time_horizont>=time_shrinking:
        x_first=x_opt[:,time_shrinking]
    x_init=from_x_u_y_to_solution(x_opt[:,time_shrinking:],u_opt[:,time_shrinking:],y_opt[:,time_shrinking:],shrinking_time_horizont)
    print(from_x_u_y_to_solution(x_opt[:,time_shrinking:],u_opt[:,time_shrinking:],y_opt[:,time_shrinking:],shrinking_time_horizont).shape)

Y=np.squeeze(Y)
U=np.squeeze(U)
plt.subplot(2,1,1)
plt.grid()
plt.plot(Y)
plt.subplot(2,1,2)
plt.plot(U,'.')
plt.grid()
plt.show()

