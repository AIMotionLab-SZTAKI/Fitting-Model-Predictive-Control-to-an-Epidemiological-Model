import numpy as np
time_horizont=205
real_population=9800000
real_latent=80
hospital_capacity=20000
min_control_value=0
max_control_value=17
grace_time=30
dt=1
holding_time=7
x0 = np.array([(real_population-real_latent)/real_population, real_latent/real_population,0.,0.,0.,0.,0.,0.])