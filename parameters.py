import casadi as cs
total_time_horizont=180+100
grace_time=100
hospital_capacity = 200
min_control_value = 0
max_control_value = 16 
x0=cs.MX([-0.0089,  4.3177,  3.2526, -0.6230, -0.4863, -2.9737,  1.5976, -0.6301,
            0.9218,  3.0298, -2.0962,  1.4180, -3.7520,  3.4533, -1.0764,  0.0506])