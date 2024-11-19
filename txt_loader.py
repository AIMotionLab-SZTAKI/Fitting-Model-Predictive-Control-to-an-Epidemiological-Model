from support import *
Y_model=read_array_from_txt('pred.txt')
Y_real=read_array_from_txt('real.txt')
U_system=read_array_from_txt('control.txt')
visualize_Y_quess_vs_Y_real(Y_model,Y_real,U_system)