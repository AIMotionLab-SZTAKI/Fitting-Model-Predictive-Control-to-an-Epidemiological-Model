from support import *
Y_model=read_array_from_txt('Results with pansim/Shr/#1/pred.txt')
Y_real=read_array_from_txt('Results with pansim/Shr/#1/real.txt')
U_system=read_array_from_txt('Results with pansim/Shr/#1/control.txt')
visualize_Y_quess_vs_Y_real(Y_model,Y_real,U_system)