from support import read_array_from_txt,visualize_Y_quess_vs_Y_real
U=read_array_from_txt('input_1_control.txt')
Y_quess=read_array_from_txt('input_1_plant_quess.txt')
Y_real=read_array_from_txt('output_1_real.txt')
visualize_Y_quess_vs_Y_real(Y_quess,Y_real,U)