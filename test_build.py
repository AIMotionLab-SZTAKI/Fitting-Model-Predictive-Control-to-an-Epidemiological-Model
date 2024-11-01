import pyPanSim as sp
import random
import os
from datetime import datetime
import csv
import torch
from closed_loop_simulation import close_loop_shr_PanSim
from rolling_horizont import rolling_MPC
from shrinking_horizont import shrinking_MPC
from support import write_array_to_txt
[Y_model,Y_real,U_system]=close_loop_shr_PanSim(shrinking_MPC,0)
write_array_to_txt('pred.txt',Y_model)
write_array_to_txt('real.txt',Y_real)
write_array_to_txt('control.txt',U_system)