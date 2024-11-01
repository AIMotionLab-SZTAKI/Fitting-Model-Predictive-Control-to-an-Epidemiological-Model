import pyPanSim as sp
import random
import os
from datetime import datetime
import csv
import torch
from support import *
import numpy as np
from parameters import *
from simulate import simualte_with_PanSim
simulator = sp.SimulatorInterface()
simulator.initSimulation(init_options)
init_U=np.zeros(30)
hos=simualte_with_PanSim(simulator,init_U)
print(hos)

# for i in range(ENDTIME):
#     print(i)
#     input_idx,run_options=inputs[i],input_sets[int(inputs[i])]
#     results = simulator.runForDay(run_options)
#     results_agg.append(results)
#     inputs_agg.append(input_idx)
#     run_options_agg.append(run_options)


# nHospitalized = get_results(results_agg)
# print(inputs)
# print(nHospitalized)
# write_array_to_txt('output_1_real.txt',nHospitalized)
