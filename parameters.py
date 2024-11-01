import casadi as cs
input_sets = [["TPdef", "PLNONE", "CFNONE", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PL0", "CFNONE", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CF2000-0500", "SONONE", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SO12", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SO3", "QU0", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU2", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU3", "MA1.0"],
              ["TPdef", "PLNONE", "CFNONE", "SONONE", "QU0", "MA0.8"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU2", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU3", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO12", "QU2", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO3", "QU2", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO12", "QU3", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SO3", "QU3", "MA1.0"],
              ["TP015", "PLNONE", "CFNONE", "SONONE", "QU2", "MA0.8"],
              ["TP035", "PLNONE", "CFNONE", "SONONE", "QU3", "MA0.8"],
              ["TP035", "PL0", "CFNONE", "SO3", "QU3", "MA0.8"],
              ["TP035", "PLNONE", "CF2000-0500", "SO3", "QU3", "MA0.8"]]
init_options = ['panSim', '-r', ' ', '--diags', '0', '--quarantinePolicy', '0', '-k', '0.00041',
                '--progression', 'inputConfigFiles/progressions_Jun17_tune/transition_config.json',
                '-A', 'inputConfigFiles/agentTypes_3.json',
                '-a', 'inputRealExample/agents1.json',
                '-l', 'inputRealExample/locations0.json',
                '--infectiousnessMultiplier', '0.98,1.81,2.11,2.58,4.32,6.8,6.8',
                '--diseaseProgressionScaling', '0.94,1.03,0.813,0.72,0.57,0.463,0.45',
                '--closures', 'inputConfigFiles/emptybbRules.json'
                ]
debug = 0
total_time_horizont = 180
grace_time = 40
holding_time = 7
if total_time_horizont%holding_time==0:
    total_time_horizont_extended=total_time_horizont
else:
    total_time_horizont_extended=total_time_horizont-(total_time_horizont%holding_time)
hospital_capacity = 200
min_control_value = 0
max_control_value = 17 
rolling_horizont=(total_time_horizont_extended-grace_time)-((total_time_horizont_extended-grace_time)%holding_time)
