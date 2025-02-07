import numpy as np

# A PanSim bemenetének értelmezése, az egyes sorok egy-egy beavatkozási stratégiát reprezentálnak.
# Egy beavatkozás különböző korlátozásokat jelent a társadalomban: kijárási tilalom, kötelező maszkviselés stb.
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

# A PanSim szimuláció inicializálásához szükséges paraméterek és konfigurációs fájlok.
init_options = ['panSim', '-r', ' ', '--diags', '0', '--quarantinePolicy', '0', '-k', '0.00041',
                '--progression', 'inputConfigFiles/progressions_Jun17_tune/transition_config.json',
                '-A', 'inputConfigFiles/agentTypes_3.json',
                '-a', 'inputRealExample/agents1.json',
                '-l', 'inputRealExample/locations0.json',
                '--infectiousnessMultiplier', '0.98,1.81,2.11,2.58,4.32,6.8,6.8',
                '--diseaseProgressionScaling', '0.94,1.03,0.813,0.72,0.57,0.463,0.45',
                '--closures', 'inputConfigFiles/emptybbRules.json']

# Az irányítási időhorizont (napokban), amely során a beavatkozásokat alkalmazzuk.
total_time_horizont = 180 + 40  # 180 nap beavatkozás + 40 nap türelmi időszak

# A türelmi időszak hossza (napokban), amely után minden beavatkozás megszűnik.
grace_time = 40

# Az időintervallum hossza (napokban), amelyen belül a beavatkozás nem változik (ZOH - Zero-Order Hold).
holding_time = 7

# Annak biztosítása, hogy az időhorizont és a türelmi idő osztható legyen a tartási idővel.
if total_time_horizont % holding_time == 0:
    total_time_horizont_extended = total_time_horizont
else:
    total_time_horizont_extended = total_time_horizont - (total_time_horizont % holding_time)

if grace_time % holding_time == 0:
    grace_time_extended = grace_time
else:
    grace_time_extended = grace_time + (holding_time - (grace_time % holding_time))

# A beavatkozások minimális és maximális értékeinek beállítása.
min_control_value = 0  # Minimális beavatkozás (pl. nincs korlátozás)
max_control_value = 1  # Maximális beavatkozás (pl. teljes lezárás)

# A populáció mérete és a modell paraméterei.
real_population = 9800000  # Valós népességméret
real_latent = 10  # Kezdeti látens esetek száma

# Az idődiszkretizáció lépésköze (napokban).
dt = 1

# A kórházi kapacitás aránya a teljes populációhoz képest (csak kompartmentális modell esetén releváns).
hospital_capacity = 20000 / real_population

# Kezdeti állapot a kompartmentális modellhez.
x0_comparmental = np.array([(real_population - real_latent) / real_population,  # Egészségesek aránya
                             real_latent / real_population,  # Látens fertőzöttek aránya
                             0., 0., 0., 0., 0., 0.])  # További állapotok (pl. fertőzöttek, kórházi esetek, gyógyultak)


margin = 50
