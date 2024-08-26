import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import optimalization_modell_with_wtc as m
import logging
from pyomo.util.infeasible import log_infeasible_constraints
dt=m.dt
#terminal_state=mp.get_terminal_state()
terminal_state=[m.real_population*0.8]
# Paraméterek a szabályozásnak:
# Az idő horizont melyen irányítani szeretnénk 
t_end=m.t_end
# Kezdőállapot felvétele
x0 = m.x0
# inicializációs az optimalizációnak
x_init=np.ones((8,t_end),dtype=float)
# Kvantálása a beavatkozó jelnek
k=10
u_values=np.ones(k)
for  i in range (k-1):
    u_values[i]=i/10
tmin=np.ones(len(u_values))*2
tmax=np.ones(len(u_values))*4
# Ez a függvény felelős az optimalizációs probléma felépítésért
# Bemenet: Kezdőállapot a dinamikának
# Kimenet: Az optimalizálandó modell
def create_model (x0param):
    model=pyo.ConcreteModel()
    model.horizont=range(t_end)
    model.dim=range(8)
    model.u_values = pyo.Set(initialize=u_values)
        
    model.x=pyo.Var(model.horizont,model.dim)
    model.u = pyo.Var(model.horizont)
    model.u_idx=pyo.Var(model.horizont,range(len(model.u_values)),domain=pyo.Binary)
    model.H=pyo.Var(range(len(model.horizont)+1),range(len(model.u_values)))
    
    model.constraints = pyo.ConstraintList()
    model.hospital_capacity=pyo.ConstraintList()
    model.u_idx_constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()
    model.wtc_constraints_downer=pyo.ConstraintList()
    model.wtc_constraints_upper=pyo.ConstraintList()
    model.counting_constraints=pyo.ConstraintList()


    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        

                       
    for j in model.dim:
            model.x[0,j].fix(x0param[j])    
    model.u[0].fix(0)
    for i in range(len(model.u_values)):
        if i==0:
            model.H[0,i].fix(1)
        else:
            model.H[0,i].fix(0)
 
    h_counting(model)
    system_dynamic(model)
    u_idx_const(model)
    u_rule(model)
    wtc_rule_downer(model)
    wtc_rule_upper(model)
    return model

# A számolásért felelős függvény (9-es egyenlet)
def h_counting(model):
    for t in model.horizont:
        for j in  range(len(model.u_values)):
                model.counting_constraints.add(model.H[t+1,j]==(model.H[t,j]+model.u_idx[t,j])*model.u_idx[t,j])

# Függvény amely a wtc felső betartását hozza létre (11-es egyenlet)
def wtc_rule_upper(model):
    for t in model.horizont:
        for j in range(len(u_values)):
            model.wtc_constraints_upper.add(model.H[t,j]<=tmax[j])

# Függvény amely a wtc alső betartását hozza létre (10-es egyenlet)
def wtc_rule_downer(model):
  
    for t in model.horizont:
        
        if(t<max(model.horizont)):
            res1 = 0
            res2 = 0
            for j in range(len(u_values)):
                res1 = res1 + (model.u_idx[t, j] * (tmin[j] - model.H[t+1, j]))
                res2 = res2 + (model.u_idx[t, j] * model.u_idx[t+1, j]) 
        res2 = res2 * 1200000
        model.wtc_constraints_downer.add( res1 <= res2)                            
# Függvény, amely biztosítja, hogy a bevatkozás csak a megadott U halmazból kerüljön ki (7-es egyenlet)
def u_rule(model):
    for t in model.horizont:
        res=0
        for i in range(len(model.u_values)):
            res=model.u_idx[t,i]*u_values[i]+res
        model.u_rule_constraints.add(model.u[t]==res)
    

# Függvény amely biztosístja hogy egy adott időpontban csak egy index legyen aktív (8-as egyenlet)
def u_idx_const(model):
    for t in model.horizont:
        model.u_idx_constraints.add(sum(model.u_idx[t,:])==1) 
# A beavatkozásokat  minimalizálni akarjuk (20-as egyenlet) 
def obj_rule(model):
    
    return sum(((model.u[t]**2)) for t in model.horizont)

# A dinamika betartásáért felelős függvény (21-es egyenlet)
def system_dynamic(model):
    x_temp=[None]*len(model.dim)
    for t in model.horizont:
        # Korlátozzuk a kórházak kapacitásást is (22-es egyenlet)
        model.hospital_capacity.add(model.x[t,5]<=m.real_max_patients)
        for i in model.dim:
            x_temp[i]=model.x[t,i]
        # A diferenciálegyenlet rendszer numerikus megoldásának léptetésért felelős 
        res=m.runge_kutta_4_step(x_temp,model.u[t])
        for j in model.dim:

            if t < max(model.horizont):
                
                model.constraints.add(model.x[t+1,j]==res[j])
        
    
    
    
# Létrehozzuk a problémát
M=create_model(x0)
# Kiválasztjuk a solvert
solution = pyo.SolverFactory('baron')
# És megoldjuk a solver segítségével 
solution=solution.solve(M, tee=True,options={'MaxTime': -1})


# Tömbök az adatok eltárolásához
s_values=[np.float64]*len(M.horizont)
l_values=[np.float64]*len(M.horizont)
p_values=[np.float64]*len(M.horizont)
i_values=[np.float64]*len(M.horizont)
a_values=[np.float64]*len(M.horizont)
h_values=[np.float64]*len(M.horizont)
r_values=[np.float64]*len(M.horizont)
d_values=[np.float64]*len(M.horizont)
u_values=[np.float64]*len(M.horizont)
# Az adatok eltárolása, illetve a numerikus adatok korrekciója a normálás aés korrigálása érdekében
for i in M.horizont:
    s_values[i]=M.x[i,0].value
    l_values[i]=M.x[i,1].value
    p_values[i]=M.x[i,2].value
    i_values[i]=M.x[i,3].value
    a_values[i]=M.x[i,4].value
    h_values[i]=M.x[i,5].value
    r_values[i]=M.x[i,6].value
    d_values[i]=M.x[i,7].value

    u_values[i]=M.u[i].value
    
# Az adatok vizualizáció érdekében létrehozott tömb
t_values=np.linspace(0,t_end-1,len(M.horizont))
# A kórházban lévő szmosságot a folytonos modellen is szimuláljuk, így ellenörizzük, hogy valóban jók a számításaink
hospital=m.real_model_simulation(u_values)

# Végül az adtok vizualizációja
plt.figure(figsize=(12, 12))
plt.subplot(2,2,3)
plt.plot(t_values, hospital,color="k",linestyle="-",marker=".")
plt.plot(t_values,h_values,color="m",linestyle="",marker=".")
plt.legend(['The real system respond ','The predicted respond' ])
plt.xlabel("Time [days]")
plt.ylabel("Cardinality of the set [sample]")
plt.grid()
plt.subplot(2,2,2)
plt.plot(t_values,l_values,color="b",linestyle="-",marker=".")
plt.plot(t_values,p_values,color="g",linestyle="-",marker=".")
plt.plot(t_values,i_values,color="r",linestyle="-",marker=".")
plt.plot(t_values,a_values,color="c",linestyle="-",marker=".")
plt.plot(t_values,h_values,color="m",linestyle="-",marker=".")
plt.plot(t_values,d_values,color="k",linestyle="-",marker=".")
plt.legend(['Latent','Pre-symptomatic ','Symptomatic infected','Symptomatic infected but will recover','Hospital','Died'])
plt.xlabel("Time [days]")
plt.ylabel("Cardinality of the set [sample]")
plt.grid()
plt.subplot(2,2,4)
plt.plot(t_values,s_values,color="b",linestyle="-",marker=".")
plt.plot(t_values,r_values,color="g",linestyle="-",marker=".")
plt.legend(['Susceptibles','Recover'])
plt.xlabel("Time [days]")
plt.ylabel("Cardinality of the set [sample]")
plt.grid()

plt.subplot(2,2,1)
plt.plot(t_values,u_values,color="b",linestyle="",marker="o")
plt.legend(['Control signal'])
plt.xlabel("Time [days]")
plt.ylabel("Control scenarios")
plt.grid()

plt.show()
