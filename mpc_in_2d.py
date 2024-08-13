import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt

# Ezek azok a bevatkozási értékek amelyekkel beavatkozhatunk (U halmaz).
u_values={0:-10.,
          1:-8.,
          2:-6.,
          3:-4.,
          4:-2.,
          5:0.,
          6:2.,
          7:4.,
          8:6.,
          9:8.,
          10:10}
# Azok a vektorok melyek meghatározzák, hogy az egyes beavatkozások mennyi ideig lehetnek aktívak (W_{Tmin},W_{Tmax})
tmin=np.ones(len(u_values))*2
tmax=np.ones(len(u_values))*4

# Függvény amely az optimalizációs problémát hozza létre:
# Bemenet: A rendszert leíró dinamika, kiinduló állapot, illetve az inicializációs vektor
# Kimenet: Az optimaliálandó probléma (obcejt és a határtok)
def model_create(Aparam,Bparam,xreqparam,x0param):
    model = pyo.ConcreteModel() 
    model.horizont=range(30)
    model.A=Aparam
    model.B=Bparam
    model.xreq=xreqparam
    
    model.u_values = pyo.Set(initialize=u_values.values())
    
    model.x=pyo.Var(model.horizont)
    model.u = pyo.Var(model.horizont,domain=model.u_values)
    model.u_idx=pyo.Var(model.horizont,range(len(model.u_values)),domain=pyo.Binary)
    model.H=pyo.Var(range(len(model.horizont)+1),range(len(model.u_values)),domain=pyo.NonNegativeIntegers)
    
    model.constraints = pyo.ConstraintList()
    model.u_idx_constraints = pyo.ConstraintList()
    model.u_rule_constraints=pyo.ConstraintList()
    model.wtc_constraints_downer=pyo.ConstraintList()
    model.wtc_constraints_upper=pyo.ConstraintList()
    model.counting_constraing=pyo.ConstraintList()
    

    model.obj = pyo.Objective(rule=lambda model: obj_rule(model), sense=pyo.minimize)
    
    model.x[0].fix(x0param)

    h_counting(model)
    u_idx_const(model)
    u_rule(model)
    system_dynamic(model)
    wtc_rule_downer(model)
    wtc_rule_upper(model)
    return model
# Függvény amely a számoláló léptetésért felelős (9-es egyenlet)
def h_counting(model):
    for t in model.horizont:
        for j in  range(len(model.u_values)):
                model.counting_constraing.add(model.H[t+1,j]==(model.H[t,j]+model.u_idx[t,j])*model.u_idx[t,j])

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
# Az optimalizlandó kifejezés (3-as egynelet)
def obj_rule(model):
    return sum((model.x[t] - model.xreq)**2 for t in model.horizont)
# A rendszer dinamikát megvalósító hatrátok (1-es egynelet)
def system_dynamic(model):
    for t in model.horizont:
        if t < max(model.horizont):
            model.constraints.add(model.x[t+1] == model.A * model.x[t] + model.B * model.u[t])
           
# A rendszer dinamikát megahatározó paraméterek
A = 1.1
B = 1.5

# Az állapot amelybe a rendszert irányítani akarjuk
xreq=np.array([100.])
# Kiniduló állapot 
x0=np.array([0.])

# Az  idő horizont hossza
total_time=30

# Vektorok melyek az adat kirajzoltatását segítik elő
total_time_u=np.linspace(0,total_time,total_time)
total_time_x=np.linspace(0,total_time+1,total_time)
# Adatárolók melyekbe az eredményt kapjuk
xsystem = np.ones(30)
usystem=np.ones(30)


# A solver típusának beállítása
solver = pyo.SolverFactory('gurobi')



# A konrét modell "példányosítása"
model=model_create(A,B,xreq,x0)

# A probléma megoldása 
solution=solver.solve(model)
# A kapott adatok eltárolása
for i in range(total_time):
    usystem[i]=model.u[i].value
    xsystem[i] = model.x[i].value



# És végül az adtok vizualizációja 
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.title("Control signal")
plt.xlabel("Time [s]")
plt.ylabel("U")
plt.plot(total_time_u,np.squeeze(usystem),color="b",linestyle="",marker="o")
plt.grid()
y_ticks = np.arange(np.min(usystem), np.max(usystem) + 1, 1)
plt.yticks(y_ticks)


plt.subplot(1,2,2)
plt.title("State of the system")
plt.xlabel("Time [s]")
plt.ylabel("X")
plt.plot(total_time_x,np.squeeze(xsystem),color="r",linestyle="-",marker="o")
plt.grid()
y_ticks = np.arange(np.min(xsystem), np.max(xsystem) + 1, 10)
plt.yticks(y_ticks)
plt.show()
