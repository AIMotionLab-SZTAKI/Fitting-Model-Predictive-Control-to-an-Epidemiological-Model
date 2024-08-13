# Fitting-Model-Predictive-Control-to-an-Epidemiological-Model
- **`mpc_in1d/2d.py`**
  - Egyszerű lineáris rendszerekre illeszt prediktív kontrollt
  - Kirajzolja az eredményt

- **`optimization.py`**
  - Konkrét járványmodellre illeszt kontrollt
  - Az optimalizációs eljárást valósítja meg
  - Kirajzolja az eredményt
  - Meghívja a mapping.py és a optimalzation_model.py fájlokat
    
- **`optimization_model.py`**
  - A járvány model viselkedését leíró kód
  - Paramétereket tartalmazz, mint kiinduló állapot, mind időhorizont
  - A diszkretizálást is megvalósítja


- **`mapping.py`**
  - Felel a terminális halmaz megtalálásáért
  - Könnyebb futtatás érdekében a kapcsolódó információk kommentelve vannak az `optimization.py` fájlban
  - Meghívja az one_set_simlation.py fájlt

- **`one_set_simulation.py`**
  - Szimulál egy adott kezdő állapotból kiinduló rendszert
  - Segít a `mapping.py` fájlnak eldönteni, hogy egy halmaz terminális-e vagy sem
