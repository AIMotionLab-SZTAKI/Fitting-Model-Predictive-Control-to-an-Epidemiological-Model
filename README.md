# Fitting-Model-Predictive-Control-to-an-Epidemiological-Model
A nyári szakmai gyakorlat dokumentációját és kódját tartalmazó repó. A feladat különböző járványmodellekre modell-prediktív kontrollt illeszteni.
- **`mpc_in1d/2d.py`**
  - Egyszerű lineáris rendszerekre illeszt prediktív kontrollt
  - Kirajzolja az eredményt
  - Waiting time constraint-t tartalmaz

- **`optimization.py`**
  - Konkrét járványmodellre illeszt kontrollt
  - Az optimalizációs eljárást valósítja meg
  - Kirajzolja az eredményt
  - Meghívja a `mapping.py` és a `optimalzation_model.py` fájlokat
  - Waiting time constraint-t nem tartalmaz
    
- **`optimization_model.py`**
  - A járvány model viselkedését leíró kód
  - Paramétereket tartalmazz, mint kiinduló állapot, mind időhorizont
  - A diszkretizálást is megvalósítja


- **`mapping.py`**
  - Felel a terminális halmaz megtalálásáért
  - Könnyebb futtatás érdekében a kapcsolódó információk kommentelve vannak az `optimization.py` fájlban
  - Meghívja az `one_set_simlation.py` fájlt

- **`one_set_simulation.py`**
  - Szimulál egy adott kezdő állapotból kiinduló rendszert
  - Segít a `mapping.py` fájlnak eldönteni, hogy egy halmaz terminális-e vagy sem
  
- **`optimalization_with_wtc`**
  - Ugyan az mint az `optimization.py` csak waiting time constraint-t is tartalmaz
  - Sokkal kisebb léptékeket alkalmaz
  - Nem tartalmazza a terminális halmaz elérését
    
- **`optimalization_modell_with_wtc`**
  - Ugyan az mint az `optimization_modell.py` csak waiting time constraint-t is tartalmaz
  - Nem tartalmazza a terminális halmaz elérését
    
- **`optimalization_quasi_terminal_set`**
  - Ugyan az mint az `optimization.py` csak tovább dondolt controllingal
  - Tartalmazza a kvázi terminális halmaz elérését

- **`optimalization_modell_quasi_terminal_set`**
  - Ugyan az mint az `optimization_modell.py` csak tovább gondolt controllingal
  - Tartalmazza a kvázi terminális halmaz elérését
