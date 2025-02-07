# How to Use the Code 

To use the code, the `test_site.py` file needs to be executed, instantiating the appropriate classes. To control the system, it is necessary to instantiate a model class and a plant class.  The former serves as the foundation for model predictive control, based on which the algorithm computes the intervention input sequence, while the latter represents the actual system response.  The control-related parameters can be found in the `parameters.py` file, such as the control horizon and the intervention intervals.  

The constructor of the `Model` class is defined as follows:  

```python
def __init__(self, dimension, dynamic_for_one_step, output_mapping):
```
- **dimension:** The number of dimensions of the state vector.
- **dynamic_for_one_step:** The stepping function responsible for the model dynamics.
- **output_mapping:** The function that generates the output from the current state vector.

The constructor of the `Plant` class is the same as the `Model` class constructor.

The constructor of the `PanSim` class is defined as follows:  

```python
def __init__(self,simulator,encoder):
```
- **simulator:** The instance of the PanSim simulator.
- **encoder:** The SUBNET's encoder network.

After the appropriate parameters and classes have been instantiated, a strategy can be applied using the `shrinking_MPC` and `rolling_MPC` functions.  

```python
[Y_real,Y_model,U,X,time]=shrinking_MPC(noise_MPC, noise_plant, time_horizon, x, grace_time, model, plant, discr)
[Y_real,Y_model,U,X,time]=rolling_MPC(noise_MPC, noise_plant, time_horizon, rolling_horizon, x, grace_time, model, plant, discr)
```
- **noise_MPC:** Determines whether noise is added to the model.
- **noise_plant:** Determines whether noise is added to the plant.
- **time_horizont:** The time horizon of the control. The control ends when the system reaches this point.
- **x:** The initial state vector.
- **rolling_horizont:** The time interval over which `rolling_MPC` calculates the model's response.
- **grace_time:** The time interval during which a zero control input is applied from the end of the time horizont.
-  **model:** The instantiated of the `Model` class.
- **plant:** The instantiated of the `Plant` class.
- **discr:** Determines whether the control input is rounded.
  The return values:
- **Y_real:** The plant's response during the control.
- **Y_model:** The model's response during the control.
- **U**: The applied control input
- **X**: The model's state vectors during the control.
- **time**: The reqired time to solve the optimization problem.

Note: The instance of the `Pansim` class can also be used to describe the plant.

## Tests and examples

This subsection presents the class instantiations and function calls used in the implementation of each test case in the Thesis.  


### Test Scenario 1 in Section 3.2.2
- **Model:** Compartmental description.  
- **Plant:** Compartmental model without noise.  
- **The intervention signal is not rounded.**  

#### Code Implementation  
```python
comparmental_plant = Plant(8, runge_kutta_4_step, compartmental_model_mapping)
comparmental_model = Model(8, runge_kutta_4_step, compartmental_model_mapping)

[Y_real, Y_model, U, X, time] = shrinking_MPC(
    0, 0, total_time_horizont_extended, x0_comparmental, 
    grace_time_extended, comparmental_model, comparmental_plant, 0
)
```
### Test Scenario 2 in Section 3.2.2
- **Model:** Compartmental description.  
- **Plant:** Compartmental model without noise.  
- **The intervention signal is rounded.**  

#### Code Implementation  
```python
comparmental_plant = Plant(8, runge_kutta_4_step, compartmental_model_mapping)
comparmental_model = Model(8, runge_kutta_4_step, compartmental_model_mapping)

[Y_real, Y_model, U, X, time] = shrinking_MPC(
    0, 0, total_time_horizont_extended, x0_comparmental, 
    grace_time_extended, comparmental_model, comparmental_plant, 1
)
```
### Test in Section 3.3.3
- **Model:** Compartmental description.  
- **Plant:** Compartmental model with noise.  
- **The intervention signal is rounded.**  

#### Code Implementation  
```python
comparmental_plant = Plant(8, runge_kutta_4_step, compartmental_model_mapping)
comparmental_model = Model(8, runge_kutta_4_step, compartmental_model_mapping)

[Y_real, Y_model, U, X, time] = shrinking_MPC(
    0, 1, total_time_horizont_extended, x0_comparmental, 
    grace_time_extended, comparmental_model, comparmental_plant, 1
)
[Y_real, Y_model, U, X, time] = rolling_MPC(
    0, 1, total_time_horizont_extended,119, x0_comparmental, 
    grace_time_extended, comparmental_model, comparmental_plant, 1
)
```
### Test in Section 6.4
- **Model:** SUBNET model.  
- **Plant:** PanSim simulator.
- **The intervention signal is rounded.**  

#### Code Implementation  
```python
pansim=PanSim(simulator,encoder)
subnet=Modell(16,system_step_neural,output_mapping_neural)

[Y_real, Y_model, U, X, time] = shrinking_MPC(
    0, 1, total_time_horizont_extended, x0, 
    grace_time_extended, subnet, pansim, 1
)
[Y_real, Y_model, U, X, time] = rolling_MPC(
    0, 1, total_time_horizont_extended,119, x0, 
    grace_time_extended, pansim, pansim, 1
)
```



