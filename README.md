# How to Use the Code 

To use the code, the `test_site.py` file needs to be executed, instantiating the appropriate classes. To control the system, it is necessary to instantiate a model class and a plant class.  The former serves as the foundation for model predictive control, based on which the algorithm computes the intervention input sequence, while the latter represents the actual system response.  The control-related parameters can be found in the `parameters.py` file, such as the control horizon and the intervention intervals.  

After the appropriate parameters and classes have been instantiated, a strategy can be applied using the `shrinking_MPC` and `rolling_MPC` functions.  

## Tests and examples

This subsection presents the class instantiations and function calls used in the implementation of each test case.  


### Test Scenario 1 in Section 3.2.2
- **Model:** Compartmental description  
- **Plant:** Compartmental model without noise  
- **The intervention signal is not rounded**  

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
- **Model:** Compartmental description  
- **Plant:** Compartmental model without noise  
- **The intervention signal is rounded**  

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
- **Model:** Compartmental description  
- **Plant:** Compartmental model with noise  
- **The intervention signal is rounded**  

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
- **Model:** SUBNET description  
- **Plant:** PanSim  
- **The intervention signal is rounded**  

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

