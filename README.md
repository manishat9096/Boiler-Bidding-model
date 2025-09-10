# Boiler-Bidding-model
This is the repository for master's thesis on System identification and Optimal bidding model for Electric Boiler plant for PICASSO aFRR market. 

The work implements a real-time optimisation (RTO) framework in Python using Gurobi, with both open-loop and recursive closed-loop formulations. Input data consists of aFRR market prices, day-ahead spot prices, and boiler schedules. The final completed **base model** is in the directory `Base RTO Model with Actual Prices` (formulated with actual market prices), and the **estimated prices model** is in the directory `Moving Average RTO Model`. The `main.py` script supports different test cases to run the model simulations, parametric sweep to assess sensitivity to weighting factors, and varying prediction horizons. Simulation results are exported as CSV files. 

## How to Run  
Run the main script:  
```bash
python main.py
```

main.py is the entry point and depending on the case selected it will call:  

- `recurssive_RTO.py` – for recursive closed-loop optimisation  
- `OL_nonrecurssive_RTO.py` – for open-loop optimisation  
- `rt_op_modified.py` – defines the Gurobi optimisation model and handles data loading  

Results are written to the `Simulation Output/` folder and can be visualised with Plotly.

 



