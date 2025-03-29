import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd

"""
To initialise all known parameters of the single MTU steady state optimisation, I am using a csv file later
Here all values are arbitrarily set to test the optimisation 
"""

def load_values(df, start = 13, horizon = 8):
    results = pd.DataFrame()
    # system parameters
    system_data = {
        'pricelimit': 15000, #Eur/Mwh
        'Emin': 9, #MWh
        'Emax': 51, #MWh
        'Einit': 40  #MWh
    }
    price_data = df.iloc[start:start + horizon].reset_index(drop = True)

    md = steady_state_multiple_MTU_OP(price_data, system_data, horizon)
    if md.status == GRB.OPTIMAL:
        print("Optimization was successful.")
        
        total_revenue = md.ObjVal  # Objective function value
        total_bid_UP = [md.getVarByName(f"E_up[{t}]").x for t in range(horizon)]
        total_bid_Down = [md.getVarByName(f"E_down[{t}]").x for t in range(horizon)]
        total_price_UP = [md.getVarByName(f"P_up[{t}]").x for t in range(horizon)]
        total_price_Down = [-md.getVarByName(f"P_down[{t}]").x for t in range(horizon)]
        total_activation = [md.getVarByName(f'u_Act[{t}]').x for t in range(horizon)]
        total_cleared = [md.getVarByName(f'ifCleared[{t}]').x for t in range(horizon)]
        bid_cleared_UP = [md.getVarByName(f"Ecleared_up[{t}]").x for t in range(horizon)]
        bid_cleared_Down = [md.getVarByName(f"Ecleared_down[{t}]").x for t in range(horizon)]
        system_state = [md.getVarByName(f"Esys[{t}]").x for t in range(horizon-1)]
        system_state = [system_data['Einit']] + system_state
        results_dict = {
            'Objective Value': total_revenue,
            'Ebid Up (MWh)': total_bid_UP,
            'Ebid Down (MWh)': total_bid_Down,
            'Pbid Up (Eur/MWh)': total_price_UP,
            'Pbid Down (Eur/MWh)': total_price_Down,
            'Activation 1 - Up 0 - Down': total_activation,
            'Cleared 1 - Yes 0 - No': total_cleared,
            'Ebid Up Cleared (MWh)': bid_cleared_UP,
            'Ebid Down Cleared (MWh)': bid_cleared_Down,
            'System state (MWh)': system_state
        }
        
        print(f"Total Revenue: {total_revenue}")     

        # print_results_flag = False
        # if print_results_flag:
        #         print(f"Optimization Results for {horizon} MTUs:\n")
        #         # Display each variable in a table format
        #         for key, value in results_dict.items():
        #             df = pd.DataFrame(value, columns)
        #             df.index.name = "Hour"
        #             print(f"{key}:\n")
        #             print(df)
        #             print("\n" + "-" * 50 + "\n")

    elif md.status == gp.GRB.INFEASIBLE:
        print("Optimization was not successful: INFEASIBLE")
        md.computeIIS()
        md.write("model.ilp")

    elif md.status == gp.GRB.UNBOUNDED:
        print("Optimization was not successful: UNBOUNDED")

    elif md.status == gp.GRB.INF_OR_UNBD:
        print("Optimization was not successful. Model is infeasible or unbounded.")
        md.write("model.lp")

    elif md.status != gp.GRB.OPTIMAL:
        print(f"Optimization was not successful. Status: {md.status}")
    return results_dict, md

def steady_state_multiple_MTU_OP(price_data, system_data, horizon):
    model = gp.Model('Multiple_MTU_problem')
    
    T = list(range(horizon)) # Horizon = 8 MTU
    M = 1e6

    # parameters
    lambda_cl_UP = price_data['aFRR_UpActivatedPriceEUR']
    lambda_cl_DOWN = -price_data['aFRR_DownActivatedPriceEUR']
    spotprice = price_data['SpotPriceEUR']
    alpha = price_data['Total MWh activated']  # loaded as % values
    Cap_UP = price_data['UpCapacityBid']
    Cap_DOWN = price_data['DownCapacityBid']
    pricelimit = system_data['pricelimit']  # system_data is a dict of physical parameters of the system
    Emin = system_data['Emin']
    Emax = system_data['Emax']
    Einit = system_data['Einit']

    # initalise the decision variables
    E_up = model.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'E_up')
    E_down = model.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'E_down')
    P_up = model.addVars(T, lb = 0, vtype = GRB.CONTINUOUS, name = 'P_up')
    P_down = model.addVars(T, lb = 0, vtype = GRB.CONTINUOUS, name = 'P_down')
    u = model.addVars(T, vtype = GRB.BINARY, name = 'u_Act')
    beta = model.addVars(T, vtype = GRB.BINARY, name = 'ifCleared')

    # To track the state of the system
    Esys = model.addVars(T, lb = Emin, ub = Emax, vtype = GRB.INTEGER, name = 'Esys')
    
    # auxillary variable
    Ecl_up = model.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'Ecleared_up')
    Ecl_down = model.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'Ecleared_down')

    # Constraints on the system limits
    for t in T:
        # Bid MWh is equal to or higher than the DA Capacity schedule
        model.addConstr(E_up[t] >= (Cap_UP[t] * 0.25), name = f'UP_Capacity_schedule_limit{t}')
        model.addConstr(E_down[t] >= (Cap_DOWN[t] * 0.25), name = f'DOWN_Capacity_schedule_limit{t}')
    
    # Bid Qty (MWh) is  less than or equal to the available Mwh
    model.addConstr(Einit - E_up[0] >= Emin, name = f'Available_Mwh_UP_{0}')
    model.addConstr(Einit + E_down[0] <= Emax, name = f'Available_Mwh_Down_{0}')
    for t in T[1:]:
        model.addConstr(Esys[t-1] - E_up[t] >= Emin, name = f'Available_Mwh_UP_{t}')
        model.addConstr(Esys[t-1] + E_down[t] <= Emax, name = f'Available_Mwh_Down_{t}')

    # Constraint for the System state after MTU activation
    model.addConstr(Einit + (Ecl_down[0] * (1 - u[0]) - Ecl_up[0] * u[0]) * alpha[0] - Esys[0] == 0, name = f'System_state_after_{0}')
    for t in T[1:]:
        model.addConstr(Esys[t-1] + (Ecl_down[t] * (1 - u[t])  - Ecl_up[t] * u[t]) * alpha[t] - Esys[t] == 0, name = f'System_state_after_{t}')

    # Constraints on the price logic
    for t in T:
        # Bid price (Up) is spot price and Bid Price (Down) is 0 (-ve) when activated else Pricelimit/-Pricelimit
        model.addConstr(P_up[t] == spotprice[t] * u[t] + pricelimit * (1 - u[t]), name = f'UP_price_limit_{t}')
        model.addConstr(P_down[t] == pricelimit * u[t], name = f'Down_price_limit_{t}')

    # constraint to enforce beta (Consider the revenue only when a bid is cleared)
    for t in T:
        model.addConstr(P_up[t] <= lambda_cl_UP[t] + M * (1 - beta[t]) + M * (1 - u[t]), name = f'UP_ifCleared_{t}')
        model.addConstr(-P_down[t] >= lambda_cl_DOWN[t] - M * (1 - beta[t]) - M * u[t], name = f'Down_ifCleared_{t}')
 
    #constraint to enforce auxillary variables
    for t in T:
        model.addConstr(Ecl_up[t] <= E_up[t], name = f'Ecleared_UP_a_{t}')
        model.addConstr(Ecl_up[t] <= M * beta[t], name = f'Ecleared_UP_b_{t}')
        model.addConstr(Ecl_up[t] >= E_up[t] - M * (1 - beta[t]), name = f'Ecleared_UP_c_{t}')

        model.addConstr(Ecl_down[t] <= E_down[t], name = f'Ecleared_Down_a_{t}')
        model.addConstr(Ecl_down[t] <= M * beta[t], name = f'Ecleared_Down_b_{t}')
        model.addConstr(Ecl_down[t] >= E_down[t] - M * (1 - beta[t]), name = f'Ecleared_Down_c_{t}')

    # define the objective function
    Z = gp.quicksum((lambda_cl_UP[t] * Ecl_up[t] * alpha[t] * u[t]) + 
                    (-lambda_cl_DOWN[t] * Ecl_down[t] * alpha[t] * (1-u[t]))
                    for t in T)
    model.setObjective(Z, GRB.MAXIMIZE)

    model.optimize()

    return model