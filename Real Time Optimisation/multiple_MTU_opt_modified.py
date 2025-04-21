import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
"""
To initialise all known parameters of the single MTU steady state optimisation, I am using a csv file later
Here all values are arbitrarily set to test the optimisation 
"""

def load_values(df, start = 13, horizon = 8):
    # system parameters
    system_data = {
        'pricelimit': 15000, #Eur/Mwh
        'Emin': 9, #MWh
        'Emax': 51, #MWh
        'Einit': 40  #MWh
    }
    sec_in_MTU = 225
    price_data = df.iloc[start:start + (horizon*sec_in_MTU)].reset_index(drop = True)
    price_data_reshape = price_data.values.reshape(horizon, sec_in_MTU, -1)
    columns = {col: idx for idx, col in enumerate(price_data.columns)}
    afrr_up_price = price_data_reshape[:, :, columns['aFRR_UpActivatedPriceEUR']]
    afrr_down_price = price_data_reshape[:, :, columns['aFRR_DownActivatedPriceEUR']]
    activation_rate = price_data_reshape[:, :, columns['activation_rate']]

    md = steady_state_multiple_MTU_OP(price_data_reshape, columns, system_data, horizon)
    if md.status == GRB.OPTIMAL:
        print("Optimization was successful.")
        
        forecasted_revenue = md.ObjVal  # Objective function value
        total_bid_UP = [md.getVarByName(f"C_up[{t}]").x for t in range(horizon)]
        total_bid_Down = [md.getVarByName(f"C_down[{t}]").x for t in range(horizon)]
        total_price_UP = [md.getVarByName(f"P_up[{t}]").x for t in range(horizon)]
        total_price_Down = [-md.getVarByName(f"P_down[{t}]").x for t in range(horizon)]
        total_activation = [md.getVarByName(f'u_Act[{t}]').x for t in range(horizon)]
        total_cleared = [[md.getVarByName(f'ifCleared[{t},{k}]').x for k in range(225)] for t in range(horizon)]
        bid_cleared_UP = [[md.getVarByName(f"Ecleared_up[{t},{k}]").x for k in range(225)] for t in range(horizon)]
        bid_cleared_Down = [[md.getVarByName(f"Ecleared_down[{t},{k}]").x for k in range(225)] for t in range(horizon)]
        system_state = [[md.getVarByName(f"Esys[{t},{k}]").x for k in range(225)] for t in range(horizon)]
        # system_state = [system_data['Einit']] + system_state
        actual_revenue = sum((sum(afrr_up_price[t][k] * bid_cleared_UP[t][k] * activation_rate[t][k] for k in range(sec_in_MTU)) * total_activation[t] )+ (sum(afrr_down_price[t][k] * bid_cleared_Down[t][k] * activation_rate[t][k] for k in range(sec_in_MTU)) * (1 - total_activation[t])) for t in range(horizon))

        results_dict = {
            'Objective Value': forecasted_revenue,
            'Bid Up (MW)': total_bid_UP,
            'Bid Down (MW)': total_bid_Down,
            'Pbid Up (Eur/MWh)': total_price_UP,
            'Pbid Down (Eur/MWh)': total_price_Down,
            'Activation 1 - Up 0 - Down': total_activation,
            'Cleared 1 - Yes 0 - No': total_cleared,
            'Energy Up Cleared (MWh)': bid_cleared_UP,
            'Energy Down Cleared (MWh)': bid_cleared_Down,
            'System state (MWh)': system_state,
            'Perfect Obj value': actual_revenue
        }
        
        print(f"Total Revenue: {forecasted_revenue}")     

        flattened_data = {
            'Objective Value': [forecasted_revenue] * (horizon * 225),
            'Bid Up (MW)': [item for sublist in total_bid_UP for item in [sublist] * 225],
            'Bid Down (MW)': [item for sublist in total_bid_Down for item in [sublist] * 225],
            'Pbid Up (Eur/MWh)': [item for sublist in total_price_UP for item in [sublist] * 225],
            'Pbid Down (Eur/MWh)': [item for sublist in total_price_Down for item in [sublist] * 225],
            'Activation 1 - Up 0 - Down': [item for sublist in total_activation for item in [sublist] * 225],
            'Cleared 1 - Yes 0 - No': [item for sublist in total_cleared for item in sublist],
            'Energy Up Cleared (MWh)': [item for sublist in bid_cleared_UP for item in sublist],
            'Energy Down Cleared (MWh)': [item for sublist in bid_cleared_Down for item in sublist],
            'System state (MWh)': [item for sublist in system_state for item in sublist],
            'Perfect Obj value': [actual_revenue] * (horizon * 225),

        }
        results = pd.DataFrame(flattened_data)
    
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

    results =  pd.concat([results, price_data], axis=1)
    return md , results_dict, results

def steady_state_multiple_MTU_OP(price_data, columns, system_data, horizon):
    model = gp.Model('Multiple_MTU_problem')
    
    T = list(range(horizon)) # Horizon = 8 MTU
    seconds_MTU = list(range(4, 15 * 60 +1, 4)) # 4s intervals in a 15 minute step
    M = 1e6
    deltaT = 15/60
    # actual clearing prices

    # parameters
    lambda_cl_UP = price_data[:, :, columns['aFRR_UP_Price_prediction']]
    lambda_cl_DOWN = -price_data[:, :, columns['aFRR_DOWN_Price_prediction']]
    spotprice = price_data[:, 0, columns['SpotPriceEUR']]
    alpha = price_data[:, :, columns['activation_rate']]  # loaded as % values
    Cap_UP = price_data[:, 0, columns['UpCapacityBid']]
    Cap_DOWN = price_data[:, 0, columns['DownCapacityBid']]
    pricelimit = system_data['pricelimit']  # system_data is a dict of physical parameters of the system
    Emin = system_data['Emin']
    Emax = system_data['Emax']
    Einit = system_data['Einit']

    # initalise the decision variables
    C_up = model.addVars(T, lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'C_up')
    C_down = model.addVars(T, lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'C_down')
    P_up = model.addVars(T, lb = 0, vtype = GRB.CONTINUOUS, name = 'P_up')
    P_down = model.addVars(T, lb = 0, vtype = GRB.CONTINUOUS, name = 'P_down')
    u = model.addVars(T, vtype = GRB.BINARY, name = 'u_Act')
    beta = model.addVars(T, len(seconds_MTU), vtype = GRB.BINARY, name = 'ifCleared') # 4s resolution parameter

    # To track the state of the system
    Esys = model.addVars(T, len(seconds_MTU), lb = Emin, ub = Emax, vtype = GRB.CONTINUOUS, name = 'Esys')
    
    # auxillary variable
    Ecl_up = model.addVars(T, len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'Ecleared_up')
    Ecl_down = model.addVars(T, len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'Ecleared_down')

    # Constraints on the system limits
    for t in T:
        # Bid MW is equal to or higher than the DA Capacity schedule
        model.addConstr(C_up[t] >= Cap_UP[t], name = f'UP_Capacity_schedule_limit{t}')
        model.addConstr(C_down[t] >= Cap_DOWN[t], name = f'DOWN_Capacity_schedule_limit{t}')
    
    # Bid Qty (MWh) is  less than or equal to the available Mwh
    model.addConstr(Einit - (deltaT*C_up[0]) >= Emin, name = f'Available_Mwh_UP_{0}')
    model.addConstr(Einit + (deltaT*C_down[0]) <= Emax, name = f'Available_Mwh_Down_{0}')
    for t in T[1:]:
        model.addConstr(Esys[t-1, len(seconds_MTU)-1] - (deltaT*C_up[t]) >= Emin, name = f'Available_Mwh_UP_{t}')
        model.addConstr(Esys[t-1, len(seconds_MTU)-1] + (deltaT*C_down[t]) <= Emax, name = f'Available_Mwh_Down_{t}')

    # Constraint for the System state after MTU activation
    model.addConstr(Einit + (Ecl_down[0,0] * (1 - u[0]) - Ecl_up[0,0] * u[0]) * alpha[0,0] - Esys[0,0] == 0, name = f'System_state_after_{0}MTU_{seconds_MTU[0]}s')

    for k in range(len(seconds_MTU))[1:]:
        model.addConstr(Esys[0,k-1] + (Ecl_down[0,k] * (1 - u[0])  - Ecl_up[0,k] * u[0]) * alpha[0,k] - Esys[0,k] == 0, name = f'System_state_after_{0}MTU_{seconds_MTU[k]}s')

    for t in T[1:]:
        model.addConstr(Esys[t-1, len(seconds_MTU)-1] + (Ecl_down[t,0] * (1 - u[t])  - Ecl_up[t,0] * u[t]) * alpha[t,0] - Esys[t,0] == 0, name = f'System_state_after_{t}MTU_{seconds_MTU[0]}s')
        for k in range(len(seconds_MTU))[1:]:
            model.addConstr(Esys[t,k-1] + (Ecl_down[t,k] * (1 - u[t])  - Ecl_up[t,k] * u[t]) * alpha[t,k] - Esys[t,k] == 0, name = f'System_state_after_{t}MTU_{seconds_MTU[k]}s')

    # Constraints on the price logic
    for t in T:
        # Bid price (Up) is spot price and Bid Price (Down) is 0 (-ve) when activated else Pricelimit/-Pricelimit
        model.addConstr(P_up[t] == (spotprice[t] + 10) * u[t] + pricelimit * (1 - u[t]), name = f'UP_price_limit_{t}')
        model.addConstr(P_down[t] == pricelimit * u[t], name = f'Down_price_limit_{t}')

    # constraint to enforce beta (Consider the revenue only when a bid is cleared)
    for t in T:
        for k in range(len(seconds_MTU)):
            model.addConstr(P_up[t] <= lambda_cl_UP[t,k] + M * (1 - beta[t,k]) + M * (1 - u[t]), name = f'UP_ifCleared_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(-P_down[t] >= lambda_cl_DOWN[t,k] - M * (1 - beta[t,k]) - M * u[t], name = f'Down_ifCleared_{t}MTU_{seconds_MTU[k]}s')
 
    #constraint to enforce auxillary variables
    for t in T:
        for k in range(len(seconds_MTU)):    
            model.addConstr(Ecl_up[t,k] <= ((4/3600)*C_up[t]), name = f'Ecleared_UP_a_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(Ecl_up[t,k] <= M * beta[t,k], name = f'Ecleared_UP_b_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(Ecl_up[t,k] >= ((4/3600)*C_up[t]) - M * (1 - beta[t,k]), name = f'Ecleared_UP_c_{t}MTU_{seconds_MTU[k]}s')

            model.addConstr(Ecl_down[t,k] <= ((4/3600)*C_down[t]), name = f'Ecleared_Down_a_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(Ecl_down[t,k] <= M * beta[t,k], name = f'Ecleared_Down_b_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(Ecl_down[t,k] >= ((4/3600)*C_down[t]) - M * (1 - beta[t,k]), name = f'Ecleared_Down_c_{t}MTU_{seconds_MTU[k]}s')
    
    # define the objective function
    Z = gp.quicksum(
        (gp.quicksum(lambda_cl_UP[t, k] * Ecl_up[t, k] * alpha[t, k] for k in range(len(seconds_MTU))) * u[t]) + 
        (gp.quicksum(-lambda_cl_DOWN[t,k] * Ecl_down[t,k] * alpha[t,k] for k in range(len(seconds_MTU))) * (1 - u[t]))
        for t in T)
    
    model.setObjective(Z, GRB.MAXIMIZE)

    model.optimize()

    return model

if __name__ == '__main__':
    horizon = 1
    activation_rate = 0.5

    # load the data from csv files
    aFRR_energy_prices = pd.read_csv('aFRRenergymarket_2025-03-10_to_2025-03-15.csv')
    spotprices = pd.read_csv('spotprices_2025-03-10_to_2025-03-15.csv')
    aFRR_energy_prices = aFRR_energy_prices.fillna(0)
    aFRR_energy_prices['SecondUTC'] = pd.to_datetime(aFRR_energy_prices['SecondUTC'])
    aFRR_energy_prices = aFRR_energy_prices.drop(columns=['PriceArea'])
    aFRR_energy_prices = aFRR_energy_prices.set_index('SecondUTC').resample('4s').mean().reset_index()
    spotprices['HourUTC'] = pd.to_datetime(spotprices['HourUTC'])
    df = pd.merge(aFRR_energy_prices, spotprices[['HourUTC','SpotPriceEUR']], how='left', left_on='SecondUTC', right_on='HourUTC')
    df = df.drop(columns=['HourUTC'])
    df['SpotPriceEUR'] =  df['SpotPriceEUR'].ffill()
    df['activation_rate'] = activation_rate
    df['UpCapacityBid'] = 0
    df['DownCapacityBid'] = 0

    md, result_dict, results = load_values(df, start = 1801, horizon = horizon)

    results['Bid (MW)'] = results.apply(lambda row: row['Bid Up (MW)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Bid Down (MW)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
    results['Energy Cleared (MWh)'] = results.apply(lambda row: row['Energy Up Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Energy Down Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
    results['Total Bid (Eur/MWh)'] = results.apply(lambda row: row['Pbid Up (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else row['Pbid Down (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

    # Add traces
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_UpActivatedPriceEUR'], mode='lines', name='aFRR_UpActivatedPriceEUR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['SpotPriceEUR'], mode='lines', name='SpotPriceEUR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Total Bid (Eur/MWh)'], mode='lines', name='Total Bid (Eur/MWh)'), row=1, col=1)

    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Energy Cleared (MWh)'], mode='lines', name='Energy Cleared (MWh)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['System state (MWh)'], mode='lines', name='System state (MWh)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Bid (MW)'], mode='lines', name='Total bid (MW)'), row=4, col=1)

    # Update layout with size and borders
    fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Cleared (MWh)',yaxis3_title='System state (MWh)', yaxis4_title='Bid MW',  
    legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

    for i in range(1, 5):
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

    fig.show()
