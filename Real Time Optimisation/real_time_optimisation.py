import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_values(df, boiler_schedule, start = 0, horizon = 96):
    # system parameters
    system_data = {
        'P_max': 12, #MW
        'Emax': 60, #MWh
        'eff_SOE': 0.99, #Mwh
        'eff_P2H': 0.97,
        'Einit': 30 #MWh
    }
    sec_in_MTU = 225
    price_data = df.iloc[start:start + (horizon*sec_in_MTU)].reset_index(drop = True)
    price_data_reshape = price_data.values.reshape(horizon, sec_in_MTU, -1)
    columns = {col: idx for idx, col in enumerate(price_data.columns)}
    # afrr_up_price = price_data_reshape[:, :, columns['aFRR_UpActivatedPriceEUR']]
    # afrr_down_price = price_data_reshape[:, :, columns['aFRR_DownActivatedPriceEUR']]
    results = pd.DataFrame()
    md = steady_state_multiple_MTU_OP(price_data_reshape, columns, boiler_schedule, system_data, horizon)
    if md.status == GRB.OPTIMAL:
        print("Optimization was successful.")
        
        forecasted_revenue = - md.ObjVal  # Objective function value
        total_bid_UP = [md.getVarByName(f"P_up[{t}]").x for t in range(horizon)]
        total_bid_Down = [md.getVarByName(f"P_down[{t}]").x for t in range(horizon)]
        power_boiler = [md.getVarByName(f"P_boiler[{t}]").x for t in range(horizon)]
        energy_UP = [[md.getVarByName(f"E_up[{t},{k}]").x for k in range(225)] for t in range(horizon)]
        energy_Down = [[md.getVarByName(f"E_down[{t},{k}]").x for k in range(225)] for t in range(horizon)]
        system_state_SoE = [[md.getVarByName(f"SoE[{t},{k}]").x for k in range(225)] for t in range(horizon)]
        system_state = [[(md.getVarByName(f"SoE[{t},{k}]").x * 60) for k in range(225)] for t in range(horizon)]
        activation = [md.getVarByName(f"u_Act[{t}]").x for t in range(horizon)]
        cleared_status = [[md.getVarByName(f"ifCleared[{t},{k}]").x for k in range(225)] for t in range(horizon)]

        results_dict = {
            'Objective Value': forecasted_revenue,
            'Bid Up (MW)': total_bid_UP,
            'Bid Down (MW)': total_bid_Down,
            'Power Boiler(MW)': power_boiler,
            'Energy Up Cleared (MWh)': energy_UP,
            'Energy Down Cleared (MWh)': energy_Down,
            'System SoE(%)': system_state_SoE,
            'System State(MWh)': system_state,
            'activation': activation,
            'cleared_status': cleared_status
        }
        
        print(f"Total Revenue: {forecasted_revenue}")     

        flattened_data = {
            'Objective Value': [forecasted_revenue] * (horizon * 225),
            'Bid Up (MW)': [item for sublist in total_bid_UP for item in [sublist] * 225],
            'Bid Down (MW)': [item for sublist in total_bid_Down for item in [sublist] * 225],
            'Power Boiler (MW)': [item for sublist in power_boiler for item in [sublist] * 225],
            'Energy Up Cleared (MWh)': [item for sublist in energy_UP for item in sublist],
            'Energy Down Cleared (MWh)': [item for sublist in energy_Down for item in sublist],
            'System SoE (%)': [item for sublist in system_state_SoE for item in sublist],
            'System State (MWh)': [item for sublist in system_state for item in sublist],
            'activation': [item for sublist in activation for item in [sublist] * 225],
            'cleared_status': [item for sublist in cleared_status for item in sublist]
        }
        results = pd.DataFrame(flattened_data)
    
    # elif md.status == gp.GRB.INFEASIBLE:
    #     print("Optimization was not successful: INFEASIBLE")
    #     md.computeIIS()
    #     md.write("model.ilp")

    # elif md.status == gp.GRB.UNBOUNDED:
    #     print("Optimization was not successful: UNBOUNDED")

    # elif md.status == gp.GRB.INF_OR_UNBD:
    #     print("Optimization was not successful. Model is infeasible or unbounded.")
    #     md.write("model.lp") 

    # elif md.status != gp.GRB.OPTIMAL:
    #     print(f"Optimization was not successful. Status: {md.status}")

    results =  pd.concat([results, price_data], axis=1)
    # return md , results_dict, results
    return md, results

def steady_state_multiple_MTU_OP(price_data, columns, boiler_schedule, system_data, horizon):
    model = gp.Model('RTOptimisation')

    T = list(range(horizon))
    seconds_MTU = list(range(4, 15 * 60 +1, 4)) # 4s intervals in a 15 minute step
    deltaT = 4/3600  #4s interval in a hour
    M = 10e6
    # demand parameters - These paramters are for each MTU
    P_scheduled = boiler_schedule['L11 EK plan [MW]']
    demand = boiler_schedule['Heat load forecast [MW]']

    # boiler parameters
    P_max = system_data['P_max'] #MW
    Emax = system_data['Emax'] #MWh
    eff_SOE = system_data['eff_SOE'] 
    eff_P2H = system_data['eff_P2H']
    Einit = system_data['Einit'] #MWh

    # price parameters
    lambda_cl_UP = price_data[:, :, columns['aFRR_UpActivatedPriceEUR']] # These parameters are for every 4s interval of each MTU
    lambda_cl_DOWN = -price_data[:, :, columns['aFRR_DownActivatedPriceEUR']]
    spotprice = price_data[:, 0, columns['SpotPriceEUR']]

    # variables
    P_up = model.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'P_up')
    P_down = model.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'P_down')

    # Auxillary Variables
    P_boiler = model.addVars(T, lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'P_boiler')
    SoE = model.addVars(T, len(seconds_MTU), lb = 0.3, ub = 0.97, vtype = GRB.CONTINUOUS, name = 'SoE')
    E_up = model.addVars(T, len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'E_up') # Rows - MTU Columns - 4s intervals
    E_down = model.addVars(T, len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'E_down')

    # Binary variables    
    u = model.addVars(T, vtype = GRB.BINARY, name = 'u_Act')
    beta = model.addVars(T, len(seconds_MTU), vtype = GRB.BINARY, name = 'ifCleared') # 4s resolution parameter
    # beta_DOWN = model.addVars(T, len(seconds_MTU), vtype = GRB.BINARY, name = 'ifCleared_DOWN') # 4s resolution parameter

    # constraints
    # for t in T:
    #     model.addConstr(P_boiler[t] == np.ceil(P_scheduled[t]), name = f'Scheduled_Capacity_at_{t}')

    # Constraints
    for t in T:
        model.addConstr(P_up[t] <= P_boiler[t], name = f'UP_Capacity_available_at_{t}')
        model.addConstr(P_down[t] <= P_max - P_boiler[t], name = f'Down_Capacity_available_at_{t}')

    for t in T:
        for k in range(len(seconds_MTU)):
            model.addConstr(E_up[t,k] <= eff_P2H * P_up[t] * deltaT * u[t], name = f'E_UP_1_at_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(E_up[t,k] >= eff_P2H * P_up[t] * deltaT * u[t] - M * (1 - beta[t,k]), name = f'E_UP_2_at_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(E_up[t,k] <= M * beta[t,k], name = f'E_UP_3_at_{t}MTU_{seconds_MTU[k]}s')

            model.addConstr(E_down[t,k] <= eff_P2H * P_down[t] * deltaT * (1 - u[t]), name = f'E_DOWN_1_at_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(E_down[t,k] >= eff_P2H * P_down[t] * deltaT * (1 - u[t]) - M * (1 - beta[t,k]), name = f'E_DOWN_2_at_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(E_down[t,k] <= M * beta[t,k], name = f'E_DOWN_3_at_{t}MTU_{seconds_MTU[k]}s')

    # constraint to enforce beta (Consider the revenue only when a bid is cleared)
    for t in T:
        for k in range(len(seconds_MTU)):
            model.addConstr(spotprice[t] <= lambda_cl_UP[t,k] + M * (1 - beta[t,k]) + M * (1 - u[t]), name = f'UP_at_{t}MTU_{seconds_MTU[k]}s')
            model.addConstr(-lambda_cl_DOWN[t,k] >= M * (1 - beta[t,k]) - M * u[t], name = f'Down_at_{t}MTU_{seconds_MTU[k]}s')
 
    eff_SOE_4s = eff_SOE**(1/225)

    # Constraint for the System state after activation
    model.addConstr(SoE[0,0] == eff_SOE_4s * Einit/Emax + (deltaT * P_boiler[0] + E_down[0,0] - E_up[0,0])/Emax  - (deltaT * demand[0])/Emax, name = f'System_state_after_{0}MTU_{seconds_MTU[0]}s')

    for k in range(len(seconds_MTU))[1:]:
        model.addConstr(SoE[0,k] == eff_SOE_4s * SoE[0,k-1] + (deltaT * P_boiler[0] + E_down[0,k] - E_up[0,k])/Emax  - (deltaT * demand[0])/Emax, name = f'System_state_after_{0}MTU_{seconds_MTU[k]}s')

    for t in T[1:]:
        model.addConstr(SoE[t,0] == eff_SOE_4s * SoE[t-1, len(seconds_MTU)-1] + (deltaT * P_boiler[t] + E_down[t,0] - E_up[t,0])/Emax  - (deltaT * demand[t])/Emax, name = f'System_state_after_{t}MTU_{seconds_MTU[0]}s')
        for k in range(len(seconds_MTU))[1:]:
            model.addConstr(SoE[t,k] == eff_SOE_4s * SoE[t,k-1] + (deltaT * P_boiler[t] + E_down[t,k] - E_up[t,k])/Emax  - (deltaT * demand[t])/Emax, name = f'System_state_after_{t}MTU_{seconds_MTU[k]}s')

    # objective function
    W = 100
    Z = W * gp.quicksum((P_boiler[t] - np.ceil(P_scheduled[t]))**2 for t in T) - gp.quicksum(
            (gp.quicksum(lambda_cl_UP[t, k] * E_up[t, k] for k in range(len(seconds_MTU))) * u[t]) + 
            (gp.quicksum(-lambda_cl_DOWN[t,k] * E_down[t,k] for k in range(len(seconds_MTU))) * (1 - u[t]))
            for t in T) 

    model.setObjective(Z, GRB.MINIMIZE)
    model.optimize()
    return model

if __name__ == '__main__':
    horizon = 8 # 15 min blocks over a day
    folder_path = "D:\\ms\\January 2024\\Thesis\\Boiler-Bidding-model\\Datasets\\"

    # load the data from csv files
    aFRR_energy_prices = pd.read_csv(folder_path +'aFRRenergymarket_2025-03-10_to_2025-03-15.csv')
    spotprices = pd.read_csv(folder_path +'spotprices_2025-03-10_to_2025-03-15.csv')
    linheat_df = pd.read_csv(folder_path +'boiler-12025041008_15_13_LHP_2_PHLIT_export.csv')
    linheat_df = linheat_df[['Heat load forecast [MW]','L11 EK plan [MW]']].iloc[0:96]
    aFRR_energy_prices = aFRR_energy_prices.fillna(0)
    aFRR_energy_prices['SecondUTC'] = pd.to_datetime(aFRR_energy_prices['SecondUTC'])
    aFRR_energy_prices = aFRR_energy_prices.drop(columns=['PriceArea'])
    aFRR_energy_prices = aFRR_energy_prices.set_index('SecondUTC').resample('4s').mean().reset_index()
    spotprices['HourUTC'] = pd.to_datetime(spotprices['HourUTC'])
    df = pd.merge(aFRR_energy_prices, spotprices[['HourUTC','SpotPriceEUR']], how='left', left_on='SecondUTC', right_on='HourUTC')
    df = df.drop(columns=['HourUTC'])
    df['SpotPriceEUR'] =  df['SpotPriceEUR'].ffill()
    df = df[df['SecondUTC'] >= pd.to_datetime('2025-03-10 00:00:00')].reset_index(drop=True)
    # manually changing linheat_df
    # linheat_df.loc[0:8, 'L11 EK plan [MW]'] = 0
    md, results = load_values(df, linheat_df, start = 0, horizon = horizon)

    # results['Bid (MW)'] = results.apply(lambda row: row['Bid Up (MW)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Bid Down (MW)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
    # results['Energy Cleared (MWh)'] = results.apply(lambda row: row['Energy Up Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Energy Down Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
    # results['Total Bid (Eur/MWh)'] = results.apply(lambda row: row['Pbid Up (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else row['Pbid Down (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)

    if md.status == GRB.OPTIMAL:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

        # Add traces
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_UpActivatedPriceEUR'], mode='lines', name='aFRR_UpActivatedPriceEUR'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['SpotPriceEUR'], mode='lines', name='SpotPriceEUR'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_DownActivatedPriceEUR'], mode='lines', name='aFRR_DownActivatedPriceEUR'), row=1, col=1)

        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Energy Up Cleared (MWh)'] * results['cleared_status'], mode='lines+markers', name='Energy Up Cleared (MWh)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Energy Down Cleared (MWh)'] * results['cleared_status'], mode='lines+markers', name='Energy Down Cleared (MWh)'), row=2, col=1)

        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['System SoE (%)'], mode='lines', name='System SoE (%)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['activation'], mode='lines', name='activation'), row=3, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['cleared_status'], mode='lines', name='cleared_status'), row=3, col=1)


        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Bid Up (MW)'], mode='lines', name='Bid Up (MW)'), row=4, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Bid Down (MW)'], mode='lines', name='Bid Down (MW)'), row=4, col=1)
        fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Power Boiler (MW)'], mode='lines', name='Power Boiler (MW)'), row=4, col=1)

        # Update layout with size and borders
        fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Cleared (MWh)',yaxis3_title='System state (%)', yaxis4_title='Bid MW',  
        legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

        for i in range(1, 5):
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

        fig.show()