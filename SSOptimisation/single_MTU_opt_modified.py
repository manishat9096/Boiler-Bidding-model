import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import matplotlib.pyplot as plt

"""
To initialise all known parameters of the single MTU steady state optimisation, I am using a csv file later
Here all values are arbitrarily set to test the optimisation 
"""

def load_values(df):
    results = pd.DataFrame()
    Einit = 40 #MWh
    for idx, row in df.iterrows():
        lambda_cl_UP = row['aFRR_UpActivatedPriceEUR']
        lambda_cl_DOWN = row['aFRR_DownActivatedPriceEUR']
        spotprice = row['SpotPriceEUR']
        activation = 0.5
        pricelimit = 15000
        Emin = 9 #MWh
        Emax = 51 #MWh
        Cap_UP = 0 #MW
        Cap_DOWN = 0 #MW
        M = 1e6
        md = steady_state_single_MTU_OP(lambda_cl_UP, -lambda_cl_DOWN, spotprice, activation, pricelimit, Emin, Emax, Cap_UP, Cap_DOWN, Einit, M)
        if md.status == GRB.OPTIMAL:
            result = {
                'Objective Value': md.ObjVal,
                'Bid Up (MW)': md.getVarByName("C_up").x,
                'Bid Down (MW)': md.getVarByName("C_down").x,
                'ECleared Up (MWh)': md.getVarByName("Ecleared_up").x,
                'ECleared Down (MWh)': md.getVarByName("Ecleared_down").x,
                'Pbid Up (Eur/MWh)': md.getVarByName("P_up").x,
                'Pbid Down (Eur/MWh)': -md.getVarByName("P_down").x,
                'Activation 1 - Up 0 - Down': md.getVarByName("u_Act").x,
                'Cleared 1 - Yes 0 - No': md.getVarByName("ifCleared").x,
                'System state': md.getVarByName('Esys').x
            }
            
            results = pd.concat([results, pd.DataFrame([result])], ignore_index=True)
            
            if md.getVarByName("u_Act").x == 1:
                Einit -= md.getVarByName("Ecleared_up").x * activation
            elif md.getVarByName("u_Act").x == 0:
                Einit += md.getVarByName("Ecleared_down").x * activation
    return results

def steady_state_single_MTU_OP(lambda_cl_UP, lambda_cl_DOWN, spotprice, activation, pricelimit, Emin, Emax, Cap_UP, Cap_DOWN, Einit, M): # Down Prices are negative (TSO pays BRP)
    model = gp.Model('Steady_state_1MTU_problem')
    
    seconds_MTU = list(range(4, 15 * 60 +1, 4))
    # initalise the decision variables
    C_up = model.addVar(lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'C_up')
    C_down = model.addVar(lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'C_down')
    P_up = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'P_up')
    P_down = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'P_down')
    u = model.addVar(vtype = GRB.BINARY, name = 'u_Act')
    beta = model.addVars(len(seconds_MTU), vtype = GRB.BINARY, name = 'ifCleared')
    deltaT = 15/60

    # auxillary variable
    Ecl_up = model.addVars(len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'Ecleared_up')
    Ecl_down = model.addVars(len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'Ecleared_down')
    Esys = model.addVars(len(seconds_MTU), lb = 0, vtype = GRB.CONTINUOUS, name = 'Esys')


    # define the objective function
    Z = gp.quicksum(lambda_cl_UP[k] * Ecl_up[k] * activation[k] for k in range(len(seconds_MTU))) * u + \
        gp.quicksum(-lambda_cl_DOWN[k] * Ecl_down[k] * activation[k] for k in range(len(seconds_MTU))) * (1-u)

    model.setObjective(Z, GRB.MAXIMIZE)
    
    #define the constraints
    model.addLConstr(Einit - (deltaT*C_up) >= Emin, name = 'c1')
    model.addLConstr(Einit + (deltaT*C_down) <= Emax, name = 'c2')
    model.addLConstr(C_up >= Cap_UP, name = 'c3')
    model.addLConstr(C_down >= Cap_DOWN, name = 'c4')

    model.addLConstr(P_up == spotprice[0] * u + pricelimit * (1 - u), name = 'c5')
    model.addLConstr(P_down == pricelimit * u, name = 'c6')

    # constraint to enforce beta
    for k in range(len(seconds_MTU)):
        model.addConstr(P_up <= lambda_cl_UP[k] + M * (1 - beta[k]) + M * (1 - u), name = f'c8_{seconds_MTU[k]}')
        model.addConstr(-P_down >= lambda_cl_DOWN[k] - M * (1 - beta[k]) - M * u, name = f'c9_{seconds_MTU[k]}')

        #constraint to enforce auxillary variables
        model.addConstr(Ecl_up[k] <= ((4/3600)*C_up), name = f'c101_{seconds_MTU[k]}')
        model.addConstr(Ecl_up[k] <= M * beta[k], name = f'c102_{seconds_MTU[k]}')
        model.addConstr(Ecl_up[k] >= ((4/3600)*C_up) - M * (1 - beta[k]), name = f'c103_{seconds_MTU[k]}')

        model.addConstr(Ecl_down[k] <= ((4/3600)*C_down), name = f'c111_{seconds_MTU[k]}')
        model.addConstr(Ecl_down[k] <= M * beta[k], name = f'c112_{seconds_MTU[k]}')
        model.addConstr(Ecl_down[k] >= ((4/3600)*C_down) - M * (1 - beta[k]), name = f'c113_{seconds_MTU[k]}')

    model.addConstr(Einit + (Ecl_down[0] * (1 - u) - Ecl_up[0] * u) * activation[0] - Esys[0] == 0, name = f'System_state_after_{seconds_MTU[0]}s')
    for k in range(1, len(seconds_MTU)):
        model.addConstr(Esys[k-1] + (Ecl_down[k] * (1 - u) - Ecl_up[k] * u) * activation[k] - Esys[k] == 0, name = f'System_state_after_{seconds_MTU[k]}s')

    model.optimize()
    return model


if __name__ == '__main__':
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
    activation_rate = 0.5
    df['activation_rate'] = activation_rate
    df['MTU'] = (df['SecondUTC'].dt.hour * 4) + (df['SecondUTC'].dt.minute // 15)
    df['Date'] = df['SecondUTC'].dt.day
    test_by_row = True
    if test_by_row == True:
        MTU_df = df.loc[(df['MTU'] == 92) & (df['Date'] == 9)].reset_index(drop = True)    
        lambda_cl_UP = MTU_df['aFRR_UpActivatedPriceEUR']
        lambda_cl_DOWN = -MTU_df['aFRR_DownActivatedPriceEUR']
        spotprice = MTU_df['SpotPriceEUR']
        activation = MTU_df['activation_rate']
        pricelimit = 15000
        Emin = 9 #MWh
        Emax = 51 #MWh
        Einit = 40 #MWh
        Cap_UP = 0 #MW
        Cap_DOWN = 0 #MW
        M = 1e6
        md = steady_state_single_MTU_OP(lambda_cl_UP, lambda_cl_DOWN, spotprice, activation, pricelimit, Emin, Emax, Cap_UP, Cap_DOWN, Einit, M) # Down Prices are negative (TSO pays BRP)
        if md.status == GRB.OPTIMAL:
            seconds_MTU = list(range(4, 15 * 60 +1, 4))
            result = {
                'Objective Value': md.ObjVal,
                'Bid Up (MW)': md.getVarByName("C_up").x,
                'Bid Down (MW)': md.getVarByName("C_down").x,
                'ECleared Up (MWh)': [np.round(md.getVarByName(f"Ecleared_up[{t}]").x,6) for t in range(len(seconds_MTU))],
                'ECleared Down (MWh)': [np.round(md.getVarByName(f"Ecleared_down[{t}]").x,6) for t in range(len(seconds_MTU))],
                'Pbid Up (Eur/MWh)': md.getVarByName("P_up").x,
                'Pbid Down (Eur/MWh)': -md.getVarByName("P_down").x,
                'Activation 1 - Up 0 - Down': md.getVarByName("u_Act").x,
                'Cleared 1 - Yes 0 - No': [md.getVarByName(f"ifCleared[{t}]").x for t in range(len(seconds_MTU))],
                'System state': [np.round(md.getVarByName(f'Esys[{t}]').x,6) for t in range(len(seconds_MTU))]
            }
        results = pd.DataFrame(result)
        results['Bid Placed (MW)'] = results.apply(lambda row: row['Bid Up (MW)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Bid Down (MW)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
        results['Energy Cleared (MWh)'] = results.apply(lambda row: row['ECleared Up (MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['ECleared Down (MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
        results['Total Bid (Eur/MWh)'] = results.apply(lambda row: row['Pbid Up (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else row['Pbid Down (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

        # Add traces
        fig.add_trace(go.Scatter(x=results.index, y=lambda_cl_UP, mode='lines', name='act'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=spotprice, mode='lines', name='sp'), row=1, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=results['Pbid Up (Eur/MWh)'], mode='lines', name='bid'), row=1, col=1)

        fig.add_trace(go.Scatter(x=results.index, y=results['Energy Cleared (MWh)'], mode='lines', name='Total_bids (MWh)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=results['System state'], mode='lines', name='Total_bids (MW)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=results.index, y=results['Bid Placed (MW)'], mode='lines', name='Total_bids (MW)'), row=4, col=1)

        # Update layout with size and borders
        fig.update_layout(title='Output of OP',xaxis_title='Time',yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Activated (MWh)',yaxis3_title='System state (MWh)', yaxis4_title='Bid MW',  
        legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

        for i in range(1, 5):
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

        fig.show()
