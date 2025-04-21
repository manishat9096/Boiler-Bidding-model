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
    # initalise the decision variables
    C_up = model.addVar(lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'C_up')
    C_down = model.addVar(lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'C_down')
    P_up = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'P_up')
    P_down = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'P_down')
    u = model.addVar(vtype = GRB.BINARY, name = 'u_Act')
    beta = model.addVar(vtype = GRB.BINARY, name = 'ifCleared')
    deltaT = 15/60

    # auxillary variable
    Ecl_up = model.addVar(lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'Ecleared_up')
    Ecl_down = model.addVar(lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'Ecleared_down')
    Esys = model.addVar(lb = 0, vtype = GRB.CONTINUOUS, name = 'Esys')
    # define the objective function
    Z = (lambda_cl_UP * Ecl_up * activation * u) + (-lambda_cl_DOWN * Ecl_down * activation * (1-u) )

    model.setObjective(Z, GRB.MAXIMIZE)
    
    #define the constraints
    model.addLConstr(Einit - (deltaT*C_up) >= Emin, name = 'c1')
    model.addLConstr(Einit + (deltaT*C_down) <= Emax, name = 'c2')
    model.addLConstr(C_up >= Cap_UP, name = 'c3')
    model.addLConstr(C_down >= Cap_DOWN, name = 'c4')

    model.addLConstr(P_up == spotprice * u + pricelimit * (1 - u), name = 'c5')
    model.addLConstr(P_down == pricelimit * u, name = 'c6')

    # constraint to enforce beta
    model.addConstr(P_up <= lambda_cl_UP + M * (1 - beta) + M * (1 - u), name = 'c8')
    model.addConstr(-P_down >= lambda_cl_DOWN - M * (1 - beta) - M * u, name = 'c9')

    #constraint to enforce auxillary variables
    model.addConstr(Ecl_up <= (deltaT*C_up), name = 'c101')
    model.addConstr(Ecl_up <= M * beta, name = 'c102')
    model.addConstr(Ecl_up >= (deltaT*C_up) - M * (1 - beta), name = 'c103')

    model.addConstr(Ecl_down <= (deltaT*C_down), name = 'c111')
    model.addConstr(Ecl_down <= M * beta, name = 'c112')
    model.addConstr(Ecl_down >= (deltaT*C_down) - M * (1 - beta), name = 'c113')

    model.addConstr(Einit + (Ecl_down * (1 - u) - Ecl_up * u) * activation - Esys == 0, name = 'System_state_after_bid')
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
    MTU_df = df[df['SecondUTC'].dt.second == 0]
    MTU_df = MTU_df[MTU_df['SecondUTC'].dt.minute % 15 == 0]
    MTU_df.reset_index(drop = True, inplace = True)
    # output_singleMTU = load_values(MTU_df.iloc[12:22].reset_index(drop=True))
    test_by_row = True
    if test_by_row == True:
        results = pd.DataFrame()
        row = MTU_df.iloc[0]
        lambda_cl_UP = row['aFRR_UpActivatedPriceEUR']
        lambda_cl_DOWN = row['aFRR_DownActivatedPriceEUR']
        spotprice = row['SpotPriceEUR']
        activation = 0.5
        pricelimit = 15000
        Emin = 9 #MWh
        Emax = 51 #MWh
        Einit = 40 #MWh
        Cap_UP = 0 #MW
        Cap_DOWN = 0 #MW
        M = 1e6
        md = steady_state_single_MTU_OP(lambda_cl_UP, lambda_cl_DOWN, spotprice, activation, pricelimit, Emin, Emax, Cap_UP, Cap_DOWN, Einit, M) # Down Prices are negative (TSO pays BRP)
