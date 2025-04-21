import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Realtime_Optimisation():
    def __init__(self, day, afrrfile, spotpricefile, schedulefile, horizon = 96):
        self.afrrfile = afrrfile
        self.spotpricefile = spotpricefile
        self.schedulefile = schedulefile
        self.date = datetime.strptime(day, '%Y-%m-%d').strftime('%Y-%m-%d %H:%M:%S')
        self.horizon = horizon
        self.results = pd.DataFrame()
        
    def load_csv(self):
        folder_path = "D:\\ms\\January 2024\\Thesis\\Boiler-Bidding-model\\Datasets\\"
        
        aFRR_energy_prices = pd.read_csv(folder_path + self.afrrfile)
        spotprices = pd.read_csv(folder_path + self.spotpricefile)
        aFRR_energy_prices = aFRR_energy_prices.fillna(0)
        aFRR_energy_prices['SecondUTC'] = pd.to_datetime(aFRR_energy_prices['SecondUTC'])
        aFRR_energy_prices = aFRR_energy_prices.drop(columns=['PriceArea'])
        aFRR_energy_prices = aFRR_energy_prices.set_index('SecondUTC').resample('4s').mean().reset_index()
        spotprices['HourUTC'] = pd.to_datetime(spotprices['HourUTC'])
        df = pd.merge(aFRR_energy_prices, spotprices[['HourUTC','SpotPriceEUR']], how='left', left_on='SecondUTC', right_on='HourUTC')
        df = df.drop(columns=['HourUTC'])
        df['SpotPriceEUR'] =  df['SpotPriceEUR'].ffill()
        df = df[df['SecondUTC'] >= pd.to_datetime(self.date)].reset_index(drop=True)
        self.price_data = df

        self.boiler_schedule = pd.read_csv(folder_path + self.schedulefile)
        self.boiler_schedule['UTC'] = pd.to_datetime(self.boiler_schedule['UTC']).dt.tz_localize(None)
        self.boiler_schedule = self.boiler_schedule[['UTC','Heat load forecast [MW]','L11 EK plan [MW]']]
        self.boiler_schedule = self.boiler_schedule[self.boiler_schedule['UTC'] >= pd.to_datetime(self.date)].reset_index(drop=True).iloc[0:96]

    def model(self):
        self.md = gp.Model('RTOptimisation')

        # Define time sets and constants
        T = list(range(self.horizon))
        seconds_MTU = list(range(4, 15 * 60 +1, 4)) # 4s intervals in a 15 minute step
        deltaT = 15/60 #15m interval in a hour
        deltaK = 4/3600 #4s interval in a hour
        M = 10e6

        # demand parameters - These paramters are for each MTU
        P_scheduled = self.boiler_schedule['L11 EK plan [MW]']
        demand = self.boiler_schedule['Heat load forecast [MW]']

        # boiler parameters
        P_max = self.system_data['P_max'] #MW
        Emax = self.system_data['Emax'] #MWh
        eff_SOE = self.system_data['eff_SOE'] 
        eff_P2H = self.system_data['eff_P2H']
        Einit = self.system_data['Einit'] #MWh

        # price parameters
        lambda_cl_UP = self.price_data_reshape[:, :, self.columns['aFRR_UpActivatedPriceEUR']] # These parameters are for every 4s interval of each MTU
        lambda_cl_DOWN = self.price_data_reshape[:, :, self.columns['aFRR_DownActivatedPriceEUR']]
        spotprice = self.price_data_reshape[:, 0, self.columns['SpotPriceEUR']]

        # variables
        P_up = self.md.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'P_up')
        P_down = self.md.addVars(T, lb = 0, vtype = GRB.INTEGER, name = 'P_down')

        # Auxillary Variables
        P_boiler = self.md.addVars(T, lb = 0, ub = 12, vtype = GRB.INTEGER, name = 'P_boiler')
        SoE = self.md.addVars(T, len(seconds_MTU), lb = 0, ub = 1, vtype = GRB.CONTINUOUS, name = 'SoE')
        E_up = self.md.addVars(T, len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'E_up') # Rows - MTU Columns - 4s intervals
        E_down = self.md.addVars(T, len(seconds_MTU), lb = 0, ub = 3, vtype = GRB.CONTINUOUS, name = 'E_down')

        # Binary variables    
        u = self.md.addVars(T, vtype = GRB.BINARY, name = 'u_Act')
        beta = self.md.addVars(T, len(seconds_MTU), vtype = GRB.BINARY, name = 'ifCleared') # 4s resolution parameter
        # beta_DOWN = self.md.addVars(T, len(seconds_MTU), vtype = GRB.BINARY, name = 'ifCleared_DOWN') # 4s resolution parameter

        # constraints
        # for t in T:
        #     self.md.addConstr(P_boiler[t] == np.ceil(P_scheduled[t]), name = f'Scheduled_Capacity_at_{t}')

        # Constraints
        for t in T:
            self.md.addConstr(P_up[t] <= P_boiler[t], name = f'UP_Capacity_available_at_{t}')
            self.md.addConstr(P_down[t] <= P_max - P_boiler[t], name = f'Down_Capacity_available_at_{t}')

        # Bid Qty (in MWh over 15m) is  less than or equal to the available Mwh
        self.md.addConstr(eff_SOE * Einit/Emax - (deltaT*P_up[0])/Emax >= 0.1, name = f'Available_Mwh_UP_{0}')
        self.md.addConstr(eff_SOE * Einit/Emax + (deltaT*P_down[0])/Emax <= 0.9, name = f'Available_Mwh_Down_{0}')
        for t in T[1:]:
            self.md.addConstr(eff_SOE * SoE[t-1, len(seconds_MTU)-1] - (deltaT*P_up[t])/Emax >= 0.1, name = f'Available_Mwh_UP_{t}')
            self.md.addConstr(eff_SOE * SoE[t-1, len(seconds_MTU)-1] + (deltaT*P_down[t])/Emax <= 0.9, name = f'Available_Mwh_Down_{t}')

        for t in T:
            for k in range(len(seconds_MTU)):
                self.md.addConstr(E_up[t,k] <= eff_P2H * P_up[t] * deltaK * u[t], name = f'E_UP_1_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_up[t,k] >= eff_P2H * P_up[t] * deltaK * u[t] - M * (1 - beta[t,k]), name = f'E_UP_2_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_up[t,k] <= M * beta[t,k], name = f'E_UP_3_at_{t}MTU_{seconds_MTU[k]}s')

                self.md.addConstr(E_down[t,k] <= eff_P2H * P_down[t] * deltaK * (1 - u[t]), name = f'E_DOWN_1_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_down[t,k] >= eff_P2H * P_down[t] * deltaK * (1 - u[t]) - M * (1 - beta[t,k]), name = f'E_DOWN_2_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_down[t,k] <= M * beta[t,k], name = f'E_DOWN_3_at_{t}MTU_{seconds_MTU[k]}s')

        # constraint to enforce beta (Consider the revenue only when a bid is cleared)
        for t in T:
            for k in range(len(seconds_MTU)):
                # self.md.addConstr(spotprice[t] <= lambda_cl_UP[t,k] + M * (1 - beta[t,k]) + M * (1 - u[t]), name = f'UP_at_{t}MTU_{seconds_MTU[k]}s')
                # self.md.addConstr(lambda_cl_DOWN[t,k] <= M * (1 - beta[t,k]) + M * u[t], name = f'Down_at_{t}MTU_{seconds_MTU[k]}s')
                # When u[t] = 1, beta[t,k] should be 1 if spotprice[t] <= lambda_cl_UP[t,k]
                self.md.addConstr((u[t] == 1) >> (beta[t,k] == (spotprice[t] <= lambda_cl_UP[t,k])))
                
                # When u[t] = 0, beta[t,k] should be 1 if lambda_cl_DOWN[t,k] < 0
                self.md.addConstr((u[t] == 0) >> (beta[t,k] == (lambda_cl_DOWN[t,k] < 0)))
                
        eff_SOE_4s = eff_SOE**(1/225)

        # Constraint for the System state after activation
        self.md.addConstr(SoE[0,0] == eff_SOE_4s * Einit/Emax + (deltaK * P_boiler[0] + E_down[0,0] - E_up[0,0])/Emax  - (deltaK * demand[0])/Emax, name = f'System_state_after_{0}MTU_{seconds_MTU[0]}s')

        for k in range(len(seconds_MTU))[1:]:
            self.md.addConstr(SoE[0,k] == eff_SOE_4s * SoE[0,k-1] + (deltaK * P_boiler[0] + E_down[0,k] - E_up[0,k])/Emax  - (deltaK * demand[0])/Emax, name = f'System_state_after_{0}MTU_{seconds_MTU[k]}s')

        for t in T[1:]:
            self.md.addConstr(SoE[t,0] == eff_SOE_4s * SoE[t-1, len(seconds_MTU)-1] + (deltaK * P_boiler[t] + E_down[t,0] - E_up[t,0])/Emax  - (deltaK * demand[t])/Emax, name = f'System_state_after_{t}MTU_{seconds_MTU[0]}s')
            for k in range(len(seconds_MTU))[1:]:
                self.md.addConstr(SoE[t,k] == eff_SOE_4s * SoE[t,k-1] + (deltaK * P_boiler[t] + E_down[t,k] - E_up[t,k])/Emax  - (deltaK * demand[t])/Emax, name = f'System_state_after_{t}MTU_{seconds_MTU[k]}s')

        # objective function
        # W = 100         W * gp.quicksum((P_boiler[t] - np.ceil(P_scheduled[t]))**2 for t in T)
        Z =  - gp.quicksum(
                (gp.quicksum(lambda_cl_UP[t, k] * E_up[t, k] for k in range(len(seconds_MTU))) * u[t]) + 
                (gp.quicksum(-lambda_cl_DOWN[t,k] * E_down[t,k] for k in range(len(seconds_MTU))) * (1 - u[t]))
                for t in T) 

        self.md.setObjective(Z, GRB.MINIMIZE)
        self.md.optimize()
        return self.md


    def optmize(self, Einit):
        # system parameters
        self.system_data = {
            'P_max': 12, #MW
            'Emax': 60, #MWh
            'eff_SOE': 0.99, #Mwh
            'eff_P2H': 0.97,
            'Einit': Einit #MWh
        }
        sec_in_MTU = 225
        self.load_csv()
        self.price_data = self.price_data.iloc[0:(self.horizon*sec_in_MTU)].reset_index(drop = True)
        self.price_data_reshape = self.price_data.values.reshape(self.horizon, sec_in_MTU, -1)
        self.columns = {col: idx for idx, col in enumerate(self.price_data.columns)}

        self.model()
        
        if self.md.status == GRB.OPTIMAL:
            print("Optimization was successful.")
            
            revenue = - self.md.ObjVal  # Objective function value
            total_bid_UP = [self.md.getVarByName(f"P_up[{t}]").x for t in range(self.horizon)]
            total_bid_Down = [self.md.getVarByName(f"P_down[{t}]").x for t in range(self.horizon)]
            power_boiler = [self.md.getVarByName(f"P_boiler[{t}]").x for t in range(self.horizon)]
            energy_UP = [[self.md.getVarByName(f"E_up[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            energy_Down = [[self.md.getVarByName(f"E_down[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            system_state_SoE = [[self.md.getVarByName(f"SoE[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            system_state = [[(self.md.getVarByName(f"SoE[{t},{k}]").x * 60) for k in range(225)] for t in range(self.horizon)]
            activation = [self.md.getVarByName(f"u_Act[{t}]").x for t in range(self.horizon)]
            cleared_status_up = [[self.md.getVarByName(f"ifCleared[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            # cleared_status_down = [[self.md.getVarByName(f"ifCleared_DOWN[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]

            print(f"Total Revenue: {revenue}")     

            flattened_data = {
                'Objective Value': [revenue] * (self.horizon * 225),
                'Bid Up (MW)': [item for sublist in total_bid_UP for item in [sublist] * 225],
                'Bid Down (MW)': [item for sublist in total_bid_Down for item in [sublist] * 225],
                'Power Boiler (MW)': [item for sublist in power_boiler for item in [sublist] * 225],
                'Energy Up Cleared (MWh)': [item for sublist in energy_UP for item in sublist],
                'Energy Down Cleared (MWh)': [item for sublist in energy_Down for item in sublist],
                'System SoE (%)': [item for sublist in system_state_SoE for item in sublist],
                'System State (MWh)': [item for sublist in system_state for item in sublist],
                'activation': [item for sublist in activation for item in [sublist] * 225],
                'cleared_status_up': [item for sublist in cleared_status_up for item in sublist],
                # 'cleared_status_down': [item for sublist in cleared_status_down for item in sublist]
            }
            self.results = pd.DataFrame(flattened_data)
        
        elif self.md.status == gp.GRB.INFEASIBLE:
            print("Optimization was not successful: INFEASIBLE")
            # self.md.computeIIS()
            # self.md.write("model.ilp")

        elif self.md.status == gp.GRB.UNBOUNDED:
            print("Optimization was not successful: UNBOUNDED")

        elif self.md.status == gp.GRB.INF_OR_UNBD:
            print("Optimization was not successful. Model is infeasible or unbounded.")
            # md.write("model.lp") 

        elif self.md.status != gp.GRB.OPTIMAL:
            print(f"Optimization was not successful. Status: {self.md.status}")
        
        self.results =  pd.concat([self.results, self.price_data], axis=1)
        self.plot_results()
        return self.md, self.results

    def plot_results(self):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

        # Add traces
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['aFRR_UpActivatedPriceEUR'], mode='lines', name='aFRR_UpActivatedPriceEUR'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['SpotPriceEUR'], mode='lines', name='SpotPriceEUR'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['aFRR_DownActivatedPriceEUR'], mode='lines', name='aFRR_DownActivatedPriceEUR'), row=1, col=1)

        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Energy Up Cleared (MWh)'], mode='lines+markers', name='Energy Up Cleared (MWh)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Energy Down Cleared (MWh)'], mode='lines+markers', name='Energy Down Cleared (MWh)'), row=2, col=1)

        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['System SoE (%)'], mode='lines', name='System SoE (%)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['activation'], mode='lines', name='activation'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['cleared_status_up'], mode='lines', name='cleared_status_up'), row=3, col=1)
        # fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['cleared_status_down'], mode='lines', name='cleared_status_down'), row=3, col=1)


        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Bid Up (MW)'], mode='lines', name='Bid Up (MW)'), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Bid Down (MW)'], mode='lines', name='Bid Down (MW)'), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Power Boiler (MW)'], mode='lines', name='Power Boiler (MW)'), row=4, col=1)

        # Update layout with size and borders
        fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Cleared (MWh)',yaxis3_title='System state (%)', yaxis4_title='Bid MW',  
        legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

        for i in range(1, 5):
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

        fig.show()




if __name__ == '__main__': 
    day = '2025-04-01' # YYYY-MM-DD
    # afrrfile = 'aFRRenergymarket_2025-03-10_to_2025-03-15.csv'
    # spotpricefile = 'spotprices_2025-03-10_to_2025-03-15.csv'
    # schedulefile = 'boiler-12025031008_00_54_LHP_2_PHLIT_export.csv'
    afrrfile = 'aFRRenergymarket_2025-04-01_to_2025-04-20.csv'
    spotpricefile = 'spotprices_2025-04-01_to_2025-04-20.csv'
    schedulefile = 'linheat_schedule_2025-04-01_to_2025-04-20.csv'
    RTO = Realtime_Optimisation(day, afrrfile, spotpricefile, schedulefile, horizon = 20)
    model, results = RTO.optmize(Einit = 30)
    print(model.status)