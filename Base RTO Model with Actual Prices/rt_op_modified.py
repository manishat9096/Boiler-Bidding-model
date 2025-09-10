import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Realtime_Optimisation():
    def __init__(self, day, afrrfile, spotpricefile, schedulefile, imbalancefile, horizon = 96):
        self.afrrfile = afrrfile
        self.spotpricefile = spotpricefile
        self.schedulefile = schedulefile
        self.imbalancefile = imbalancefile
        self.date = datetime.strptime(day, '%Y-%m-%d %H:%M:%S')
        self.end = self.date + timedelta(hours = horizon/4)
        self.horizon = horizon
        self.results = pd.DataFrame()
        
    def load_csv(self):
        folder_path = "D:\\ms\\January 2024\\Thesis\\Boiler-Bidding-model\\Datasets\\"
        
        aFRR_energy_prices = pd.read_csv(folder_path + self.afrrfile)
        spotprices = pd.read_csv(folder_path + self.spotpricefile)
        imbalanceprices = pd.read_csv(folder_path + self.imbalancefile)
        aFRR_energy_prices = aFRR_energy_prices.fillna(0)
        aFRR_energy_prices['SecondUTC'] = pd.to_datetime(aFRR_energy_prices['SecondUTC'])
        aFRR_energy_prices = aFRR_energy_prices.drop(columns=['PriceArea'])
        aFRR_energy_prices = aFRR_energy_prices.set_index('SecondUTC').resample('4s').mean().reset_index()
        spotprices['HourUTC'] = pd.to_datetime(spotprices['HourUTC'])
        imbalanceprices['TimeUTC'] = pd.to_datetime(imbalanceprices['TimeUTC'])
        df1 = pd.merge(aFRR_energy_prices, spotprices[['HourUTC','SpotPriceEUR']], how='left', left_on='SecondUTC', right_on='HourUTC')
        df1 = df1.drop(columns=['HourUTC'])
        df1['SpotPriceEUR'] =  df1['SpotPriceEUR'].ffill()
        df = pd.merge(df1, imbalanceprices[['TimeUTC','NewImbalancePrice']], how='left', left_on='SecondUTC', right_on='TimeUTC')
        df = df.drop(columns=['TimeUTC'])
        df['NewImbalancePrice'] =  df['NewImbalancePrice'].ffill()
        df = df[(df['SecondUTC'] >= self.date) & (df['SecondUTC'] <= self.end)].reset_index(drop=True)
       
        self.price_data = df

        self.boiler_schedule = pd.read_csv(folder_path + self.schedulefile)
        self.boiler_schedule['UTC'] = pd.to_datetime(self.boiler_schedule['UTC']).dt.tz_localize(None)
        self.boiler_schedule = self.boiler_schedule[['UTC','Heat load forecast [MW]','L11 EK plan [MW]','L11 Heat storage energy [MWh]']] #, 'Capacity Bid']]
        self.boiler_schedule['Heat load forecast [MW]'] = (self.boiler_schedule['Heat load forecast [MW]'] - 1.5).clip(lower=0)

        # self.boiler_schedule['Capacity bid direction'] = np.where(self.boiler_schedule['Capacity Bid'] > 0, 1, np.where(self.boiler_schedule['Capacity Bid'] < 0, -1, 0))

        # self.boiler_schedule = self.boiler_schedule[self.boiler_schedule['UTC'] >= pd.to_datetime(self.date)].reset_index(drop=True).iloc[0:96]
        self.boiler_schedule = self.boiler_schedule[(self.boiler_schedule['UTC'] >= self.date) & (self.boiler_schedule['UTC'] <= self.end)].reset_index(drop=True)

    def get_Einit(self, timestamp):
        self.load_csv()
        timestamp = pd.to_datetime(timestamp)
        return self.boiler_schedule[self.boiler_schedule['UTC'] == timestamp]['L11 Heat storage energy [MWh]'].iloc[0]

    def model(self):
        self.md = gp.Model('RTOptimisation')
        self.md.setParam("TimeLimit", 60)
        self.md.setParam('OutputFlag', 0)
        # Define time sets and constants
        T = list(range(self.horizon))
        seconds_MTU = list(range(4, 15 * 60 +1, 4)) # 4s intervals in a 15 minute step
        deltaT = 15/60 #15m interval in a hour
        deltaK = 4/3600 #4s interval in a hour
        M = 10e6
        W = self.system_data['W']

        # demand parameters - These paramters are for each MTU
        P_scheduled = self.boiler_schedule['L11 EK plan [MW]']
        # P_capacity_bid = self.boiler_schedule['Capacity Bid']
        # P_capacity_bid_direction = self.boiler_schedule['Capacity bid direction']
        demand = self.boiler_schedule['Heat load forecast [MW]']

        # boiler parameters
        P_max = self.system_data['P_max'] #MW
        Emax = self.system_data['Emax'] #MWh
        eff_P2H = self.system_data['eff_P2H']
        Einit = self.system_data['Einit'] #MWh
        eff_P2H_4s = eff_P2H**(1/225)
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
        z = self.md.addVars(T, vtype = GRB.BINARY, name = 'ifBid')
        
        # Constraints
        for t in T:
            self.md.addConstr(P_up[t] <= P_boiler[t], name = f'UP_Capacity_available_at_{t}')
            self.md.addConstr(P_down[t] <= (P_max - P_boiler[t]), name = f'Down_Capacity_available_at_{t}')
        
        # #constraints to enforce the DA capacity bids
        # for t in T:
        #     if P_capacity_bid_direction[t] == 1:
        #         # If bid is positive, enforce P_up[t] = P_capacity_bid[t] and z[t] = 1
        #         self.md.addConstr(P_up[t] == P_capacity_bid[t], name=f'force_P_up_if_positive_bid_{t}')
            
        #     elif P_capacity_bid_direction[t] == -1:
        #         self.md.addConstr(P_down[t] == -P_capacity_bid[t], name=f'force_P_down_if_negative_bid_{t}')

        # Bid Qty (in MWh over 15m) is  less than or equal to the available Mwh (if full activation)
        self.md.addConstr(Einit/Emax - (eff_P2H*deltaT*P_up[0])/Emax >= 0.1, name = f'Available_Mwh_UP_{0}')
        self.md.addConstr(Einit/Emax + (eff_P2H*deltaT*P_down[0])/Emax <= 0.9, name = f'Available_Mwh_Down_{0}')
        for t in T[1:]:
            self.md.addConstr(SoE[t-1, len(seconds_MTU)-1] - (eff_P2H*deltaT*P_up[t])/Emax >= 0.1, name = f'Available_Mwh_UP_{t}')
            self.md.addConstr(SoE[t-1, len(seconds_MTU)-1] + (eff_P2H*deltaT*P_down[t])/Emax <= 0.9, name = f'Available_Mwh_Down_{t}')
        
        # With new power setpoint, soe is  within limits (if no activation)
        # decreasing the buffer to 11% and 89% to account for decimal point errors
        self.md.addConstr(Einit/Emax + (eff_P2H*deltaT*P_boiler[0])/Emax - (deltaT*demand[0])/Emax  >= 0.1, name = f'SoE_lower_limit_{0}')
        self.md.addConstr(Einit/Emax + (eff_P2H*deltaT*P_boiler[0])/Emax - (deltaT*demand[0])/Emax <= 0.9, name = f'SoE_upper_limit_{0}')
        for t in T[1:]:
            self.md.addConstr(SoE[t-1, len(seconds_MTU)-1] + (eff_P2H*deltaT*P_boiler[t])/Emax - (deltaT*demand[t])/Emax >= 0.1, name = f'SoE_lower_limit_{t}')
            self.md.addConstr(SoE[t-1, len(seconds_MTU)-1] + (eff_P2H*deltaT*P_boiler[t])/Emax - (deltaT*demand[t])/Emax <= 0.9, name = f'SoE_upper_limit_{t}')

        for t in T:
            for k in range(len(seconds_MTU)):
                self.md.addConstr(E_up[t,k] <= eff_P2H_4s * P_up[t] * deltaK * u[t], name = f'E_UP_1_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_up[t,k] >= eff_P2H_4s * P_up[t] * deltaK * u[t] - M * (1 - beta[t,k]), name = f'E_UP_2_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_up[t,k] <= M * beta[t,k], name = f'E_UP_3_at_{t}MTU_{seconds_MTU[k]}s')

                self.md.addConstr(E_down[t,k] <= eff_P2H_4s * P_down[t] * deltaK * (1 - u[t]), name = f'E_DOWN_1_at_{t}MTU_{seconds_MTU[k]}s')
                self.md.addConstr(E_down[t,k] >= eff_P2H_4s * P_down[t] * deltaK * (1 - u[t]) - M * (1 - beta[t,k]), name = f'E_DOWN_2_at_{t}MTU_{seconds_MTU[k]}s')
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

        # Constraint for the System state after activation
        self.md.addConstr(SoE[0,0] == Einit/Emax + (deltaK * P_boiler[0] * eff_P2H_4s + E_down[0,0] - E_up[0,0])/Emax  - (deltaK * demand[0])/Emax, name = f'System_state_after_{0}MTU_{seconds_MTU[0]}s')

        for k in range(len(seconds_MTU))[1:]:
            self.md.addConstr(SoE[0,k] == SoE[0,k-1] + (deltaK * P_boiler[0] * eff_P2H_4s + E_down[0,k] - E_up[0,k])/Emax  - (deltaK * demand[0])/Emax, name = f'System_state_after_{0}MTU_{seconds_MTU[k]}s')

        for t in T[1:]:
            self.md.addConstr(SoE[t,0] == SoE[t-1, len(seconds_MTU)-1] + (deltaK * P_boiler[t] * eff_P2H_4s + E_down[t,0] - E_up[t,0])/Emax  - (deltaK * demand[t])/Emax, name = f'System_state_after_{t}MTU_{seconds_MTU[0]}s')
            for k in range(len(seconds_MTU))[1:]:
                self.md.addConstr(SoE[t,k] == SoE[t,k-1] + (deltaK * P_boiler[t] * eff_P2H_4s + E_down[t,k] - E_up[t,k])/Emax  - (deltaK * demand[t])/Emax, name = f'System_state_after_{t}MTU_{seconds_MTU[k]}s')

        # objective function

        deviation = gp.quicksum((P_boiler[t] - np.ceil(P_scheduled[t]))**2 for t in T)
        revenue = gp.quicksum(
                (gp.quicksum(lambda_cl_UP[t, k] * E_up[t, k] for k in range(len(seconds_MTU))) * u[t]) + 
                (gp.quicksum(-lambda_cl_DOWN[t,k] * E_down[t,k] for k in range(len(seconds_MTU))) * (1 - u[t]))
                for t in T) 
        Z =   W * deviation - revenue
        self.md._deviation_expr = deviation
        self.md._revenue_expr = revenue
        self.md._Z_expr = Z

        self.md.setObjective(Z, GRB.MINIMIZE)
        self.md.optimize()
        return self.md


    def optmize(self, Einit, W_factor, plot_flag = False):
        # system parameters
        self.system_data = {
            'P_max': 12, #MW
            'Emax': 60, #MWh
            'eff_SOE': 0.99, #Mwh
            'eff_P2H': 0.97,
            'Einit': Einit, #MWh
            'W': W_factor
        }
        sec_in_MTU = 225
        self.load_csv()
        self.price_data = self.price_data.iloc[0:(self.horizon*sec_in_MTU)].reset_index(drop = True)
        self.price_data_reshape = self.price_data.values.reshape(self.horizon, sec_in_MTU, -1)
        self.columns = {col: idx for idx, col in enumerate(self.price_data.columns)}

        self.model()
        
        if self.md.status == GRB.OPTIMAL:
            print("Optimization was successful.")
            
            objective_value = self.md.ObjVal  # Objective function value
            total_bid_UP = [self.md.getVarByName(f"P_up[{t}]").x for t in range(self.horizon)]
            total_bid_Down = [self.md.getVarByName(f"P_down[{t}]").x for t in range(self.horizon)]
            power_boiler = [self.md.getVarByName(f"P_boiler[{t}]").x for t in range(self.horizon)]
            energy_UP = [[self.md.getVarByName(f"E_up[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            energy_Down = [[self.md.getVarByName(f"E_down[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            system_state_SoE = [[self.md.getVarByName(f"SoE[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]
            system_state = [[(self.md.getVarByName(f"SoE[{t},{k}]").x * 60) for k in range(225)] for t in range(self.horizon)]
            bid_state = [self.md.getVarByName(f"ifBid[{t}]").x for t in range(self.horizon)]
            activation = [self.md.getVarByName(f"u_Act[{t}]").x for t in range(self.horizon)]
            cleared_status = [[self.md.getVarByName(f"ifCleared[{t},{k}]").x for k in range(225)] for t in range(self.horizon)]

            print(f"Total Objective value: {objective_value}")

            deviation = sum((self.price_data_reshape[:, 0, self.columns['NewImbalancePrice']][t] * (power_boiler[t] - np.ceil(self.boiler_schedule['L11 EK plan [MW]'][t]))) for t in range(self.horizon))

            modeldeviation = sum((power_boiler[t] - np.ceil(self.boiler_schedule['L11 EK plan [MW]'][t]))**2 for t in range(self.horizon))
            print(f"Model Deviation: {modeldeviation}")
            revenue_UP = sum(sum(self.price_data_reshape[:, :, self.columns['aFRR_UpActivatedPriceEUR']][t, k] * energy_UP[t][k] for k in range(225)) * activation[t]
                                for t in range(self.horizon))
            
            revenue_Down = sum(sum(-self.price_data_reshape[:, :, self.columns['aFRR_DownActivatedPriceEUR']][t, k] * energy_Down[t][k] for k in range(225)) * (1 - activation[t])
                                for t in range(self.horizon))
            revenue = revenue_UP + revenue_Down
            print(f"Deviation: {deviation}")
            # print(f"Revenue from Up: {revenue_UP}")
            # print(f"Revenue from Down: {revenue_Down}")
            print(f"Total Expected Revenue (model): {revenue}")
            print(f'calculate objective: {self.system_data['W'] * modeldeviation - revenue}')
            flattened_data = {
                'Objective Value': [objective_value] * (self.horizon * 225),
                'Deviation': [deviation] * (self.horizon * 225),
                'Total Revenue': [revenue] * (self.horizon * 225),
                'Bid Up (MW)': [item for sublist in total_bid_UP for item in [sublist] * 225],
                'Bid Down (MW)': [item for sublist in total_bid_Down for item in [sublist] * 225],
                'Power Boiler (MW)': [item for sublist in power_boiler for item in [sublist] * 225],
                'Energy Up Cleared (MWh)': [item for sublist in energy_UP for item in sublist],
                'Energy Down Cleared (MWh)': [item for sublist in energy_Down for item in sublist],
                'System SoE (%)': [item for sublist in system_state_SoE for item in sublist],
                'System State (MWh)': [item for sublist in system_state for item in sublist],
                'bid_state': [item for sublist in bid_state for item in [sublist] * 225],
                'activation': [item for sublist in activation for item in [sublist] * 225],
                'cleared_status': [item for sublist in cleared_status for item in sublist],
            }
            self.results = pd.DataFrame(flattened_data)
            self.results =  pd.concat([self.results, self.price_data], axis=1)
            self.results = pd.merge_asof(
                            self.results.sort_values('SecondUTC'), 
                            self.boiler_schedule.sort_values('UTC'), 
                            left_on='SecondUTC', 
                            right_on='UTC',
                            direction='backward'
                        )
            plot = plot_flag
            if plot == True:
                self.plot_results()
            # run the market_outcome check 
            # self.market_outcome() # not required when optimised over actual prices
        
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
        
        
        return self.md, self.results

    def market_outcome(self):
        self.results['aFRR_UpActivatedPriceEUR_actual'] = self.results['aFRR_UpActivatedPriceEUR']
        self.results['aFRR_DownActivatedPriceEUR_actual'] = self.results['aFRR_DownActivatedPriceEUR']
        # real clearing status from market (using actual prices)
        self.results['actual_bid_outcome'] = (((self.results['SpotPriceEUR'] < self.results['aFRR_UpActivatedPriceEUR_actual']) &(self.results['activation'] == 1)) |
                                             ((self.results['aFRR_DownActivatedPriceEUR_actual'] < 0) &(self.results['activation'] == 0))).astype(int)
        
        # energy activated in each 4s interval
        eff_P2H_4s = self.system_data['eff_P2H']**(1/225)
        deltaK = 4/3600
        Emax = self.system_data['Emax']
        self.results['Actual Energy Up Cleared (MWh)'] = (self.results['Bid Up (MW)'] * self.results['activation'] * self.results['actual_bid_outcome'] * eff_P2H_4s * deltaK)
        self.results['Actual Energy Down Cleared (MWh)'] = (self.results['Bid Down (MW)'] * (1 - self.results['activation']) * self.results['actual_bid_outcome'] * eff_P2H_4s * deltaK)

        # find SoE trajectory with actual activations over the horizon
        self.results['Actual System SoE (%)'] = 0.0
        self.results.iloc[0, self.results.columns.get_loc('Actual System SoE (%)')] = self.results.iloc[0, self.results.columns.get_loc('System SoE (%)')]

        for i in range(1, len(self.results)):
            prev = self.results.iloc[i - 1, self.results.columns.get_loc('Actual System SoE (%)')]
            boiler = self.results.iloc[i, self.results.columns.get_loc('Power Boiler (MW)')] * deltaK * eff_P2H_4s
            down = self.results.iloc[i, self.results.columns.get_loc('Actual Energy Down Cleared (MWh)')]
            up = self.results.iloc[i, self.results.columns.get_loc('Actual Energy Up Cleared (MWh)')]
            load = self.results.iloc[i, self.results.columns.get_loc('Heat load forecast [MW]')] * deltaK
            self.results.iloc[i, self.results.columns.get_loc('Actual System SoE (%)')] = prev + (boiler + down - up) / Emax - load / Emax

        # actual realised revenue 
        total_realised_revenue = (self.results['Actual Energy Up Cleared (MWh)'] * self.results['aFRR_UpActivatedPriceEUR_actual'] +
                                    self.results['Actual Energy Down Cleared (MWh)'] * -self.results['aFRR_DownActivatedPriceEUR_actual']).sum()
        print(f"Actual Realised Revenue: {total_realised_revenue}")
        self.results['Actual Realised Revenue'] = total_realised_revenue

        return

    def plot_results(self):
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

        # Add traces
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['aFRR_UpActivatedPriceEUR'], mode='lines', name='aFRR_UpActivatedPriceEUR'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['SpotPriceEUR'], mode='lines', name='SpotPriceEUR'), row=1, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['aFRR_DownActivatedPriceEUR'], mode='lines', name='aFRR_DownActivatedPriceEUR'), row=1, col=1)

        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Energy Up Cleared (MWh)'], mode='lines+markers', name='Energy Up Cleared (MWh)'), row=2, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Energy Down Cleared (MWh)'], mode='lines+markers', name='Energy Down Cleared (MWh)'), row=2, col=1)

        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['System SoE (%)'], mode='lines', name='System SoE (%)'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['bid_state'], mode='lines', name='bid_state'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['activation'], mode='lines', name='activation'), row=3, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['cleared_status'], mode='lines', name='cleared_status'), row=3, col=1)

        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Bid Up (MW)'], mode='lines', name='Bid Up (MW)'), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Bid Down (MW)'], mode='lines', name='Bid Down (MW)'), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['Power Boiler (MW)'], mode='lines', name='Power Boiler (MW)'), row=4, col=1)
        # fig.add_trace(go.Scatter(x=self.boiler_schedule['UTC'], y=self.boiler_schedule['L11 EK plan [MW]'], mode='lines', name='L11 EK plan [MW]'), row=4, col=1)
        fig.add_trace(go.Scatter(x=self.results['SecondUTC'], y=self.results['L11 EK plan [MW]'], mode='lines', name='L11 EK plan [MW]'), row=4, col=1)

        # Update layout with size and borders
        fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Cleared (MWh)',yaxis3_title='System state (%)', yaxis4_title='Bid MW',  
        legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

        for i in range(1, 5):
            fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
            fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

        fig.show()




if __name__ == '__main__': 
    day = '2025-03-01 04:00:00' # YYYY-MM-DD
    # afrrfile = 'aFRRenergymarket_2025-03-10_to_2025-03-15.csv'
    # spotpricefile = 'spotprices_2025-03-10_to_2025-03-15.csv'
    # schedulefile = 'boiler-12025031008_00_54_LHP_2_PHLIT_export.csv'
    afrrfile = 'aFRRenergymarket_2025-03-01_to_2025-03-31.csv'
    spotpricefile = 'spotprices_2025-03-01_to_2025-03-31.csv'
    schedulefile = 'LinHeat_schedule_March.csv'
    RTO = Realtime_Optimisation(day, afrrfile, spotpricefile, schedulefile, horizon = 24)
    model, results = RTO.optmize(Einit = 38.60732072595322, W_factor= 10)
    print(model.status)