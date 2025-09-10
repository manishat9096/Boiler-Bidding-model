import numpy as np
import pandas as pd
from rt_op_modified import Realtime_Optimisation
from model_simulator import ARXSimulator
from datetime import datetime, timedelta
from tqdm import tqdm

class RecursiveRTO():
    def __init__(self, start_date, end_date, print_flag = False, plot_flag = False):
        self.afrrfile = 'aFRRenergymarket_2025-03-01_to_2025-03-31.csv'
        self.spotpricefile = 'spotprices_2025-03-01_to_2025-03-31.csv'
        self.schedulefile = 'LinHeat_schedule_with_capacity_bids.csv'
        # self.afrrfile = 'aFRRenergymarket_2025-04-30_to_2025-05-31.csv'
        # self.spotpricefile = 'spotprices_2025-05-01_to_2025-06-01.csv'
        # self.schedulefile = 'LinHeat_schedule_MAY_with_capacity_bids.csv'
        if type(start_date) == str:
            self.current = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
            self.start = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
        else:
            self.current = start_date
            self.start = start_date
        if type(end_date) == str:
            self.end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
        else:
            self.end = end_date
        self.print_flag = print_flag
        self.plot_flag = plot_flag
        self.output = []
        self.Einit = 53.87 # Start the simulation with 90% SoE
        self.recurssive_period = 0.25 # for 15m (the RTO is rerun every 15 mins)
        total_hours = (self.end - self.current).total_seconds() / 3600
        self.total_steps = int(total_hours / self.recurssive_period) + 1
        self.day_iterator = tqdm(total=self.total_steps, desc=f"Running RH Model", unit="step")

    def ARX_model(self, results):
        ARX_simulator = False
        if ARX_simulator == True:
            B = [-0.00091497,  0.14029288]
            A = [-1.04411331, 0.00942735,  0.00399007,  0.05331014]
            boiler_model = ARXSimulator(A, B)

            print(self.current)
            timestamps = [self.current - timedelta(hours=1) + timedelta(minutes=15 * i) for i in range(4)]
            y_lags = []
            for ts in timestamps:
                y = results.loc[results['SecondUTC'] == ts, 'System State (MWh)']
                if not y.empty:
                    y_lags.append(y.iloc[0])
                else:
                    y_lags.append(54)
            x_lags = []
            for ts in timestamps[-2:]:
                x = results.loc[results['SecondUTC'] == ts, 'Power Boiler (MW)']
                if not x.empty:
                    x_lags.append(x.iloc[0])
                else:
                    x_lags.append(12)
            timestamps = [self.current + timedelta(minutes=15 * i) for i in range(4)]

            schedule = results.loc[results['SecondUTC'].isin(timestamps), 'Power Boiler (MW)'].astype(int).tolist()
            mwh_trajectory = boiler_model.MW_to_MWh(schedule, y_lags, x_lags)
            # add or subtract any activation from final MWh
            total_cleared = results.loc[(results['SecondUTC'].dt.date == self.current.date()) & (results['SecondUTC'].dt.hour == self.current.hour), 'Energy Down Cleared (MWh)'].sum() \
                            - results.loc[(results['SecondUTC'].dt.date == self.current.date()) & (results['SecondUTC'].dt.hour == self.current.hour), 'Energy Up Cleared (MWh)'].sum()
            self.Einit = mwh_trajectory[-1] + total_cleared
        else:
            self.Einit = results.loc[results['SecondUTC'] == self.current + timedelta(hours=self.recurssive_period), 'Actual System SoE (%)'].iloc[0] * 60 # Emax
            # if the SoE deviates too far out of expected trajectory
            if self.Einit > 0.9*60:
                print(f'Einit at {self.current}: {self.Einit:.3f}MWh/ {self.Einit/60:.3f}%')
                print(f'Model reset to 90% SoE at {self.current}')
                self.Einit = 54 #reset
                # raise ValueError("Einit is outside acceptable range")
            elif self.Einit < 0.1*60:
                print(f'Einit at {self.current}: {self.Einit:.3f}MWh/ {self.Einit/60:.3f}%')
                print(f'Model reset to 10% SoE at {self.current}')
                self.Einit = 6 #reset

        self.current += timedelta(hours=self.recurssive_period) 
        if self.print_flag == True:
            print(f'Fetching Einit at {self.current}: {self.Einit:.3f}MWh/ {self.Einit/60:.3f}%')
        return self.Einit

    def save_results(self):
        final_results = pd.concat(self.output, ignore_index=True)
        final_results['MTU Revenue Row'] = (
            final_results['Actual Energy Up Cleared (MWh)'] * final_results['aFRR_UpActivatedPriceEUR_actual'] +
            final_results['Actual Energy Down Cleared (MWh)'] * -final_results['aFRR_DownActivatedPriceEUR_actual'])
        
        mtu_revenue_per_utc = final_results.groupby('UTC')['MTU Revenue Row'].sum()
        final_results['MTU Revenue'] = final_results['UTC'].map(mtu_revenue_per_utc)
        final_results.drop(columns=['MTU Revenue Row'], inplace=True) 
        return final_results
    
    def run_RTO_recursively(self, w = 0, horizon = 96):
        while self.current <= self.end:
            day = self.current.strftime('%Y-%m-%d %H:%M:%S')
            RTO = Realtime_Optimisation(day, self.afrrfile, self.spotpricefile, self.schedulefile, horizon= horizon)
            # if day == self.start:
            #     model, results = RTO.optmize(Einit = 54, W_factor = w, plot_flag= self.plot_flag)
            # else:
            model, results = RTO.optmize(Einit = self.Einit, W_factor = w, plot_flag= self.plot_flag)
            if model.status != 2:
                print(f'Model infeasible for {self.current} with w = {w}')
                feasible = False
                for new_w in range(w - 5, -1, -5):
                    model, results = RTO.optmize(Einit = self.Einit, W_factor = new_w, plot_flag= self.plot_flag)
                    if model.status == 2:
                        print(f'Model feasible for {self.current} with relaxed w = {new_w}')
                        feasible = True
                        break
                if not feasible:
                    print(f'No feasible solution: Skipping timestamp {self.current}')
                    # break 
                    self.current += timedelta(hours=self.recurssive_period)
                    self.day_iterator.update(1)

            if self.print_flag == True:
                print(f'Optimisation of day {self.current} completed ')
            if model.status == 2:
                self.output.append(results[(results['SecondUTC'] >= self.current) & (results['SecondUTC'] < self.current + timedelta(hours=self.recurssive_period))])
                self.Einit = self.ARX_model(results)
            self.day_iterator.update(1)
        
        final_results = self.save_results()
        self.day_iterator.close()
        return final_results








        # results['Updated Power (MW)'] = np.where(
        #     (results['bid_state'] == 1) & (results['activation'] == 1) & (results['cleared_status'] == 1),
        #     results['Power Boiler (MW)'] - results['Bid Up (MW)'],
        #     np.where(
        #         (results['bid_state'] == 1) & (results['activation'] == 0) & (results['cleared_status'] == 1),
        #         results['Power Boiler (MW)'] + results['Bid Down (MW)'],
        #         results['Power Boiler (MW)']
        #     )
        # )