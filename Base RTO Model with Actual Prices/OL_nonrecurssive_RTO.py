import numpy as np
import pandas as pd
from rt_op_modified import Realtime_Optimisation
from datetime import datetime, timedelta


class OpenLoopRTO():
    def __init__(self, start_date, end_date, print_flag = False, plot_flag = False):
        self.afrrfile = 'aFRRenergymarket_2025-03-01_to_2025-03-31.csv'
        self.spotpricefile = 'spotprices_2025-03-01_to_2025-03-31.csv'
        self.schedulefile = 'LinHeat_schedule_with_capacity_bids.csv'
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
        self.Einit = 54 # Start the simulation with 90% SoE
    
    def run_RTO(self, w = 0, horizon = 96):
        while self.current <= self.end:
            day = self.current.strftime('%Y-%m-%d %H:%M:%S')
            RTO = Realtime_Optimisation(day, self.afrrfile, self.spotpricefile, self.schedulefile, horizon= horizon)
            if day == self.start:
                model, results = RTO.optmize(Einit = 54, W_factor = w, plot_flag= self.plot_flag)
            else:
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
                    print(f'No feasible solution at {self.current}')
                    break 
            self.output.append(results)
            self.Einit = results['System SoE (%)'].iloc[-1] * 60
            if self.print_flag == True:
                print(f'Optimisation of day {self.current} completed with Einit: {self.Einit}')
            self.current += timedelta(days=1)
        final_results = pd.concat(self.output, ignore_index=True)
        return final_results
