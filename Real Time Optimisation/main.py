import numpy as np
import matplotlib.pyplot as plt
from rt_op_modified import Realtime_Optimisation
from datetime import datetime, timedelta

def run_RTO(start_date, end_date):
    afrrfile = 'aFRRenergymarket_2025-04-01_to_2025-04-20.csv'
    spotpricefile = 'spotprices_2025-04-01_to_2025-04-20.csv'
    schedulefile = 'linheat_schedule_2025-04-01_to_2025-04-20.csv'

    current = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    while current <= end:
        day = current.strftime('%Y-%m-%d')
        RTO = Realtime_Optimisation(day, afrrfile, spotpricefile, schedulefile)
        if day == start_date:
            model, results = RTO.optmize(Einit = 30)
        else:
            model, results = RTO.optmize(Einit)
        Einit = results['System State (MWh)'].iloc[-1]
        current += timedelta(days=1)


if __name__ == '__main__': 
    start_date = '2025-04-01' #YYYY-MM-DD
    end_date = '2025-04-08' #YYYY-MM-DD
    run_RTO(start_date, end_date)

