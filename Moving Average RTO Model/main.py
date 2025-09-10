import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rt_op_modified import Realtime_Optimisation
from recurssive_RTO import RecursiveRTO
from OL_nonrecurssive_RTO import OpenLoopRTO
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm
import time

def run_RTO(start_date, end_date, w):
    afrrfile = 'aFRRenergymarket_2025-03-01_to_2025-03-31.csv'
    spotpricefile = 'spotprices_2025-03-01_to_2025-03-31.csv'
    schedulefile = 'LinHeat_schedule_with_capacity_bids.csv'
    current = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    # current = start_date
    # end = end_date
    total_days = (end - current).days + 1  # Include end date
    day_iterator = tqdm(total=total_days, desc=f"Running days for W={w}", unit="day")
    output = []
    while current <= end:
        day = current.strftime('%Y-%m-%d %H:%M:%S')
        RTO = Realtime_Optimisation(day, afrrfile, spotpricefile, schedulefile, horizon= 96)
        if day == start_date:
            model, results = RTO.optmize(Einit = 54, W_factor = w)
        else:
            model, results = RTO.optmize(Einit = 54, W_factor = w)
        if model.status != 2:
            print(f'Model infeasible for {current} for w: {w}')
            current += timedelta(days=1)
            day_iterator.update(1)
            continue
        print(f'Optimisation of day {current} completed for w: {w}')

        current += timedelta(days=1) # recurssive every 1 day
        day_iterator.update(1)
        row = results[['Deviation','Total Revenue', 'Actual Realised Revenue','UTC']].loc[1]
        row['W_factor'] = w
        output.append(row)
    day_iterator.close()
    return output

def plot_results(results):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

    # Add traces
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_UpActivatedPriceEUR'], mode='lines', name='MA UP price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_UpActivatedPriceEUR_actual'], mode='lines', name='aFRR_UpActivatedPriceEUR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['SpotPriceEUR'], mode='lines', name='SpotPriceEUR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_DownActivatedPriceEUR'], mode='lines', name='MA Down Price'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['aFRR_DownActivatedPriceEUR_actual'], mode='lines', name='aFRR_DownActivatedPriceEUR'), row=1, col=1)

    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Actual Energy Up Cleared (MWh)'], mode='lines+markers', name='Energy Up Cleared (MWh)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Actual Energy Down Cleared (MWh)'], mode='lines+markers', name='Energy Down Cleared (MWh)'), row=2, col=1)
    fig.add_shape(type="line",x0=results['SecondUTC'].min(), x1=results['SecondUTC'].max(), y0=0.9, y1=0.9,line=dict(color="black", width=1),xref="x3",yref="y3")
    fig.add_shape(type="line",x0=results['SecondUTC'].min(), x1=results['SecondUTC'].max(), y0=0.1, y1=0.1,line=dict(color="black", width=1),xref="x3",yref="y3")

    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['System SoE (%)'], mode='lines', name='System SoE (%)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Actual System SoE (%)'], mode='lines', name='Actual System SoE (%)'), row=3, col=1)

    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['bid_state'], mode='lines', name='bid_state'), row=3, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['activation'], mode='lines', name='activation'), row=3, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['actual_bid_outcome'], mode='lines', name='cleared_status'), row=3, col=1)

    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Bid Up (MW)'], mode='lines', name='Bid Up (MW)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Bid Down (MW)'], mode='lines', name='Bid Down (MW)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Power Boiler (MW)'], mode='lines', name='Power Boiler (MW)'), row=4, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['L11 EK plan [MW]'], mode='lines', name='L11 EK plan [MW]'), row=4, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Capacity Bid'], mode='lines', name='Capacity Bid'), row=4, col=1)
    fig.add_trace(go.Scatter(x=results['SecondUTC'], y=results['Heat load forecast [MW]'], mode='lines', name='Heat load forecast [MW]'), row=4, col=1)

    # Update layout with size and borders
    fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Cleared (MWh)',yaxis3_title='System state (%)', yaxis4_title='Bid MW',  
    legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

    for i in range(1, 5):
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

    fig.show()


if __name__ == '__main__': 
    folder_path = "D:\\ms\\January 2024\\Thesis\\Boiler-Bidding-model\\Simulation Output\\"
    select_case = 'base' # base / W_factor / prediction_horizon
    start_date = '2025-03-02 00:00:00' #YYYY-MM-DD
    end_date = '2025-03-16 23:45:00' #YYYY-MM-DD
    
    if select_case == 'base':
        # to run the model for a specific date
        w = 10
        horizon = 24
        recurrsive = True
        plot = False
        if recurrsive == False:
            RTO = OpenLoopRTO(start_date, end_date, print_flag= True, plot_flag= False)
            results = RTO.run_RTO(w = w, horizon= horizon)
        else:
            RTO = RecursiveRTO(start_date, end_date, print_flag= True, plot_flag= False)
            results = RTO.run_RTO_recursively(w = w, horizon= horizon)

        if plot == True:
            plot_results(results)
        results.to_csv(folder_path + f'recursive_March_2_to_16_w10_6h_MA_6h_window_all_constraints.csv', index=False)

    if select_case == 'W_factor':
        # to run the model for different W_factors
        W_factors = np.concatenate([np.array([0, 2, 4, 5, 6, 8, 10]), np.arange(15, 51, 5)]) #np.arange(0, 50, 5)
        output = []
        for w in tqdm(W_factors):
            results = run_RTO(start_date, end_date, w)
            plot = False
            if plot == True:
                plot_results(results)
            output.extend(results)
        W_revenue = pd.DataFrame(output)
        # W_revenue.to_csv('deviation_revenue_vs_w_factor_DA_1.csv', index = False)

    if select_case == 'prediction_horizon':
        # to run the model for different horizon
        # horizon_list = np.sort(np.append(np.arange(8, 97, 8), [4]))
        horizon_list = np.arange(80, 97, 8)
        output = []
        for h in tqdm(horizon_list):
            start_timer = time.time() 
            RTO = RecursiveRTO(start_date, end_date, print_flag= True, plot_flag= False)
            df = RTO.run_RTO_recursively(w = 20, horizon= h)
            computation_time = time.time() - start_timer
            df['direction'] = np.where((df['bid_state'] == 0), 0, np.where((df['bid_state'] == 1) & (df['activation'] == 1), 1, -1))
            up_eur = df.loc[df['direction'] == 1, 'MTU Revenue'].unique().sum()
            down_eur = df.loc[df['direction'] == -1, 'MTU Revenue'].unique().sum()
            total_eur = df['MTU Revenue'].unique().sum()
            df1 = df[['UTC', 'Power Boiler (MW)', 'L11 EK plan [MW]']].drop_duplicates(subset=['UTC'], keep='first', ignore_index=True)
            df1['Deviation'] = ((df1['Power Boiler (MW)'] - np.ceil(df1['L11 EK plan [MW]'])) ** 2)
            deviation = df1['Deviation'].sum()
            row = {'horizon': h, 'up_eur': up_eur, 'down_eur': down_eur, 'total_eur': total_eur, 'deviation': deviation, 'computation_time_s': computation_time}
            output.append(row)
        horizon_revenue = pd.DataFrame(output)
        # horizon_revenue.to_csv(f'horizon_revenue_05_03_recurrsive15m_W{w}.csv', index=False)


# forgot what this is :(
    # to run the model openloop and closedloop
    # output = []
    # start_date = '2025-03-01 00:00:00' #YYYY-MM-DD    
    # end_date = '2025-03-02 00:00:00' #YYYY-MM-DD
    # w = 50
    # horizon = 96
    # start = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S')
    # end = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    # # while start <= end:
    # end_in_1day = start + timedelta(days=1) - timedelta(minutes=15)
    # w = 50
    
    # OL_RTO = OpenLoopRTO(start, end_in_1day, print_flag= True, plot_flag= False)
    # OL_results = OL_RTO.run_RTO(w = w, horizon= horizon)
    # row1 = OL_results[['Deviation','Total Revenue','UTC']].loc[1]
    # print(f'Open Loop optimisation for {start} completed')
    # plot_results(OL_results)
    # #  output.append(row1)
    # RTO = RecursiveRTO(start, end_in_1day, print_flag= True, plot_flag= False)
    # CL_results = RTO.run_RTO_recursively(w = w, horizon= 96)
    # print(f'Recursive CL optimisation for {start} completed')
    # plot_results(CL_results)

    # total_eur = CL_results['MTU Revenue'].unique().sum()
    # row2 = {'date': start, 'total_eur_recurrsive': total_eur}
    # output.append(row2)
    # start += timedelta(days=1)