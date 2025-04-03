from multiple_MTU_opt_modified import load_values
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == '__main__':
    horizon = 8
    activation_rate = 0.2
    start = 1 #MTU to start with
    folder_path = "D:\\ms\\January 2024\\Thesis\\Boiler-Bidding-model\\Datasets\\"

    # load the data from csv files
    aFRR_UP_prices = pd.read_csv(folder_path + 'Up_Price_Forecast_backtest.csv')
    spotprices = pd.read_csv(folder_path + 'spotprice_2025-03-25_to_2025-03-30.csv')
    aFRR_DOWN_prices = pd.read_csv(folder_path + 'Down_Price_Forecast_backtest.csv')
    aFRR_UP_prices['time'] = pd.to_datetime(aFRR_UP_prices['time'])
    aFRR_DOWN_prices['time'] = pd.to_datetime(aFRR_DOWN_prices['time'])
    spotprices['time'] = pd.to_datetime(spotprices['HourUTC'])
    aFRR_UP_prices = aFRR_UP_prices.set_index('time').resample('4s').ffill()
    aFRR_DOWN_prices = aFRR_DOWN_prices.set_index('time').resample('4s').ffill()
    spotprices = spotprices.set_index('time').resample('4s').ffill()

    temp = pd.merge(aFRR_UP_prices, aFRR_DOWN_prices, how='left', left_on='time', right_on='time')
    df = pd.merge(temp, spotprices['SpotPriceEUR'], how='left', left_on='time', right_on='time')
    df.reset_index(inplace=True)
    df['activation_rate'] = activation_rate
    df['UpCapacityBid'] = 0
    df['DownCapacityBid'] = 0

    md, result_dict, results = load_values(df, start = 1801 * start, horizon = horizon)

    results['Bid (MW)'] = results.apply(lambda row: row['Bid Up (MW)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Bid Down (MW)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
    results['Energy Cleared (MWh)'] = results.apply(lambda row: row['Energy Up Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Energy Down Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
    results['Total Bid (Eur/MWh)'] = results.apply(lambda row: row['Pbid Up (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else row['Pbid Down (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)

    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.01,row_heights=[0.25, 0.25, 0.25, 0.25] )

    # Add traces
    fig.add_trace(go.Scatter(x=results['time'], y=results['aFRR_UpActivatedPriceEUR'], mode='lines', name='aFRR_UpActivatedPriceEUR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['time'], y=results['aFRR_UP_Price_prediction'], mode='lines', name='aFRR_UP_Price_prediction'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['time'], y=results['SpotPriceEUR'], mode='lines', name='SpotPriceEUR'), row=1, col=1)
    fig.add_trace(go.Scatter(x=results['time'], y=results['Total Bid (Eur/MWh)'], mode='lines', name='Total Bid (Eur/MWh)'), row=1, col=1)

    fig.add_trace(go.Scatter(x=results['time'], y=results['Energy Cleared (MWh)'], mode='lines', name='Energy Cleared (MWh)'), row=2, col=1)
    fig.add_trace(go.Scatter(x=results['time'], y=results['System state (MWh)'], mode='lines', name='System state (MWh)'), row=3, col=1)
    fig.add_trace(go.Scatter(x=results['time'], y=results['Bid (MW)'], mode='lines', name='Total bid (MW)'), row=4, col=1)

    # Update layout with size and borders
    fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Price (EUR/MWh)', yaxis2_title='Energy Cleared (MWh)',yaxis3_title='System state (MWh)', yaxis4_title='Bid MW',  
    legend_title='Legend',template='plotly_white',height=800,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

    for i in range(1, 5):
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

    fig.show()

    # check forecast objective value to Perfect_information objective value
    output = pd.DataFrame()
    indices = list(range(0, 10000, 1800+1)) #1800 4s intervals in 1 MTU
    for start in indices:
        df1 = df.copy()
        md, result_dict, results = load_values(df, start = start, horizon = horizon)

        results['Bid (MW)'] = results.apply(lambda row: row['Bid Up (MW)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Bid Down (MW)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
        results['Energy Cleared (MWh)'] = results.apply(lambda row: row['Energy Up Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else -row['Energy Down Cleared (MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)
        results['Total Bid (Eur/MWh)'] = results.apply(lambda row: row['Pbid Up (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 1 else row['Pbid Down (Eur/MWh)'] if row['Activation 1 - Up 0 - Down'] == 0 else 0, axis=1)

        output = pd.concat([output, results[['time', 'Bid (MW)', 'Objective Value', 'Perfect Obj value']]], ignore_index=True)
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.01)

    # Add traces
    fig.add_trace(go.Scatter(x=output['time'], y=output['Objective Value'], mode='lines', name='Forecasted Objective value'), row=1, col=1)
    fig.add_trace(go.Scatter(x=output['time'], y=output['Perfect Obj value'], mode='lines', name='Perfect Infomation Obj val.'), row=1, col=1)

    fig.add_trace(go.Scatter(x=output['time'], y=output['Bid (MW)'], mode='lines', name='Bid MW'), row=2, col=1)

    # Update layout with size and borders
    fig.update_layout(title='Output of OP',xaxis_title='Time', yaxis1_title='Objective value (EUR)', yaxis2_title='Bid MW',  
    legend_title='Legend',template='plotly_white',height=400,  margin=dict(t=60, b=60, l=60, r=60),  showlegend=True)

    for i in range(1, 5):
        fig.update_yaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)
        fig.update_xaxes(showline=True, linewidth=2, linecolor='black', row=i, col=1)

    fig.show()