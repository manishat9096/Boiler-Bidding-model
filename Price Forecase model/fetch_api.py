import time
from datetime import datetime, timedelta
import csv
import pandas as pd
import numpy as np
import random as rd
from matplotlib.dates import DateFormatter
import requests
import xml.etree.ElementTree as ET
import pytz

def generate_date_range(start, end):
    """
    Issue with ENTSO-E data: data for the last minute of the day is missing (between 23:59 and 00:00)
    """
    
    end = (datetime.strptime(end, '%Y%m%d') + pd.Timedelta(days=1)).strftime('%Y%m%d')
    dates0 = pd.date_range(start=start, end = end, freq = '4S', inclusive = 'left')
    dates = [(d - pd.Timedelta(minutes=1) if d.hour == 23 and d.minute == 59 else d) for d in dates0]
    dates = pd.DatetimeIndex(dates).drop_duplicates()
    return dates

def get_entso_response_root(start, end, key_entso):
    response = requests.get(url = "https://web-api.tp.entsoe.eu/api?securityToken=" + key_entso + "&documentType=A84&processtype=A67&businessType=A96&controlArea_Domain=10YDK-1--------W&periodStart="+start+"0000&periodEnd="+end+"2359")
    response.raise_for_status()
    root = ET.fromstring(response.text)
    return root

def process_root(start, end, key_entso, frequency):
    root = get_entso_response_root(start, end, key_entso)
    dates = generate_date_range(start, end)

    data_up = []
    data_down = []
    # Loop through TimeSeries elements
    for ts in root.findall(".//{urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1}TimeSeries"):
        mRID = ts.find(".//{urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1}mRID").text

        # Extract Points within each TimeSeries
        for point in ts.findall(".//{urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1}Point"):
            position = int(point.find(".//{urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1}position").text)
            if point.find(".//{urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1}activation_Price.amount") == None:
                price=np.nan
            else:
                price = float(point.find(".//{urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:1}activation_Price.amount").text)

            # Append data based on mRID (1 for up, 2 for down)
            if mRID == '1':
                data_up.append({'position': position, 'activation_Price_up': price})
            elif mRID == '2':
                data_down.append({'position': position, 'activation_Price_down': price})
    
    # Convert lists to DataFrames
    df_up = pd.DataFrame(data_up).set_index('position')
    df_down = pd.DataFrame(data_down).set_index('position')

    # Merge the DataFrames
    df = pd.merge(df_up, df_down, how = 'outer', left_index = True, right_index = True)
    index_all = range(int(int(frequency[:-1])*(24*60*60-60)/4)) ## if the data is for one week 
    df = df.reindex(index_all).ffill()
    df.index= dates
    df['MTU'] = df.index.floor('15T')
    df['MTU_time'] = df['MTU'].apply(lambda x: x.replace(year=2024, month=1, day=1))
    return df

def merge_all(start_all, end_all, key_entso, frequency):
    N_weeks = len(start_all)
    df = pd.DataFrame()
    for i in range(N_weeks):
        start = start_all[i].strftime('%Y%m%d')
        end = end_all[i].strftime('%Y%m%d')
        df_new = process_root(start, end, key_entso, frequency)
        df = pd.concat([df, df_new], axis = 0)
    return df

def entsoe_api(start_date, frequency, period):
    today = datetime.today()
    today = today.replace(hour=0, minute=0, second=0, microsecond=0)
    print('Today:', today.date())

    # start_date = '20241018'
    # dt = '7D'
    start_all = pd.date_range(start=start_date, freq=frequency, periods=period) # periods = number of weeks to analyse
    # end_all = start_all + pd.Timedelta(days=6)
    if frequency.endswith('D'):  # Daily frequency
        duration = pd.Timedelta(days=int(frequency[:-1]) - 1)
    elif frequency.endswith('H'):  # Hourly frequency
        duration = pd.Timedelta(hours=int(frequency[:-1]) - 1)
    elif frequency.endswith('W'):  # Weekly frequency
        duration = pd.Timedelta(weeks=int(frequency[:-1]) - 1)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")

    end_all = start_all + duration

    print('Market data from:', start_all[0].date(), 'to', end_all[-1].date())

    key_entso = '077d7e61-14ac-4a16-9fa6-f0be98f08a8e'

    df = merge_all(start_all, end_all, key_entso, frequency)
    return df

# def energinet_api():
#     p = {
#             "limit": 0,
#             "start": datetime(year=2024, month = 11, day = 5).strftime("%Y-%m-%d"),
#             "end": datetime(year=2024, month = 11, day = 8).strftime("%Y-%m-%d")
#             # "end": datetime.now().strftime("%Y-%m-%d")
#         }
#     response = requests.get(url="https://api.energidataservice.dk/dataset/PowerSystemRightNow", params=p)

#     result = response.json()

#     for k, v in result.items():
#         print(k, v)

#     records = result.get('records', [])
#     time_utc = 'HourUTC'
#     imbalance_prices = pd.DataFrame(records)
#     imbalance_prices = imbalance_prices[[ 'Minutes1UTC', 'Minutes1DK', 'aFRR_ActivatedDK1', 'aFRR_ActivatedDK2','ImbalanceDK1', 'ImbalanceDK2']]


#     import requests



def energinet_api(dataset, start_date_utc, end_date_utc = None):
    if end_date_utc is None:
        end_date_utc = datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    utc_tz = pytz.UTC
    local_tz = pytz.timezone('CET')

    start_dt_utc = utc_tz.localize(datetime.strptime(start_date_utc, "%Y-%m-%d"))
    end_dt_utc = utc_tz.localize(datetime.strptime(end_date_utc, "%Y-%m-%d"))

    start_local = start_dt_utc.astimezone(local_tz)
    end_local = end_dt_utc.astimezone(local_tz)

    params = {
        "limit": 0,  # Fetch all available records
        "start": start_local.strftime("%Y-%m-%d"),
        "end": end_local.strftime("%Y-%m-%d")
    }

    url = "https://api.energidataservice.dk/dataset/"+dataset
    response = requests.get(url=url, params=params)
    response.raise_for_status()  # Raise an error for HTTP issues
    result = response.json()

    # Extract records
    records = result.get('records', [])
    if not records:
        print("No data found for the specified period.")
        return pd.DataFrame()
    df = pd.DataFrame(records)
    return df
