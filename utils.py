import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import pickle
import copy
import os

def sum_country_regions(df, country):
    # Get indices of the countries
    country_indices = df.loc[df['Country/Region'] == country].index
    columns = df.columns

    has_colony = pd.isna(df.loc[country_indices]).any(axis=1)
    if has_colony.any():
        # Case 1: imperialist nations
        df.drop(has_colony[-has_colony].index, inplace=True)
        return df
    else:
        # Case 2: federal country
        # Get the summation data of the countries
        # and replace the first region with the result 

        df.loc[country_indices[0], 4:] = df.loc[country_indices].sum(axis=0)[4:]
        # Remove the rest regions
        df.drop(country_indices[1:], inplace=True)

        return df
    
def get_change_rates(series, days):
    series_change_rate = copy.deepcopy(series)
    for day_index in range(len(series)):
        min_day = day_index - days if day_index - days >= 0 else 0
        change_rate = 1 + (series[day_index] - series[min_day]) / series[min_day] if int(series[min_day]) is not 0 else 1
        series_change_rate[day_index] = change_rate
        if change_rate >99:
            print(change_rate)
        
    return series_change_rate
def generate_df_change_rate(df, days):
    df_change_rate = df.copy(deep=True)
    columns = df.columns
    for country in range(len(df)):
        df_change_rate.loc[country, columns[4:]] = get_change_rates(df.loc[country, columns[4:]], days)# .apply(lambda x: get_change_rates(x, days))

    return df_change_rate

def sliding_window(data, seq_length, normalize=True):
    x=[]
    y=[]
    for i in range(len(data) - seq_length):
        x_ = data[i:(i + seq_length)]
        y_ = data[i+seq_length]
        if normalize:
            x_ = [float(_)/y_ for _ in x_]
            y_ = y_/float(y_) if y_ is not 0 else 0
        x.append(x_)
        y.append(y_)

    return x, y

def generate_sequence(df_confirm, df_death, selected_index=None, selected_country=None, normalize=False, seq_length=14):
    # Check whether two data have the same number of countries
    assert len(df_confirm) == len(df_death)
    
    seq_confirm_x, seq_confirm_y = [], []
    seq_death_x, seq_death_y = [], []
    
    
    # Generate sequence data
    # the actual data starts from column index 4
    for country in range(len(df_confirm)):
        try: df_confirm.loc[country, 'Country/Region'] 
        except Exception as e: continue
        if df_confirm.loc[country, 'Country/Region'] not in selected_country:
            continue
        seq_confirm_x_tmp, seq_confirm_y_tmp = sliding_window(df_confirm.loc[country][selected_index[country]:], seq_length, normalize)
        seq_death_x_tmp, seq_death_y_tmp = sliding_window(df_death.loc[country][selected_index[country]:], seq_length, normalize)
        
        seq_confirm_x.extend(seq_confirm_x_tmp)
        seq_confirm_y.extend(seq_confirm_y_tmp)
        
        seq_death_x.extend(seq_death_x_tmp)
        seq_death_y.extend(seq_death_y_tmp)
    
    
     
    seq_death_x = np.expand_dims(np.array(seq_death_x, dtype=np.float), axis=2)
    seq_death_y = np.expand_dims(np.array(seq_death_y, dtype=np.float), axis=2)
    seq_confirm_x = np.expand_dims(np.array(seq_confirm_x, dtype=np.float), axis=2)
    seq_confirm_y = np.expand_dims(np.array(seq_confirm_y, dtype=np.float), axis=2)
    
    print(seq_death_x.shape)
    print(seq_death_y.shape)
    print(seq_confirm_x.shape)
    print(seq_confirm_y.shape)
    
    # Merge data to make 2-dim sequence data
    X = np.concatenate((seq_confirm_x, seq_death_x), 2)
    Y = np.concatenate((seq_confirm_y, seq_death_y), 1)

    print(X.shape)
    print(Y.shape)
    return X, Y


def generate_COVID_input(df_death,
                        df_death_change_1st_order,
                        df_death_change_2nd_order,
                        df_confirm,
                        df_confirm_change_1st_order,
                        df_confirm_change_2nd_order,
                        df_incoming,
                        google_trend1,google_trend2,google_trend3,google_trend4,
                        countries_Korea_inbound,
                        seq_length=14,
                        end_day = None,
                        is_7days = False,
                        scaler_list=None):
    
    # Filter out unrelated countries & columns from data
    selected_country = np.array(countries_Korea_inbound.loc[countries_Korea_inbound.visit.eq(1), 'Country'])
    selected_country = np.delete(selected_country, np.argwhere(selected_country == 'Korea, South'))
    
    
    columns = df_death.columns
    
    select = []
    for country in range(len(df_death)):
        select.append(df_death.loc[country, 'Country/Region'] in selected_country) # Indices where the selected countries are

    
    s_ = np.where(google_trend1.columns == df_incoming.loc[0, 'date'])[0][0]
    
    if end_day is None:
        e_ = np.where(google_trend1.columns == df_incoming.loc[len(df_incoming)-1, 'date'])[0][0]
    else :
        e_ = np.where(google_trend1.columns == end_day)[0][0]
        
    start_day = np.where(columns == df_incoming.loc[0, 'date'])[0][0]
    if end_day is None:
        end_day = np.where(columns == df_incoming.loc[len(df_incoming)-1, 'date'])[0][0]
    else :
        end_day = np.where(columns == end_day)[0][0]
#     print(start_day,end_day)
    selected_country = np.array(df_death.loc[select, 'Country/Region'])
    
    d1 = df_death.loc[select, columns[start_day:end_day+1]]
    d2 = df_death_change_1st_order.loc[select, columns[start_day:end_day+1]]
    d3 = df_death_change_2nd_order.loc[select, columns[start_day:end_day+1]]
    d4 = df_confirm.loc[select, columns[start_day:end_day+1]]
    d5 = df_confirm_change_1st_order.loc[select, columns[start_day:end_day+1]]
    d6 = df_confirm_change_2nd_order.loc[select, columns[start_day:end_day+1]]
    
    d7 = google_trend1.loc[:, google_trend1.columns[s_:e_+1]]
    d8 = google_trend2.loc[:, google_trend2.columns[s_:e_+1]]
    d9 = google_trend3.loc[:, google_trend3.columns[s_:e_+1]]
    d10 = google_trend4.loc[:, google_trend4.columns[s_:e_+1]]
    
    d1 = np.log(d1+1)
    d4 = np.log(d4+1)
    
    #scaling
    mean, std = scaler_list[0]
    d4 = d4.subtract(mean[select], axis='index').divide(std[select], axis='index')

    mean, std = scaler_list[1]
    d1 = d1.subtract(mean[select], axis='index').divide(std[select], axis='index')
    
    mean, std = scaler_list[2]
    d5 = d5.subtract(mean[select], axis='index').divide(std[select], axis='index')
    
    mean, std = scaler_list[3]
    d6 = d6.subtract(mean[select], axis='index').divide(std[select], axis='index')
    
    mean, std = scaler_list[4]
    d2 = d2.subtract(mean[select], axis='index').divide(std[select], axis='index')
    
    mean, std = scaler_list[5]
    d3 = d3.subtract(mean[select], axis='index').divide(std[select], axis='index')
    ##
    
    # Clean up the refined data

    d1.reset_index(inplace=True, drop=True)
    d2.reset_index(inplace=True, drop=True)
    d3.reset_index(inplace=True, drop=True)
    d4.reset_index(inplace=True, drop=True)
    d5.reset_index(inplace=True, drop=True)
    d6.reset_index(inplace=True, drop=True)
    d7.reset_index(inplace=True, drop=True)
    d8.reset_index(inplace=True, drop=True)
    d9.reset_index(inplace=True, drop=True)
    d10.reset_index(inplace=True, drop=True)
    
    columns = d1.columns
    
    
    # Generate country-wise 6-dim vectors of 14 days
    data = []
    for day in range(end_day - start_day+1 - seq_length):
        country_dict = {}
        for idx, country in enumerate(selected_country):
            _ = np.concatenate((
                np.expand_dims(np.array(d1.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d2.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d3.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d4.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d5.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d6.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d7.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d8.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d9.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1),
                np.expand_dims(np.array(d10.loc[idx, columns[day:(day+seq_length)]], dtype=np.float), axis=1)), axis=-1)
    
            country_dict[country] = _
                
        data.append(country_dict)        
    
    
    target_continent = np.array(df_incoming.loc[:, countries_Korea_inbound.continent.unique()])
    target_total = np.array(df_incoming.loc[:, 'sum'])
    
    if end_day is not None:
        # target_continent : (n,6) -> (N:39, T:14,D:6); target_total : (n,1) -> (N:39, T:14,D:1)
        target_continent = [target_continent[day+seq_length:day+seq_length+seq_length//2,:] if is_7days else target_continent[day+seq_length:day+seq_length*2,:] for day in range(end_day - start_day+1 - seq_length)]
        target_total = [target_total[day+seq_length:day+seq_length+seq_length//2,] if is_7days else target_total[day+seq_length:day+seq_length*2,] for day in range(end_day - start_day+1 - seq_length)]
        
        print()
        target_continent = np.stack(target_continent,axis=0)
        target_total = np.stack(target_total,axis=0)
        return data, target_continent, target_total
        
    else :
        return data, target_continent[-(end_day - start_day+1 - seq_length):], target_total[-(end_day - start_day+1 - seq_length):]

    
    
def generate_COVID_aux_input(df_roaming,
                            df_infection_ratio,
                            passenger_flights,
                            df_incoming):
    
    columns = df_roaming.columns
    
    start_day_roaming = np.where(df_roaming.columns == df_incoming.loc[0, 'date'])[0][0] +13
    start_day_infection = np.where(df_infection_ratio.columns == df_incoming.loc[0, 'date'])[0][0] +13
    start_day_passenger = np.where(passenger_flights.columns == df_incoming.loc[0, 'date'])[0][0] +13

    end_day = np.where(columns == '5/5/20')[0][0] 

    data = []
    for day in range(end_day - start_day_roaming + 1):
        country_dict = {}
        for idx, country in enumerate(df_roaming.Country):
            _ = np.concatenate((
                np.expand_dims(np.array(df_roaming.loc[idx, df_roaming.columns[start_day_roaming + day]], dtype=np.float), axis=1),
                np.expand_dims(np.array(df_infection_ratio.loc[idx, df_infection_ratio.columns[start_day_infection + day]], dtype=np.float), axis=1),
                np.expand_dims(np.array(passenger_flights.loc[idx, passenger_flights.columns[start_day_passenger + day]], dtype=np.float), axis=1),
            ),
                axis=-1)
    
            country_dict[country] = _
                
        data.append(country_dict)        
    
    return data