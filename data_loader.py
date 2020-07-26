import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from utils import sum_country_regions, get_change_rates, generate_df_change_rate, sliding_window, generate_COVID_input, generate_COVID_aux_input

import pickle
import copy
import os
import argparse


parser = argparse.ArgumentParser(description='Hi-covidnet DATALOADER')
    
# basic settings
parser.add_argument('--output_size', type=int, default=14, metavar='O', help='How many days you are predicting(default: 14)')
parser.add_argument('--save', action='store_true', default=False, help='Saving pre-processed data')

def normalize(df, axis=1):
    """
    @df : shape(N,D)
    """
    mean = df.iloc[:,4:].mean(axis=axis) # (D)
    std = df.iloc[:,4:].std(axis=axis) # (D)
    df.iloc[:,4:] = (df.iloc[:,4:].subtract(mean, axis='index')).divide(std, axis='index')
    return df, mean, std

def scaling(df_confirm,df_death, df_confirm_change_1st_order,
           df_confirm_change_2nd_order,df_death_change_1st_order,
           df_death_change_2nd_order, fname="x_mean_std_list_5_27.pkl"):
    ##scaling
    mean_std_list = []

    df_confirm, mean, std = normalize(df_confirm, axis=1)
    mean_std_list.append((mean,std))

    df_death, mean, std = normalize(df_death, axis=1)
    mean_std_list.append((mean,std))

    df_confirm_change_1st_order, mean, std = normalize(df_confirm_change_1st_order, axis=1)
    mean_std_list.append((mean,std))

    df_confirm_change_2nd_order, mean, std = normalize(df_confirm_change_2nd_order, axis=1)
    mean_std_list.append((mean,std))

    df_death_change_1st_order, mean, std = normalize(df_death_change_1st_order, axis=1)
    mean_std_list.append((mean,std))

    df_death_change_2nd_order, mean, std = normalize(df_death_change_2nd_order, axis=1)
    mean_std_list.append((mean,std))
    
    pickle.dump(mean_std_list, open("pickled_ds/"+fname, "wb"))

def google_trenddata_loader(fname, countries_Korea_inbound):
    google_trend = pd.read_csv('./dataset/{fname}.csv'.format(fname=fname), index_col=0)

    iso_to_country = countries_Korea_inbound.set_index('iso').to_dict()['Country']
    google_trend.rename(columns = iso_to_country, inplace = True)
    google_trend = google_trend.set_index('date').T.reset_index()
    google_trend = google_trend.rename(columns = {'index': 'Country'})
    google_trend.columns = google_trend.columns.astype(str)
    google_trend = google_trend.rename(columns = {col: str(int(col[4:6]))+'/'+str(int(col[-2:]))+'/' + col[2:4] for col in google_trend.columns[1:].astype(str)})
    google_trend.loc[:, google_trend.columns[1:]] /= 100

    google_trend.drop(np.argwhere(google_trend.Country == 'Korea, South')[0], inplace=True)

    mean, std = google_trend.iloc[:,1:].mean(axis=1), google_trend.iloc[:,1:].std(axis=1)
    google_trend.iloc[:,1:] = google_trend.iloc[:,1:].subtract(mean, axis='index').divide(std, axis='index')
    return google_trend

def dataloader(output_size, save=False):
    url_confirm = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    df_confirm = pd.read_csv(url_confirm, error_bad_lines=False)

    url_death = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    df_death = pd.read_csv(url_death, error_bad_lines=False)

    countries_with_regions = df_confirm['Country/Region'].value_counts()[df_confirm['Country/Region'].value_counts()>1].index
    for country in countries_with_regions:
        df_confirm = sum_country_regions(df_confirm, country)
        df_death = sum_country_regions(df_death, country)

    df_confirm.reset_index(inplace=True, drop=True)
    df_death.reset_index(inplace=True, drop=True)

    # Get the index of the 'last zero patients' from the countries
    selected_index = {}
    for country in range(len(df_confirm)):
        try: df_confirm.loc[country] # No country due to merged
        except Exception as e: continue 
        try:
            selected_index[country] = list(filter(lambda i: df_confirm.loc[country].eq(0)[i], range(len(df_confirm.columns))))[-1]
        except Exception as e:
            selected_index[country] = 4


    countries_Korea_inbound = pd.read_csv("./dataset/country_info.csv")
    
    countries_Korea_inbound.loc[countries_Korea_inbound['Country'] == 'China', 'continent'] = 'China'
    selected_country = countries_Korea_inbound.loc[countries_Korea_inbound.visit.eq(1), 'Country'].values

    df_confirm_change_1st_order = generate_df_change_rate(df_confirm, days=1)
    df_confirm_change_2nd_order = generate_df_change_rate(df_confirm_change_1st_order, days=1)
    df_death_change_1st_order = generate_df_change_rate(df_death, days=1)
    df_death_change_2nd_order = generate_df_change_rate(df_death_change_1st_order, days=1)
    
    scaling(df_confirm,df_death, df_confirm_change_1st_order,
           df_confirm_change_2nd_order,df_death_change_1st_order,
           df_death_change_2nd_order)
    
    fnames = ["trend_covid-19", "trend_covid_test", "trend_flu", "trend_mask"]
    google_data = [google_trenddata_loader(fname, countries_Korea_inbound) for fname in fnames]
    
    df_incoming = pd.read_csv('./dataset/confirmed_by_continent.csv')
    scaler_list = pickle.load(open("pickled_ds/x_mean_std_list_5_27.pkl", "rb"))
    
    data_model2, target_continent, target_total = generate_COVID_input(df_death,
                        df_death_change_1st_order,
                        df_death_change_2nd_order,
                        df_death_change_1st_order,
                        df_confirm_change_1st_order,
                        df_confirm_change_2nd_order,
                        df_incoming,
                        *google_data,
                        countries_Korea_inbound,
                        seq_length=14,
                        end_day='5/6/20', # '5/5/20'
                        is_7days= True if output_size==7 else False,
                        scaler_list=scaler_list)

    print("data shape is ",len(data_model2), data_model2[30]['Argentina'].shape)
    print("target_continent shape is ",target_continent.shape, "target_total shape is ",target_total.shape,)
    
    print("Loading KT roaming data")
    roaming = pd.read_csv('./dataset/roaming_preprocess.csv')
    df_roaming = pd.DataFrame(selected_country, columns=['Country'])
    columns = roaming['date'].unique()
    df_columns = pd.DataFrame(columns=columns)
    df_roaming = pd.concat((df_roaming, df_columns))
    iso_to_country = countries_Korea_inbound.set_index('iso').to_dict()['Country']
    country_to_iso = {v:k for k,v in iso_to_country.items()}
    for country in selected_country:
        df_roaming.loc[df_roaming.Country == country, df_roaming.columns[1:]] = roaming.loc[roaming.iso == country_to_iso[country], ['counts']].T.values
    df_roaming.columns = df_roaming.columns.astype(str)
    df_roaming = df_roaming.rename(columns = {col: str(int(col[4:6]))+'/'+str(int(col[-2:]))+'/' + col[2:4] for col in df_roaming.columns[1:].astype(str)})
    df_roaming.loc[:, df_roaming.columns[1:]] = df_roaming.loc[:, df_roaming.columns[1:]]/df_roaming.loc[:, df_roaming.columns[1:]].sum(axis=0)
    
    mean, std = df_roaming.iloc[:,1:].mean(axis=1), df_roaming.iloc[:,1:].std(axis=1)
    std[24] += 1
    df_roaming.iloc[:,1:] = df_roaming.iloc[:,1:].subtract(mean, axis='index').divide(std, axis='index')
    
    print("Loading infection ratio data")
    df_infection_ratio = df_roaming.copy()
    df_infection_ratio.loc[:, df_infection_ratio.columns[1:]] = 0
    for idx, country in enumerate(df_infection_ratio.Country):
        df_infection_ratio.loc[df_infection_ratio.Country == country, df_infection_ratio.columns[1:]] = df_confirm.loc[df_confirm['Country/Region'] == country, df_infection_ratio.columns[1:]].values/countries_Korea_inbound.loc[countries_Korea_inbound.Country == country, 'population'].values
    
    mean, std = df_infection_ratio.iloc[:,1:].mean(axis=1), df_infection_ratio.iloc[:,1:].std(axis=1)
    df_infection_ratio.iloc[:,1:] = df_infection_ratio.iloc[:,1:].subtract(mean, axis='index').divide(std, axis='index')

    print("Loading passenger flights data")
    passenger_flights = pd.read_csv('./dataset/ICN_arrive_preprocess.csv')
    passenger_flights.rename(columns={passenger_flights.columns[0]:'Country'}, inplace=True)
    passenger_flights.loc[:, passenger_flights.columns[1:]] = passenger_flights.loc[:, passenger_flights.columns[1:]]/passenger_flights.loc[:, passenger_flights.columns[1:]].sum(axis=0)
    
    mean, std = passenger_flights.iloc[:,1:].mean(axis=1), passenger_flights.iloc[:,1:].std(axis=1)
    passenger_flights.iloc[:,1:] = passenger_flights.iloc[:,1:].subtract(mean, axis='index').divide(std+1e-5, axis='index')

    data_AUX = generate_COVID_aux_input(df_roaming,
                            df_infection_ratio,
                            passenger_flights,
                            df_incoming)
    
    print("Normalizing continent target")
    cat_target_continent = None
    for i in range(len(target_continent)):
        if i==0:
            cat_target_continent = target_continent[i]
        else :
            cat_target_continent = np.concatenate([cat_target_continent, target_continent[i][-1:,:]],axis=0)
    mean, std = cat_target_continent.mean(axis=0),cat_target_continent.std(axis=0)
    if output_size==7:
        pickle.dump((mean, std), open("pickled_ds/target_continent_mean_std_window7.pkl", "wb"))
    else :
        pickle.dump((mean, std), open("pickled_ds/target_continent_mean_std_window14.pkl", "wb"))
    target_continent = target_continent.astype(np.float)
    for i in range(len(target_continent)):
        target_continent[i] = (target_continent[i]-mean)/std
    
    
    print("Normalizing total target")
    cat_target_total = np.concatenate([target_total[:,0],target_total[-1,1:]], axis=0)
    mean, std = cat_target_total.mean(),cat_target_total.std()
    if output_size==7:
        pickle.dump((mean, std), open("pickled_ds/target_total_mean_std_window7.pkl", "wb")) #target_total_mean_std_window14
    else:
        pickle.dump((mean, std), open("pickled_ds/target_total_mean_std_window14.pkl", "wb"))
    target_total = (target_total-mean)/std
    
    if save :
        if output_size==7:
            # train_data
            pickle.dump(countries_Korea_inbound, open("pickled_ds/countries_Korea_inbound_window7_google.pkl", "wb"))
            pickle.dump(data_model2[:-1], open("pickled_ds/data_model2_normal_window7_google.pkl", "wb"))
            pickle.dump(data_AUX[:-1], open("pickled_ds/data_AUX_normal_window7_google.pkl", "wb"))
            pickle.dump(target_continent[:-1], open("pickled_ds/target_continent_normal_window7_google.pkl", "wb"))
            pickle.dump(target_total[:-1], open("pickled_ds/target_total_normal_window7_google.pkl", "wb"))            
            # test_data
            pickle.dump(countries_Korea_inbound, open("pickled_ds/countries_Korea_inbound_window7_google_test.pkl", "wb"))
            pickle.dump(data_model2[-1], open("pickled_ds/data_model2_normal_window7_google_test.pkl", "wb"))
            pickle.dump(data_AUX[-1], open("pickled_ds/data_AUX_normal_window7_google_test.pkl", "wb"))
            pickle.dump(target_continent[-1], open("pickled_ds/target_continent_normal_window7_google_test.pkl", "wb"))
            pickle.dump(target_total[-1], open("pickled_ds/target_total_normal_window7_google_test.pkl", "wb"))
        else : #output_size==14
            # train_data
            pickle.dump(countries_Korea_inbound, open("pickled_ds/countries_Korea_inbound_window14_google.pkl", "wb"))
            pickle.dump(data_model2[:-1], open("pickled_ds/data_model2_normal_window14_google.pkl", "wb"))
            pickle.dump(data_AUX[:-1], open("pickled_ds/data_AUX_normal_window14_google.pkl", "wb"))
            pickle.dump(target_continent[:-1], open("pickled_ds/target_continent_normal_window14_google.pkl", "wb"))
            pickle.dump(target_total[:-1], open("pickled_ds/target_total_normal_window14_google.pkl", "wb"))
            # test_data
            pickle.dump(countries_Korea_inbound, open("pickled_ds/countries_Korea_inbound_window14_google_test.pkl", "wb"))
            pickle.dump(data_model2[-1], open("pickled_ds/data_model2_normal_window14_google_test.pkl", "wb"))
            pickle.dump(data_AUX[-1], open("pickled_ds/data_AUX_normal_window14_google_test.pkl", "wb"))
            pickle.dump(target_continent[-1], open("pickled_ds/target_continent_normal_window14_google_test.pkl", "wb"))
            pickle.dump(target_total[-1], open("pickled_ds/target_total_normal_window14_google_test.pkl", "wb"))

    return data_model2, data_AUX, target_continent, target_total

def main():
    global opts
    opts = parser.parse_args()
    data_model2, data_AUX, target_continent, target_total = dataloader(opts.output_size, save=opts.save)
    
    return data_model2, data_AUX, target_continent, target_total

if __name__ == '__main__':
    main()