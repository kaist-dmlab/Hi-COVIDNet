import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

from tqdm.notebook import tqdm
import pickle
import copy
import pathlib

import matplotlib.pyplot as plt
from covid_aux import COVID_AUX_Net, train_COVID_AUX_Net, GlobalRNN, train_globalrnn
import os

os.environ['CUDA_VISIBLE_DEVICES']= '5'

countries_Korea_inbound = pickle.load(open("pickled_ds/countries_Korea_inbound_window14_google.pkl", "rb"))
test_data_model2 = pickle.load(open("pickled_ds/data_model2_normal_window14_google_test.pkl", "rb"))
test_data_model2 = [test_data_model2]
test_data_AUX = pickle.load(open("pickled_ds/data_AUX_normal_window14_google_test.pkl", "rb"))
test_data_AUX = [test_data_AUX]
test_target_continent = pickle.load(open("pickled_ds/target_continent_normal_window14_google_test.pkl", "rb"))
test_target_continent = np.expand_dims(test_target_continent, axis=0)
test_target_total = pickle.load(open("pickled_ds/target_total_normal_window14_google_test.pkl", "rb"))
test_target_total = np.expand_dims(test_target_total, axis=0)

root = pathlib.PosixPath("models_grid_search/tm_14days_full")

model = COVID_AUX_Net(countries_Korea_inbound, feature_len=10, aux_len=3, hidden_size=4, is_tm = True, output_size=14,)

model_list = [model]
fnames = ["model.pt"]

for i in range(len(model_list)):
    print(i+1,"th model loading")
    state = torch.load(root/fnames[i])
    model_list[i].load_state_dict(state)
    model_list[i].to(torch.device("cuda"))
    model_list[i].eval()
    
criterion = nn.MSELoss()
alpha = .5
Valid_Loss, RMSE_Loss = [],[]
continents_ouputs = []
outputs = []
models_rmse = []

for j in range(len(model_list)):
    with torch.no_grad():
        for i in range(len(test_data_model2)):
            continent_patients_pred, total_patients_pred = model_list[j](test_data_model2[i], test_data_AUX[i])
            target_continent_i = torch.as_tensor(test_target_continent[i], dtype=torch.float)
            target_total_i = torch.as_tensor(test_target_total[i], dtype=torch.float).unsqueeze(0)
            target_continent_i[torch.isnan(target_continent_i)] = 0
            target_continent_i[torch.isinf(target_continent_i)] = 0
            target_total_i[torch.isnan(target_total_i)] = 0
            target_total_i[torch.isinf(target_total_i)] = 0

            # continent_patients_pred : (6,14)
            loss1 = criterion(continent_patients_pred, target_continent_i.cuda().transpose(1,0).contiguous())
            loss2 = criterion(total_patients_pred, target_total_i.cuda().squeeze())

            valid_loss = loss1*alpha + loss2*(1-alpha)
            Valid_Loss.append(valid_loss.item())
            RMSE_Loss.append(torch.sqrt(loss2))

            continents_ouputs.append(continent_patients_pred) #(6,14)
            outputs.append(total_patients_pred)
        
        avg_rmse_loss = sum(RMSE_Loss[-len(test_data_model2):])/len(test_data_model2)
        print("RMSE is ",avg_rmse_loss.item())
        models_rmse.append(avg_rmse_loss)
        
continent_mean, continent_std = pickle.load(open("pickled_ds/target_continent_mean_std_window14.pkl", "rb"))
total_mean, total_std = pickle.load(open("pickled_ds/target_total_mean_std_window14.pkl", "rb"))

print("The predicted number of imported cases by Hi-COVIDNet daily basis: \n", outputs[0].cpu().numpy()*total_std+total_mean)
print("The true number of imported cases daily basis: \n", test_target_total[0]*total_std+total_mean)
print()
print("The predicted number of continent-wise cases by Hi-COVIDNet: \n", continents_ouputs[0].transpose(1,0).cpu().numpy()*continent_std+continent_mean)
print("The true number of continent-wise: \n", test_target_continent[0]*continent_std+continent_mean)

