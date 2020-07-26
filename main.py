import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

import pickle
import copy
import argparse

from covid_aux import COVID_AUX_Net, train_COVID_AUX_Net, GlobalRNN, train_globalrnn
import os


parser = argparse.ArgumentParser(description='Hi-covidnet')

# basic settings
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 100)')
parser.add_argument('--model_path', default='models_grid_search/tm_14days_full/tanh_hid4', type=str, help='prefix of path of the model')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu_ids: e.g. 0,1,2,3,4,5')


# basic hyper-parameters
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 0.03)')
parser.add_argument('--beta', type=float, default=.5, metavar='BETA', help='ratio of continent loss and total loss (default: 0.5)')
parser.add_argument('--hidden_size', type=int, default=4, metavar='HIDDEN', help='hidden size of LSTM and Transformer(default: 4) e.g. 2,4,8, ... depending on your dataset')
parser.add_argument('--output_size', type=int, default=14, metavar='OUTPUT', help='How many days you are predicting')
parser.add_argument('--is_aux', action='store_true', default=False, help='use auxilary data')
parser.add_argument('--is_tm', action='store_true', default=False, help='use transformer')

def main():
    global opts
    opts = parser.parse_args()
    
    # set gpu
    os.environ['CUDA_VISIBLE_DEVICES']= opts.gpu_id
    
    # set train data
    train_data_model2 = pickle.load(open("pickled_ds/data_model2_normal_window14_google.pkl", "rb"))
    train_data_AUX = pickle.load(open("pickled_ds/data_AUX_normal_window14_google.pkl", "rb"))
    train_target_continent = pickle.load(open("pickled_ds/target_continent_normal_window14_google.pkl", "rb"))
    train_target_total = pickle.load(open("pickled_ds/target_total_normal_window14_google.pkl", "rb"))
    countries_Korea_inbound = pickle.load(open("pickled_ds/countries_Korea_inbound_window14_google.pkl", "rb"))
    print("trainset loaded")
    
    # set test data
    test_data_model2 = pickle.load(open("pickled_ds/data_model2_normal_window14_google_test.pkl", "rb"))
    test_data_model2 = [test_data_model2]
    test_data_AUX = pickle.load(open("pickled_ds/data_AUX_normal_window14_google_test.pkl", "rb"))
    test_data_AUX = [test_data_AUX]
    test_target_continent = pickle.load(open("pickled_ds/target_continent_normal_window14_google_test.pkl", "rb"))
    test_target_continent = np.expand_dims(test_target_continent, axis=0)
    test_target_total = pickle.load(open("pickled_ds/target_total_normal_window14_google_test.pkl", "rb"))
    test_target_total = np.expand_dims(test_target_total, axis=0)
    print("testset loaded")

    feature_len = train_data_model2[0]['Argentina'].shape[1]
    aux_len = train_data_AUX[0]['Argentina'].shape[0]
    
    best_models = {}
    for i in range(20):
        print("######" ,i,"th training start", "######")
        model = COVID_AUX_Net(countries_Korea_inbound, 
                              feature_len=feature_len, 
                              aux_len=aux_len,
                              hidden_size=opts.hidden_size,
                              is_tm = opts.is_tm,
                              output_size=opts.output_size)

        loss, val_loss, rmse_loss = train_COVID_AUX_Net(model, 
                                                        train_data_model2, 
                                                        train_data_AUX,train_target_continent,
                                                        train_target_total,
                                                        test_data_model2,
                                                        test_data_AUX,
                                                        test_target_continent,
                                                        test_target_total, 
                                                        num_epoch=opts.epochs,
                                                        model_name="{}_{}".format(opts.model_path,i),
                                                        lr = opts.lr,
                                                        beta=opts.beta)
        best_models["{}".format(i)] = sum(rmse_loss[-7:])/7
        print(best_models)
        
if __name__ == '__main__':
    main()