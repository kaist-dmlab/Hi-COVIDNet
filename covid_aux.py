import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils

import math
import copy
from tqdm.notebook import tqdm

import os

class COVID_AUX_Net(nn.Module):
    def __init__(self, df_countries, aux_len=4, feature_len=6, hidden_size=16, num_layers=1, is_aux = True, is_tm = False, output_size = 1):
        """
        Parameters
        ----------
        df_countries : dataframe of countries which includes
            ['Country', 'continent', 'n_confirmed', 'visit', 'iso']
        aux_len : length of auxiliary information that will be concatenated to the hidden dimension
        feature_len : dimension of the features
        hidden_size : hidden dimension size of LSTM
        num_layers : number of LSTM layers
        """
        super(COVID_AUX_Net, self).__init__()
        self.is_aux = is_aux
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.continents = df_countries['continent'].unique()
        self.countries = np.array(df_countries.loc[df_countries.visit.eq(1), 'Country'].values)
        self.countries = np.delete(self.countries, np.argwhere(self.countries == 'Korea, South'))
    
    
        self.country_continent = df_countries.set_index('Country').to_dict()['continent']
        self.continent_country_cnt = {c: len(df_countries.loc[(df_countries.visit.eq(1)) & (df_countries.continent.eq(c))]) for c in self.continents}
        self.continent_country_cnt['Asia'] -= 1
        # LSTM for each country's trend of patients within the last 14 days
        
        if is_tm :
            self.LSTM_countries = nn.ModuleDict({
                c: nn.Sequential(
                    TransformerModel(ninp=feature_len, nhead=5, nhid=hidden_size, nlayers=num_layers,),
                    nn.LSTM(feature_len, hidden_size, num_layers, batch_first=True)) for c in self.countries
                })
        else : 
            self.LSTM_countries = nn.ModuleDict({
                c: nn.Sequential(
                    nn.LSTM(feature_len, hidden_size, num_layers, batch_first=True)) for c in self.countries
                })
        
        if self.is_aux : 
            self.FCN_continent_patients = nn.ModuleDict({
                c: nn.Sequential(nn.Linear((hidden_size + aux_len)*self.continent_country_cnt[c], 8), 
                                 nn.ReLU(), 
                                 nn.Linear(8, self.output_size)) for c in self.continent_country_cnt
                })
        else : # no auxilary data
            self.FCN_continent_patients = nn.ModuleDict({
                c: nn.Sequential(nn.Linear(hidden_size*self.continent_country_cnt[c], 8), 
                                 nn.ReLU(), 
                                 nn.Linear(8, self.output_size)) for c in self.continent_country_cnt
                })
        if self.output_size > 1 :
            self.FCN_total_patients = nn.ModuleDict({
                day: nn.Linear(len(self.continents), 1) for day in np.array(list(range(self.output_size))).astype(str)
                })
        else:
            self.FCN_total_patients = nn.Linear(len(self.continents)*self.output_size, self.output_size)

    def forward(self, x, aux):
        """
        x: one day with countries {country_name: data of shape (14, feature_len)}
        """
        countries_hidden = {}
        for c in self.countries:
            x_c = torch.as_tensor(x[c], dtype=torch.float).cuda()
            x_c[torch.isnan(x_c)] = 0
            x_c[torch.isinf(x_c)] = 0
            
            out, (h_0, c_0) = self.LSTM_countries[c](x_c.unsqueeze(0)) # LSTM prediction of the country
            c_ = self.country_continent[c] # Get the continent of the country
            if self.is_aux:
                h = F.relu(out[:, -1, :].squeeze(0))
                h_ = torch.cat((h, torch.as_tensor(aux[c], dtype=torch.float).cuda()),0) # Concatenate auxiliary information
            else :
                h_ = F.relu(out[:, -1, :].squeeze(0)) # no auxiliary information
                
            countries_hidden[c_] = torch.cat((countries_hidden[c_], h_), 0) if c_ in countries_hidden else h_ # Concat hidden vectors
        
        if self.output_size > 1: 
            continent_patients_pred = torch.zeros(len(self.continents), self.output_size).cuda() # (#continents:6, output_size:14)
        else : # self.output_size = 1
            continent_patients_pred = torch.zeros(len(self.continents)).cuda() # (#continents,)
        
        for idx, c in enumerate(self.continents):
                continent_patients_pred[idx] = F.relu(self.FCN_continent_patients[c](countries_hidden[c]))
        
        if self.output_size > 1:
            total_patients_pred = []
#             output, _ = self.lstm_total_patients(continent_patients_pred.transpose(1,0).contiguous().unsqueeze(0)) #(1, T, D) 
#             output = output.squeeze() #(T:14,D:6)
#             print("output.size()", output.size())
            output = continent_patients_pred.transpose(1,0).contiguous() # (T,D)
            for idx,day in enumerate(range(output.size(0))):
                total_patients_pred.append(self.FCN_total_patients[str(day)](output[idx]))
            total_patients_pred = torch.cat(total_patients_pred, axis=0)
        else :
            total_patients_pred = self.FCN_total_patients(continent_patients_pred)
        
        return continent_patients_pred, total_patients_pred
    
def train_COVID_AUX_Net(model, train_data_model2, train_data_AUX, train_target_continent,
                        train_target_total,test_data_model2,test_data_AUX, test_target_continent,
                        test_target_total, num_epoch, model_name="AUX_Net", beta=0.4, lr=None):
    """
    @param target_total : (n, output_size : 14 or 7)
    @param target_continet : (n, output_size : 14 or 7, #continent : 6)
    """
    
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20,50,80])
    
    Loss_total = []
    Valid_Loss = []
    RMSE_Loss = []
    
    model.cuda()
    best_valid_loss = np.inf
    
    for e in tqdm(range(num_epoch)):
        ## train
        model.train()
        for i in range(len(train_data_model2)):
            continent_patients_pred, total_patients_pred = model(train_data_model2[i], train_data_AUX[i])
            target_continent_i = torch.as_tensor(train_target_continent[i], dtype=torch.float)
            target_total_i = torch.as_tensor(train_target_total[i], dtype=torch.float).unsqueeze(0)
            target_continent_i[torch.isnan(target_continent_i)] = 0
            target_continent_i[torch.isinf(target_continent_i)] = 0
            target_total_i[torch.isnan(target_total_i)] = 0
            target_total_i[torch.isinf(target_total_i)] = 0
            
            optimizer.zero_grad()
            loss1 = criterion(continent_patients_pred, target_continent_i.cuda().transpose(1,0))
            loss2 = criterion(total_patients_pred, target_total_i.cuda().squeeze())
            
            loss = loss1*beta + loss2*(1-beta)
            loss.backward()
            optimizer.step()
            Loss_total.append(loss.item())
            
            if i % 5 == 4 :
                    print("{e}th epoch train loss : {l}".format(e = e, l = loss))
            
        scheduler.step()
        if e % 20 == 19 :
            torch.save(model.state_dict(), "{model_name}_{e}.pt".format(model_name = model_name,
                                                                                   e=i))
            print(e,"th epoch: model saved!")
        
        ## validation 
        model.eval()
        with torch.no_grad():
            for i in range(len(test_data_model2)):
                continent_patients_pred, total_patients_pred = model(test_data_model2[i], test_data_AUX[i])
                target_continent_i = torch.as_tensor(test_target_continent[i], dtype=torch.float)
                target_total_i = torch.as_tensor(test_target_total[i], dtype=torch.float).unsqueeze(0)
                target_continent_i[torch.isnan(target_continent_i)] = 0
                target_continent_i[torch.isinf(target_continent_i)] = 0
                target_total_i[torch.isnan(target_total_i)] = 0
                target_total_i[torch.isinf(target_total_i)] = 0

                loss1 = criterion(continent_patients_pred, target_continent_i.cuda().transpose(1,0))
                loss2 = criterion(total_patients_pred, target_total_i.cuda().squeeze())
                
                
                valid_loss = loss1*beta + loss2*(1-beta)
                Valid_Loss.append(valid_loss.item())
                RMSE_Loss.append(torch.sqrt(loss2))
                
            avg_val_loss = sum(Valid_Loss[-len(test_data_model2):])/len(test_data_model2)
            avg_rmse_loss = sum(RMSE_Loss[-len(test_data_model2):])/len(test_data_model2)
            
            print("{e}th epoch avg_valid_modelloss : {l} avg_rmse_loss : {rmse}".format(e = e, l = avg_val_loss,
                                                                                        rmse = avg_rmse_loss))
        
        
        if avg_val_loss < best_valid_loss :
            best_valid_loss = avg_val_loss
            torch.save(model.state_dict(), "{model_name}_best.pt".format(model_name = model_name,))
            print("best model saved!")
            
        print("############ epoch finished ############\n")
    
    return Loss_total, Valid_Loss, RMSE_Loss

class GlobalRNN(nn.Module):
    def __init__(self, input_dim=6, hidden_size=None,is_tm=False):
        super(GlobalRNN, self).__init__()
        if is_tm:
            self.rnn = nn.Sequential(
                TransformerModel(ninp=input_dim, nhead=3, nhid=hidden_size, nlayers=1,),
                nn.LSTM(input_dim, hidden_size, batch_first=True))
        else : 
            self.rnn = nn.LSTM(input_dim, hidden_size, batch_first=True)

        self.linear = nn.Linear(hidden_size, input_dim)

    def forward(self, x):
        out, hidden = self.rnn(x) 
        #print(out.shape)
        #print(hidden.shape)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        result = self.linear(F.relu(out[:, -1, :]))
        return result
    

def train_globalrnn(model, data, target, valid_data, valid_target, num_epoch, batch_size, fname, lr =.03,):
    data_length = len(data)
    data_indices = list(range(data_length))
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [20,50,80])
    Loss = []
    Valid_Loss = []
    best_val_loss = np.inf

    for i in tqdm(range(num_epoch)):
        
        np.random.shuffle(data_indices)
        model.train()

        for iteration in range(int(data_length/batch_size)):
            data_mini_batch = data[data_indices[iteration*batch_size:(iteration+1)*batch_size]]
            target_mini_batch = target[data_indices[iteration*batch_size:(iteration+1)*batch_size]]
            data_mini_batch = torch.as_tensor(data_mini_batch)
            target_mini_batch = torch.as_tensor(target_mini_batch)
            
            prediction = model(data_mini_batch)
        
            optimizer.zero_grad()
            loss = criterion(prediction, target_mini_batch)
            loss.backward()
                
            optimizer.step()
            Loss.append(loss.item())
            
            if iteration % 2 == 1 :
                print("{e}th epoch train loss : {l}".format(e = i, l = loss))
            
        scheduler.step()
        if i % 20 == 19 :
            torch.save(model.state_dict(), "{fname}_{e}.pt".format(fname=fname,e=i))
        
        ## validation 
        model.eval()
        with torch.no_grad():
            valid_data = torch.as_tensor(valid_data)
            valid_target = torch.as_tensor(valid_target)
            
            valid_prediction = model(valid_data)
            valid_loss = criterion(valid_prediction, valid_target)
            Valid_Loss.append(valid_loss)
            print("{e}th epoch valid loss : {l}".format(e = i, l = valid_loss))
        
        
        if valid_loss.item() < best_val_loss :
            best_val_loss = valid_loss.item()
            torch.save(model.state_dict(), "{fname}_best.pt".format(fname=fname))
            print("best model saved!")
            
        print("############ epoch finished ############\n")
    
    return Loss, Valid_Loss

class PositionalEncoding(nn.Module):

    def __init__(self, d_model=6, dropout=0.1, max_len=15):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)# (1,15,6) batch_first : 
        pe = pe.to(torch.float32)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.to(torch.float32)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout).to(torch.float32)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,).to(torch.float32)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers,).to(torch.float32)
        self.ninp = ninp

    def forward(self, src):
        """
        src : (N, T, D) = (1,14,6)
        """
        src = src.transpose(0,1).contiguous() # (T, N, D) : (14,1,6)
        src = src * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        src = src.to(torch.float32)
        output = self.transformer_encoder(src,) #self.src_mask : we dont need attn mask! bc of the same length data
        output = output.transpose(0,1).contiguous() # (N, T, D) = (1,14,6)
        return output.to(torch.float32)
