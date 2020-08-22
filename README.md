# Hi-COVIDNet : Deep Learning Approach to Predict Inbound COVID-19 Patients and Case Study in South Korea

## About
Source code and datasets of the paper [Hi-COVIDNet : Deep Learning Approach to Predict Inbound COVID-19 Patients and Case Study in South Korea](https://dl.acm.org/doi/10.1145/3394486.3412864), KDD 2020

Since the data from Korea Telecom(KT) is not open to the public, if you would like to run the code, please contact KT. We are contacting KT for data release.


## Installation
Requirements

  - Python 3.6 (Recommend Anaconda)
  - Ubuntu 16.04.3 LTS
  - Pytorch >= 1.2.0
  - Numpy >= 1.17.2

## Usage
  - Download all codes (*\*.py*) and put them in the same folder
  - Create "model_grid_search" folder in the same folder
  - Create "pickled_ds" folder for dataset and mean_std data 
  - Open terminal in the same folder
  - Run "python data_loader.py" to preprocess and save data in ".pkl" format
  ```bash
  python data_loader.py -h
usage: data_loader.py [-h] [--output_size O] [--save]

Hi-covidnet DATALOADER

optional arguments:
  -h, --help       show this help message and exit
  --output_size O  How many days you are predicting(default: 14)
  --save           Saving pre-processed data
  ```
  example usage : 
  
  ```bash
  python data_loader.py --output_size 14
data shape is  32 (14, 10)
target_continent shape is  (32, 14, 6) target_total shape is  (32, 14)
Loading KT roaming data
Loading infection ratio data
Loading passenger flights data
Normalizing continent target
Normalizing total target
  ```
 
  - Run "python main.py" to train Hi-COVIDNet
  ```bash
  python main.py -h
usage: main.py [-h] [--epochs N] [--model_path MODEL_PATH] [--gpu_id GPU_ID]
               [--lr LR] [--beta BETA] [--hidden_size HIDDEN]
               [--output_size OUTPUT] [--is_aux] [--is_tm]

Hi-covidnet

optional arguments:
  -h, --help            show this help message and exit
  --epochs N            number of epochs to train (default: 100)
  --model_path MODEL_PATH
                        prefix of path of the model
  --gpu_id GPU_ID       gpu_ids: e.g. 0,1,2,3,4,5
  --lr LR               learning rate (default: 0.03)
  --beta BETA           ratio of continent loss and total loss (default: 0.5)
  --hidden_size HIDDEN  hidden size of LSTM and Transformer(default: 4) e.g.
                        2,4,8, ... depending on your dataset
  --output_size OUTPUT  How many days you are predicting
  --is_aux              use auxilary data
  --is_tm               use transformer
  ```
  
## Hyperparameters:
Please check the hyperparameters of Hi-COVIDNet defined in main.py
