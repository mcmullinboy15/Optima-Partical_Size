import os
import torch
import pandas as pd
from Train import Train
from model_and_defs import Model2
from loop_train import loop_train

params = {
    'EPOCHS'  : [1000],
    'BATCHES' : [2 ** i for i in range(4, 10)],
    'MODELS'  : [Model2],
    'H1S'     : [128],                               # [16 * i for i in range(1, 20)],
    'H2S'     : [128],                             # [16 * i for i in range(1, 20)],
    'H3S'     : [128],                             # [16 * i for i in range(1, 30)], Try skipping if Model1
    'LRS'     : [1e-4],        #1e-2, 1e-3,                         # [0.001, 0.01, 0.1, 1],         # https://pytorch.org/docs/stable/optim.html
    'OPTIMS'  : [torch.optim.Adam],                    #torch.optim.adadelta # torch.optim.SGD, needed to add momentum
    # 'LOSS'   : [torch.nn.L1Loss, torch.nn.MSELoss, torch.nn.BCELoss]
    # 'DROPOUT': [0-1]
}
trained = []
lowests = []

for branch in ['Branch_A', 'Branch_B']:
    data_folder = f"{branch}/data/"

    for data_filename in os.listdir(data_folder):
        data_path = f"{data_folder}{data_filename}"
        print(data_path)
        std_scaled = data_path.__contains__('Std')
        std = 'Std' if std_scaled else ''
        data_type = 'Cautious' if data_path.__contains__('Cautious') else 'Tight' if data_path.__contains__('Tight') else 'MinMax'
        # I'm going to skip the std ones
        if not std_scaled:
            trained.append(loop_train(f'{branch}/models/', f'{data_type}{std}', params, data_path, std_scaled))

lowest = 1000000000000
for traine in trained:
    for train in traine:
        if train.lowest_val_loss < lowest:
            lowest = train.lowest_val_loss
            lowests.append(train)

for low in lowests:
    print('lowest:', low.filename, low.lowest_val_loss)
