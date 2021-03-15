import torch
from model_and_defs import Model1, Model2
from loop_train import loop_train


data = "data/Compiled_Data.csv"

#  TODO https://pytorch.org/docs/stable/
params = {
    'EPOCHS'  : [16 * i for i in range(1, 20)],       # [1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 14400, 16000, 17600, 19200, 20800, 22400, 24000, 25600, 27200, 28800, 30400]
    'BATCHES' : [2 ** i for i in range(4, 10)],         # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    'MODELS'  : [Model2],                       # Never Model3
    'H1S'     : [128],                              # [16 * i for i in range(1, 20)],
    'H2S'     : [128],                             # [16 * i for i in range(1, 20)],
    'H3S'     : [128],                             # [16 * i for i in range(1, 30)], Try skipping if Model1
    'LRS'     : [1e-4],        #1e-2, 1e-3,                         # [0.001, 0.01, 0.1, 1],         # https://pytorch.org/docs/stable/optim.html
    'OPTIMS'  : [torch.optim.Adam],                    #torch.optim.adadelta # torch.optim.SGD, needed to add momentum
    # 'LOSS'   : [torch.nn.L1Loss, torch.nn.MSELoss, torch.nn.BCELoss]
    # 'DROPOUT': [0-1]
}
# if __name__ is '__main__':
loop_train('Less_Epochs', params)