import torch
from model_and_defs import Model1, Model2, Model3
from loop_train import loop_train


data = "data/ERROR/preds.csv"

params = {
    'EPOCHS'  : [1600 * i for i in range(1, 20)],       # [1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 14400, 16000, 17600, 19200, 20800, 22400, 24000, 25600, 27200, 28800, 30400]
    'BATCHES' : [10 * i for i in range(1, 8)],         # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    'MODELS'  : [Model1, Model2],
    'H1S'     : [64, 128],                              # [16 * i for i in range(1, 20)],
    'H2S'     : [128, 256],                             # [16 * i for i in range(1, 20)],
    'H3S'     : [128, 256],                             # [16 * i for i in range(1, 30)],
    'LRS'     : [1e-4],                                 # [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],         # https://pytorch.org/docs/stable/optim.html
    'OPTIMS'  : [torch.optim.Adam],                     # torch.optim.SGD, needed to add momentum
    # 'LOSS'    : [torch.nn.L1Loss, torch.nn.MSELoss, torch.nn.BCELoss]
}

if __name__ is '__main__':
    loop_train('ERROR', params, data)
