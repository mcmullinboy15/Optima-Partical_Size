import torch
from model_and_defs import Model1, Model2, Model3
import pandas as pd

from loop_train import loop_train

params = {
    'EPOCHS' : [1600 * i for i in range(1, 20)],       # [1600, 3200, 4800, 6400, 8000, 9600, 11200, 12800, 14400, 16000, 17600, 19200, 20800, 22400, 24000, 25600, 27200, 28800, 30400]
    'BATCHES': [2 ** i for i in range(4, 10)],         # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190]
    'MODELS' : [Model1, Model2],
    'H1S'    : [64, 128],                              # [16 * i for i in range(1, 20)],
    'H2S'    : [128, 256],                             # [16 * i for i in range(1, 20)],
    'H3S'    : [128, 256],                             # [16 * i for i in range(1, 30)],
    'LRS'    : [1e-4],                                 # [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],         # https://pytorch.org/docs/stable/optim.html
    'OPTIMS' : [torch.optim.Adam],                     # torch.optim.SGD, needed to add momentum
    # 'LOSS'    : [torch.nn.L1Loss, torch.nn.MSELoss, torch.nn.BCELoss]
}

if __name__ is '__main__':
    loop_train('Classifications', params, data='data/Classifications/Class_data.csv')


# building the 'data/Classifications/Class_data.csv'
# ----1 = 50-54.9
# ----2 = 55-59.9
# ----3 = 60-64.9
# ----4 = 65-69.9
# ----5 = 70-74.9
# ----6 = 75-79.9
# ----7 = 80-84.9
# ----8 = 85-89.9

if False:
    c_data = "data/Compiled_Data.csv"
    df = pd.read_csv(c_data)

    print(max(df[df.columns[0]]))
    print(min(df[df.columns[0]]))

    results = []
    for idx, row in df.iterrows():
        n = row[0]
        if 50 <= n < 55:
            results.append(1)
        if 55 <= n < 60:
            results.append(2)
        if 60 <= n < 65:
            results.append(3)
        if 65 <= n < 70:
            results.append(4)
        if 70 <= n < 75:
            results.append(5)
        if 75 <= n < 80:
            results.append(6)
        if 80 <= n < 85:
            results.append(7)
        if 85 <= n < 90:
            results.append(8)

    print(len(df), len(results))
    df['groups'] = results
    groups = df.pop('groups')
    df.insert(0, 'groups', groups)
    df.pop(df.columns[1])
    print(df[df.columns[1]])
    df.to_csv('data/Classifications/Class_data.csv', index=False)
