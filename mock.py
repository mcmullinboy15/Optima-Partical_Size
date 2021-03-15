from mocks import models

import sys, time
import torch
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
import talib

from model_and_defs import Model1, Model2, Model3, separate_train_val_test

data = "data/Compiled_Data.csv"
# data = 'data/Large_Test_DataSet.csv'

PLOT_SIZE = 50
OFF = 1
PLOT = False
INPUT = False
use_mov_ave = False
mov_ave = 10
remove_features_arr = [[], [10, 11], [13]]

df = pd.read_csv(data)

dfs = []
# Removing the unwanted features from the df before it's a numpy array
if len(remove_features_arr) > 0:
    for val in remove_features_arr:
        remove_cols = df.columns[val]
        dfs.append(df.drop(remove_cols, axis=1))

df = dfs[0]

xs = []
ys = []
# for _df in dfs:
data_np = df.to_numpy().astype(dtype='float64')

data = separate_train_val_test(data_np)
test = data['val']
xTest = test['x']
yTest = test['y'].reshape(-1, 1)
    # xs.append(xTest)
    # ys.append(yTest)


nns = []
for i, model_file in zip(range(len(models)), models):

    # print(model_file[-10:])
    try:
        # print('Model1')
        model = Model1(xTest.shape[1], 128, yTest.shape[1], H2=256, H3=None)
        model.load_state_dict(torch.load(model_file)['model_state_dict'])
        # for yTest, xTest in zip(ys, xs):
        #     _model = Model1(xTest.shape[1], 128, yTest.shape[1], H2=256, H3=None)
        #     try:
        #         _model.load_state_dict(torch.load(model_file)['model_state_dict'])
        #         model = _model
        #     except:
        #         continue
    except:
        try:
            # print('Model2')
            model = Model2(xTest.shape[1], 128, yTest.shape[1], H2=256, H3=None)
            model.load_state_dict(torch.load(model_file))
        except:
            # print('Model3')
            model = Model3(xTest.shape[1], 128, yTest.shape[1], H2=256, H3=None)
            try:
                model.load_state_dict(torch.load(model_file))
            except:
                # print(model_file[-10:], 'NOT USED')
                pass
    try:
        model.eval()
        model.double()
        nns.append(model)
    except:
        pass
        models.pop()


reals = []
preds = []
predsMA = [[None]* len(nns)] * mov_ave

final_pred = [[None]* len(nns)] * mov_ave

removes = []
rm = []  # [36, 41, 46, 72, 74]

counters = [0] * len(nns)
with torch.no_grad():
    plt.ion()
    # -1 means we didn't take the trade, -2 means it took profit already
    for i, y, x in (zip(range(len(yTest)), yTest, xTest)): # tqdm   , total=len(x)
        # if rm.__contains__(i):
        #     continue

        pred = [model(torch.from_numpy(x)).item() for model in nns]

        for i, pred_ in zip(range(len(pred)), pred):
            if (round(y.item()) - OFF) <= round(pred_) <= (round(y.item()) + OFF):
                counters[i] += 1

        mul, add_ = 1, 0
        if True: #(60 <= min(pred)) and (max(pred) <= 80):
            preds.append(pred)
            reals.append( ( y * mul ) + add_)

            # print(preds)

            predMA = []
            if use_mov_ave:
                if len(preds) > mov_ave:
                    _preds = np.array(preds)
                    for _pred in _preds.transpose():
                        ma = talib.SMA(_pred, timeperiod=mov_ave)
                        ma = pd.DataFrame(ma).iloc[-1]
                        predMA.append(ma[0])
                else:
                    continue
            predsMA.append(predMA)



        if PLOT:
            if len(preds) > PLOT_SIZE:

                print(preds)
                print(type(preds))
                print(predsMA)
                print(type(predsMA))

                plt.plot(preds, label='Predictions')
                plt.plot(reals, linewidth=4, label='y')
                if use_mov_ave:
                    plt.plot(predsMA, label='PredsMA')

                plt.legend()

                plt.draw()
                plt.pause(0.000000000001)
                plt.clf()

                # plt.show()

                preds.pop(0)
                reals.pop(0)
                if use_mov_ave:
                    predsMA.pop(0)
                # sys.exit(-3)

        print(len(preds))
        if INPUT:
            try:
                remove = int(input(f'{i}: remove:'))
                print(i, remove)
                preds.pop(remove)
                reals.pop(remove)
                removes.append(i + (remove - 25))
            except:
                pass
print()
print(*list(range(len(counters))), sep='\t')
print(*counters, f"length:{len(counters)}", sep='\t')
print(*models, sep='\t')
print(len(models))

print(preds)
# print(predsMA)
preds = pd.DataFrame(preds)
# predsMA = pd.DataFrame(predsMA)


print(preds)
print(yTest)
print(len(yTest))
# for i in reversed(range(len(rm))):
#     yTest = np.delete(yTest, i)
# print(yTest)
# print(len(yTest))

preds.insert(0, 'y', yTest)
# preds['y'] = yTest
print(preds)

preds.to_csv('data/ERROR/preds.csv', index=False)
# preds.to_csv('data/ERROR/predsMA.csv')
