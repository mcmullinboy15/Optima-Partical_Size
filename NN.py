import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm.auto import tqdm
from sklearn.preprocessing import MinMaxScaler

from model_and_defs import Model1, Model2, Model3, calculate_Accuracy, plot, getFileName, separate_train_val_test

START_TIME = datetime.now()

file = "data/Compiled_Data.csv"
name = 'Andrews_Epochs_1'
print(name)

import pandas as pd

from Test import Test
from Train import Train

TRAIN = True

df = pd.read_csv(file)

trains = [
    (),(),(),()
]

if TRAIN:
    for epo, bat in trains:
        train = Train(data_file=df, epochs=epo, batch_size=bat, H1=128, H2=256, save=True, NAME='AAPL')
        train.run()
        train.run_test()

else:

    testing = Test(data_file=df, model_file='Outputs/AAPL_1min_10shift_979585.pth', H1=128, H2=256, fx_pair='AAPL',
                   round_to=4, )
    testing.run_test(sym_0='', sym_1='', shift_pred=None, mov_ave=10)

    # TODO what if I made a MovingAverage of the Predictions to keep it above and below the real

sys.exit(-123)

df = pd.read_csv(file)  # , index_col=0)  # TODO: Add   `, index_col=0` when using the AAPL
# df = df[:-50]

data_np = np.array(df).astype(dtype='float64')
random.shuffle(data_np)  # TODO: You'll want to comment this out for market data

data = separate_train_val_test(data_np)
print(
    f"Total: {len(data_np)}, Train: {len(data['train']['x'])}, Validate: {len(data['val']['x'])}, Test: {len(data['test']['x'])}")

D_in = data['train']['x'].shape[1]
D_out = data['train']['y'].shape[1]
H1 = 128
H2 = 256
H3 = None
Model = Model1
Scale = False

learning_rate = 1e-4
epochs = 1000  # 500000
BATCH_SIZE = 10  # set to None for not being used
GOAL_LOSS = 0.000000005

# this is just something I use in the accuracy stuff at the bottom,
# if it's too low it'll cause problems
ROUND_TO = 3
SAVE = True
PERCENT_DISP = 1.05

# I don't print out the loss every time, I keep track of it in this
# and I print it when it is the new lowest loss
# but I also left it there if you wanted to un-comment it out
loss_rows = {}
loss = 100

train_loss_rows = {}
train_loss = 1000000

val_loss_rows = {}
val_loss = 1000000

model = Model(D_in, H1, D_out, H2=H2, H3=H3)
print(model)
# sys.exit(3)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if torch.cuda.is_available():
    model.cuda()


def core(phase, x, y):
    if torch.cuda.is_available():
        inputs = Variable(torch.from_numpy(x).cuda()).float()
        labels = Variable(torch.from_numpy(y).cuda())
    else:
        inputs = Variable(torch.from_numpy(x)).float()
        labels = Variable(torch.from_numpy(y))

    optimizer.zero_grad()

    y_pred = model(inputs).double()

    loss_ = criterion(y_pred, labels)
    # loss_rows.append(loss_.item())

    if phase == 'train':
        loss_.backward()
        optimizer.step()

    return loss_, y_pred


try:
    smallest_loss = 10000000.0
    ave_loss = 10000000.0

    # while ave_loss > GOAL_LOSS:
    for epoch in tqdm(range(epochs)):

        for j in range(0, len(data['train']['x']), BATCH_SIZE):
            for phase in ['train', 'val']:
                if phase == 'train':
                    print("=====  Training  =====")
                    # optimizer = scheduler(optimizer, epoch)
                    model.train(True)  # Set model to training mode
                else:
                    print("===== Validating =====")
                    model.train(False)  # Set model to evaluate mode

                x = data[phase]['x']
                y = data[phase]['y']

                if phase == 'train':
                    x = x[j:j + BATCH_SIZE]
                    y = y[j:j + BATCH_SIZE]

                if Scale:
                    s = MinMaxScaler(feature_range=(0, 1))
                    row = np.append(y, x, 1)
                    shape = row.shape
                    row1 = s.fit_transform(row.reshape(-1, 1)).reshape(shape)

                    x = row1[:, 1:]
                    y = row1[:, 0].reshape(shape[0], 1)

                loss, pred = core(
                    phase=phase,
                    x=x,
                    y=y,
                )

                if phase is 'val':
                    print()
                    print(f"LOSS: {loss}, PRED[0]: {pred[0].item()}, Y_ACTU[0]: {y[0].item()},")
                    val_loss = loss.item()
                    val_loss_rows.update(
                        {epoch: {'LOSS': val_loss, 'PRED[0]': pred[0].item(), 'Y_ACTU[0]': y[0].item()}})
                else:
                    print()
                    print(f"LOSS: {loss}, PRED[0]: {pred[0].item()}, Y_ACTU[0]: {y[0].item()},")
                    train_loss = loss.item()
                    train_loss_rows.update(
                        {epoch: {'LOSS': train_loss, 'PRED[0]': pred[0].item(), 'Y_ACTU[0]': y[0].item()}})

                loss_rows.update({epoch: {'val': val_loss, 'train': train_loss}})

            # only prints the loss and the time info when we have a smaller loss
            if float(loss.item()) < smallest_loss:
                smallest_loss = float(loss.item())
                print(f"\nSMALLEST: {smallest_loss}, AVERAGE: {ave_loss}")
                print(f"Start time: {START_TIME}, Running for: {datetime.now() - START_TIME}")
            elif loss.item() < smallest_loss * PERCENT_DISP:
                print(loss.item())

            if loss < GOAL_LOSS:
                break

        # print(loss_rows)
        ave_loss = float(sum(loss_rows)) / float(len(loss_rows))
        # loss_rows = []

    print(loss_rows)
    print(val_loss_rows)
    print(train_loss_rows)

except KeyboardInterrupt as e:
    pass

loss_df = pd.DataFrame(loss_rows, columns=["A"])

model_location = f"{os.getcwd()}\\Outputs\\Epochs\\{name}.pth"  # getFileName('pth', H, learning_rate, epochs, BATCH_SIZE, smallest_loss, folder='Models')
info = f"" \
       f"\nData Used (df):\n" \
       f"{df}\n" \
       f"\nDataFrame containing loss (loss_df): \n" \
       f"{loss_df}\n" \
       f"\nModel Used (model): \n" \
       f"{model}\n" \
       f"model saved to {model_location}\n" \
       f"\nTime Info: \n" \
       f"Start time: {START_TIME}, End time: {datetime.now()}, Length: {datetime.now() - START_TIME}\n\n"

if SAVE:
    torch.save(model.state_dict(), f"{os.getcwd()}\\Outputs\\Eight_NN\\{name}.pth")  # model_location)
    f = open(f"{os.getcwd()}\\Outputs\\Eight_NN\\Info\\{name}.txt",
             'a')  # getFileName('txt', H, learning_rate, epochs, BATCH_SIZE, smallest_loss), 'a')
    f.write(info)
else:
    f = None
    print(info)

# TODO :: Scale data for ploting and getting accuracy
# with torch.no_grad():
#     x_test = data['CrispLogic']['x']
#     y_test = data['CrispLogic']['y']
#
#     if torch.cuda.is_available():
#         predicted = model(torch.from_numpy(x_test).cuda().float()).cpu()  # .data.numpy()
#     else:
#         predicted = model(torch.from_numpy(x_test).float())  # .data.numpy()
#
# # This is where I Calculate Accuracy
# calculate_Accuracy(predicted, y_test, ROUND_TO, SAVE, f)
#
# # plotting the graph
# plot(predicted, y_test, SAVE_IMG=SAVE,
#      f=f, filename=f"{os.getcwd()}\\Outputs\\Eight_NN\\Info\\{name}.png") #getFileName('png', H, learning_rate, epochs, BATCH_SIZE, smallest_loss))
#
# print(model_location)
# if SAVE:
#     f.close()

preds = []
for _y, _x in zip(data['CrispLogic']['y'], data['CrispLogic']['x']):
    s = MinMaxScaler(feature_range=(0, 1))

    row1 = s.fit_transform(_x.reshape(-1, 1)).reshape(_x.shape)
    # print(row1.shape)
    # print('row1:', row1)

    pred = model(torch.from_numpy(row1).cuda().float()).cpu()
    # print('pred:', pred)

    pred1 = pred.item()
    # print('pred1:', pred1)

    pred2 = np.asarray(pred1).reshape(-1, 1)
    # print('pred2:', pred2)

    pred3 = s.inverse_transform(pred2)
    # print('pred3:', pred3)

    preds.append(pred3.item())

plot(preds, data['CrispLogic']['y'], filename=f"{os.getcwd()}\\Outputs\\Eight_NN\\Info\\{name}.png")
print(preds)

import matplotlib.pyplot as plt

vals = []
trains = []

for v, t in zip(val_loss_rows, train_loss_rows):
    vals.append(val_loss_rows[v]['LOSS'])
    trains.append(train_loss_rows[t]['LOSS'])

plt.plot(vals[100:], label='vals')
plt.plot(trains[100:], label='trains')
plt.legend()
plt.show()
