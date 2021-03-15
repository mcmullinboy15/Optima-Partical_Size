"""https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/"""
# importing the libraries
import random
from _datetime import datetime

import pandas as pd
import numpy as np

# for reading and displaying images
import talib
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, MSELoss, L1Loss, NLLLoss2d, Sequential, Conv1d, Conv2d, MaxPool1d, \
    MaxPool2d, Module, Softmax, BatchNorm1d, BatchNorm2d, Dropout
from torch.optim import Adam, SGD

# my Imports
import model_and_defs as helpdesk


class Model4Train:
    def __init__(self, epochs=50, h1=128, lr=0.07, data_file="data/Compiled_Data.csv", save=True):
        self.data_file = data_file
        self.h1 = h1
        self.filename = f'Outputs/{__class__.__name__}/{epochs}_{h1}_{lr}'
        self.lr = lr
        self.MODEL = helpdesk.Model4

        # saving the model .pth and the image .png
        self.SAVE = save
        # defining the number of epochs
        self.epochs = epochs
        # empty list to store training losses
        self.train_losses = []
        # empty list to store validation losses
        self.val_losses = []
        # plotION default off
        self.plotion = False

        # loading dataset
        data = pd.read_csv(data_file).to_numpy()
        random.shuffle(data)
        train_x = data[:, 1:]
        train_y = data[:, 0].reshape((-1, 1))

        # create validation set
        train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
        print((train_x.shape, train_y.shape), (val_x.shape, val_y.shape))

        # converting into torch format
        self.train_x = torch.from_numpy(train_x)
        self.train_y = torch.from_numpy(train_y)
        self.val_x = torch.from_numpy(val_x)
        self.val_y = torch.from_numpy(val_y)

        self._in = self.train_x.shape[1]
        self._out = self.train_y.shape[1]

        # defining the model
        print('Build Model')
        model = helpdesk.Model4(_in=self._in, H=h1, _out=self._out)
        model.double()
        # defining the optimizer
        optimizer = Adam(model.parameters(), lr=lr)
        # defining the loss function
        criterion = MSELoss()
        # checking if GPU is available
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

        self.info = str(self)
        print(model)

    def train(self, epoch):
        self.model.train(True)
        # getting the training set
        x_train, y_train = Variable(self.train_x), Variable(self.train_y)
        # getting the validation set
        x_val, y_val = Variable(self.val_x), Variable(self.val_y)
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        self.optimizer.zero_grad()

        # prediction for training and validation set
        output_train = self.model(x_train)
        self.model.train(False)
        output_val = self.model(x_val)

        # computing the training and validation loss
        loss_train = self.criterion(output_train, y_train)
        loss_val = self.criterion(output_val, y_val)
        self.train_losses.append(loss_train)
        self.val_losses.append(loss_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        self.optimizer.step()

        # printing the validation loss
        print(f'\rEpoch : {epoch + 1},\t{round(output_val[0].item(), 3)} = model({round(y_val[0].item(), 3)})'
              f'\tloss : {round(loss_val.item(), 3)}', end="")

        if self.plotion:
            self.plotION(epoch)

        return loss_val.item()

    # TODO separate to ['train', 'val']
    def run(self):
        # training the model
        print('Training Model')
        for epoch_ in range(self.epochs):
            loss = self.train(epoch_)
            if loss < 18:
                break

        self.finish()

    def finish(self):

        self.save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'criterion_state_dict': self.criterion.state_dict(),
            'data_file': self.data_file,
            'filename': self.filename,
            'MODEL': self.MODEL,
            '_in': self._in,
            '_out': self._out,
            'H1': self.h1,
            # 'H2': self.H2,
            # 'H3': self.H3,
            'epochs': self.epochs,
            # 'BATCH_SIZE': self.BATCH_SIZE,
            'learning_rate': self.lr,
            # 'ave_loss': self.ave_loss,
        }

        if self.SAVE:
            self.save_model()

    def __str__(self):

        return f"" \
               f"\nData Used (df):\n" \
               f"{self.data_file}\n" \
               f"\nModel Used (model): \n" \
               f"{self.model}\n" \
               f"model saved to {self.filename}\n" \
               f"Loss: {self.val_losses}\n"

    def save_model(self):

        torch.save(self.save_dict, f"{self.filename}.pth")
        f = open(f'{self.filename}.txt', 'a+')
        f.write(self.info)
        f.close()
        print(self.info)
        print()
        print(f'{self.filename}.pth')

    def plotION(self, i=None, size=50, flow=True, diff=False):
        if not self.plotion:
            self.plotion = True
            return

        plt.ion()

        train_ = self.train_losses
        train_ = talib.SMA(np.array(self.train_losses, dtype='double'), timeperiod=10)
        val_ = self.val_losses
        val_ = talib.SMA(np.array(self.val_losses, dtype='double'), timeperiod=10)

        if i > size:
            if flow:
                # just the last var(size_plotted)
                train_ = train_[-size:]
                val_ = val_[-size:]
            else:
                # get rid of the first  var(size_plotted)
                train_ = train_[size:]
                val_ = val_[size:]

        if diff:
            diffs = [va - tr for tr, va in zip(train_, val_)]
            plt.plot(diffs, label='Difference in losses')

        else:
            plt.plot(train_, label='Training loss')
            plt.plot(val_, label='Validation loss')

        plt.legend()

        plt.draw()
        plt.pause(0.0001)
        plt.clf()

    def plot(self):
        # plotting the training and validation loss
        plt.plot(self.train_losses, label='Training loss')
        plt.plot(self.val_losses, label='Validation loss')
        plt.legend()
        plt.show()

    def run_test(self):
        from Test import Test

        print(f'Get {self.filename}.pth to run Tests')
        if self.SAVE:
            test_ = Test(f'{self.filename}')
        else:
            test_ = Test(loaded_dict=self.save_dict)
        try:
            test_._test()
            test_.plot(sym_0='', sym_1='')
            if self.SAVE:
                test_.save()
        except Exception as e:
            print(f"\033[91m {e}\033[00m")


train = Model4Train(epochs=100, )#lr=1e-4)
train.plotION(flow=False)
train.run()
train.plot()
train.run_test()
