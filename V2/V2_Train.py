import random
import pandas as pd
import talib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, MSELoss, L1Loss, NLLLoss2d, Sequential, Conv1d, Conv2d, MaxPool1d, \
    MaxPool2d, Module, Softmax, BatchNorm1d, BatchNorm2d, Dropout
from torch.optim import Adam
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