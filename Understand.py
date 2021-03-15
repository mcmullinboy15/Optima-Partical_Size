import os
import sys
import csv
import time

import torch
import numpy as np
import pandas as pd
from Test import Test
from Train import Train
from pprint import pprint
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model_and_defs import Model1, Model2, Model3, Model4, separate_train_val_test


def test(filename, data):
    try:
        test_ = Test(filename, data, False)
        test_._test()
        test_.plot(sym_0='', sym_1='')
        test_.save()
    except Exception as e:
        pass
        # print(e)


def mock(filename, data, Model, h1, h2, h3):
    df = pd.read_csv(data)
    data_np = df.to_numpy().astype(dtype='float64')

    data = separate_train_val_test(data_np)
    test = data['val']
    xTest = test['x']
    yTest = test['y'].reshape(-1, 1)

    model = Model(xTest.shape[1], h1, yTest.shape[1], H2=h2, H3=h3)
    print(filename)
    try:
        loaded = torch.load(filename)
    except:
        print('Unable to load model')
    try:
        model_state_dict = loaded['model_state_dict']
    except:
        print('Unable to get model_state_dict')
    try:
        model.load_state_dict(model_state_dict)
    except:
        print('Unable to load model_state_dict')

    model.eval()
    model.double()
    with torch.no_grad():
        plt.ion()

        tensors = torch.from_numpy(xTest)
        preds = model(tensors).double()
        diff = abs(preds - yTest)
        num_correct = len(np.where(diff < OFF)[0])
        counters.append(num_correct)
        print()
        print(*list(range(len(counters))), sep='\t')
        print(*counters, f"length:{len(counters)}", sep='\t')

        if num_correct > 0:  # 10
            save_file = open(f"Outputs/{name}/understand_array.csv", 'a', newline='')
            wr = csv.writer(save_file)
            wr.writerow([filename, num_correct])
            print(f'wrote to file: {save_file}')
            # time.sleep(3)


def loop(NAME, paramaters, data="data/Compiled_Data.csv", skip_done=True):
    for epoch in tqdm(paramaters['EPOCHS']):
        for batch in tqdm(paramaters['BATCHES']):
            for Model in tqdm(paramaters['MODELS']):
                for h1 in paramaters['H1S']:
                    for h2 in paramaters['H2S']:
                        for lr in paramaters['LRS']:
                            for optim in paramaters['OPTIMS']:
                                for h3 in paramaters['H3S']:
                                    filename = f'Outputs/{NAME}/{epoch}/{batch}/{Model.__name__}_{optim.__name__}/{h1}/{h2}/{h3}/{lr}/{NAME}_epoch_{epoch}_batch_{batch}_Model_{Model.__name__}_h1_{h1}_h2_{h2}_h3_{h3}_lr_{lr}_optim_{optim.__name__}.pth'
                                    # test(filename, data)
                                    mock(filename, data, Model, h1, h2, h3)

                                    # init(data, epoch, batch, Model, h1, h2, h3=128, lr=lr, optim=optim, NAME=NAME,
                                    #      skip_done=skip_done, )


if __name__ == '__main__':

    # from Train_Yourself import params
    # name = 'Yourself'

    from Error_nn import params
    name = 'ERROR'

    # from Classifications_nn import params
    # name = 'Classifications'

    build_array = False

    if build_array:
        OFF = 1
        counters = []

        save_file = open(f"Outputs/{name}/understand_array.csv", 'w', newline='')
        wr = csv.writer(save_file)
        wr.writerow(['FileName', f'Correct_By_{OFF}'])
        save_file.close()

        print(name)
        pprint(params)

        loop(name, params)

    else:
        names = ['']
        best = [0]

        understand_file = open(f"Outputs/{name}/understand_array.csv", 'r')
        understand_file.readline()
        for row in understand_file.readlines():
            row = row.split(',')
            filename = row[0]
            num_correct = int(row[1])
            if num_correct > min(best):
                idx = best.index(min(best))
                print(idx)
                if len(best) > 10:
                    names.pop(idx)
                    best.pop(idx)
                names.append(filename)
                best.append(num_correct)
                print(names)
                print(best)

        for file in names:
            print(file)
            loaded_dict = torch.load(file, map_location=torch.device('cpu'))
            model_state_dict = loaded_dict['model_state_dict']
            H1 = loaded_dict['H1']
            H2 = loaded_dict['H2']
            H3 = loaded_dict['H3']
            _in = loaded_dict['_in']
            _out = loaded_dict['_out']
            try:
                MODEL = loaded_dict['MODEL']
            except:
                pass

                Model_name = loaded_dict['Model']
                if Model_name.__contains__('1'):
                    MODEL = Model1
                elif Model_name.__contains__('2'):
                    MODEL = Model2
                elif Model_name.__contains__('3'):
                    MODEL = Model3
                elif Model_name.__contains__('4'):
                    MODEL = Model4
                else:
                    raise Exception(f"{Model_name} is not a real Model Name")

            print(MODEL)
            model = MODEL(_in, H1, _out, H2=H2, H3=H3)
            model.load_state_dict(model_state_dict)
            model.eval()
            # print(model)

            test(file[:-4], data="data/Compiled_Data.csv")


