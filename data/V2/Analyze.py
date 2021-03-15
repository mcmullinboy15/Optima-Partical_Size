import sys

import talib
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)



def manipulate(df, COL, MINMAX=True, MIN=None, MAX=None, SORT=True, PLOT=True, PRINT=False):
    if PRINT:
        print(df[COL])

    # drop all NaN values
    df = df.dropna(how='any', subset=[COL])
    length = len(df)

    if MINMAX:
        # Filter Values outside of MIN and MAX
        if MIN is None and MAX is not None:
            df = df[(df[COL] <= MAX)]
        elif MAX is None and MIN is not None:
            df = df[(df[COL] >= MIN)]
        elif MAX is None and MIN is None:
            df = df
        else:
            df = df[ (df[COL] >= MIN) & (df[COL] <= MAX) ]
        if PRINT:
            print('MIN MAX:\n', df)

    if SORT:
        # Sort the Dataframe by that Column
        df = df.sort_values(by=[COL], ignore_index=True)
        if PRINT:
            print('Sorted:\n', df[COL])

    if PLOT:
        # plot the ordered column to see the outliers
        plt.plot(df[COL], 'go', label=COL)
        plt.legend()
        plt.show()

    removed_rows = length - len(df)
    print(COL, removed_rows)
    return df


def mov_ave(df, col, ave):
    """ Replaces the col with the moving average
    Then removes NaN """
    """ Be carefull, if you loop through the columns and 
    remove the mov ave every time it removes to much"""

    print('df:', len(df))
    df[col] = talib.SMA(df[col], timeperiod=ave)
    df = df.dropna(how='any', subset=[col])
    print('df_:', len(df))
    return df


df = pd.read_csv('V2_3_DataSet.csv', header=None)
cols = [
    # (0, 55, 85),
    (1, 4650, 5250),
    (2, 275, 850), # 751
    (3, 7.5, 17.5),
    (4, 48, None),
    (5, 650, None),
    (6, 99, 650),
    (7, 2750, 10000),
    # (8, 24, 44),
    # (9, 250, None),
    # (10, 0, None),
    # (11, 1950, None),
    # (12, 65, None),
    # (13, 6, 10),
]
for idx, _min, _max in cols:
    print(idx, _min, _max)
    manipulate(df=df, COL=idx, MINMAX=True, MIN=_min, MAX=_max, SORT=True, PLOT=True, PRINT=True)
    manipulate(df=df, COL=idx, MINMAX=True, MIN=_min, MAX=_max, SORT=False, PLOT=True, PRINT=True)
    manipulate(df=df, COL=idx, MINMAX=False, MIN=_min, MAX=_max, SORT=False, PLOT=True, PRINT=True)



df.to_csv('V2_DataSet_minmax.csv', index=None, header=None)
