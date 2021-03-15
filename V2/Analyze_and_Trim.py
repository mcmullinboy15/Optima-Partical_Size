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



cautious_cols = [
    (0, 55, 85),
    (1, 4650, 5250),
    (2, 275, 850),
    (3, 7.5, 17.5),
    (4, 48, None),
    # (5, 650, 2900),
    # (6, 99, 650),
    (7, 2750, 10000),
    # (8, 24, 44),
    (9, 250, None),
    (10, 0, None),
    (11, 1950, None),
    (12, 65, None),
    (13, 6, 10),
]
tight_cols = [
    (0, 60, 80),
    (1, 4800, 5200),
    (2, 295, 760),
    (3, 9, 16),
    (4, 50, None),
    # (5, 800, 2600),
    # (6, 28, 43),
    (7, 4300, 9700),
    # (8, 24, 44),
    (9, 280, 660),
    (10, 0, 66),
    (11, 1950, None),
    (12, 68, None),
    (13, 6, 10),
]

cols = [
    (0, None, None), 
    (1, None, None), 
    (2, None, None), 
    (3, None, None), 
    (4, None, None), 
    # (5, None, None),
    # (6, None, None),
    (7, None, None), 
    # (8, None, None),
    (9, None, None), 
    (10, None, None), 
    (11, None, None), 
    (12, None, None), 
    (13, None, None), 
]

df = pd.read_csv('V2_4_DataSet_minmax.csv', header=None)

# cols = cautious_cols
cols = tight_cols
# cols = cols

print(df)
df.pop(5)
df.pop(6)
df.pop(8)
print(df)

for idx, _min, _max in cols:
    print(idx, _min, _max)
    df = manipulate(df=df, COL=idx, MINMAX=True, MIN=_min, MAX=_max, SORT=False, PLOT=False, PRINT=True)
    # manipulate(df=df, COL=idx, MINMAX=True, MIN=_min, MAX=_max, SORT=False, PLOT=True, PRINT=True)
    # manipulate(df=df, COL=idx, MINMAX=False, MIN=_min, MAX=_max, SORT=False, PLOT=True, PRINT=True)

print(len(df))
# I used this to create the branch datasets,
# Branch_A = remove 5 and 6
# Branch_B = remove 5, 6, and 8
df.to_csv(f'Branch_B/data/V2_DataSet_B_Tight-{len(df)}.csv', index=None, header=None)
