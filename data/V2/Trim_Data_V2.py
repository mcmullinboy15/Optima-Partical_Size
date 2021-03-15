import sys

import pandas as pd
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


def toNAN(df):
    """ Converts all non-numbers to Nan"""
    def is_float(x):
        try:
            return float(x)
        except:
            return None
    return df.apply(lambda x: x.apply(is_float))


def filter_man_pass(df, Passing, Passing_Shifted='Real_Manu_Pass_shifted'):
    """ Does Man Passing change filtering and removes the lines"""

    df[Passing_Shifted] = df[Passing].shift(-1)
    df = df.loc[df[Passing] != df[Passing_Shifted]]
    df.pop(Passing)
    shifted = df.pop(Passing_Shifted)
    df.insert(0, Passing_Shifted, shifted)
    df.drop(df.index[0], inplace=True)
    df = df.iloc[::2]
    return df


def nopen(df, index=13, amount=12):
    """ sums the last `amount` columns and sets the sum to `index` column"""
    # 'Nopen' index is 13

    cols = list(range(len(df.columns) - amount, len(df.columns)))

    # Removing all Nan is the last 12 columns
    for i in reversed(cols):
        df = df.dropna(subset=[i])

    # setting the index column with the sum
    df[index] = [sum([int(row[col]) for col in cols]) for idx, row in df.iterrows()]

    # Removing the last 12 columns
    for i in reversed(cols):
        del df[i]
    return df


def getmaxmin(df):
    """ Gets the min and max of each column """
    minmax = {}
    for col in df.columns:
        _max = max(df[col])
        _min = min(df[col])
        minmax.update({columns[col]: {'min':_min, 'max':_max}})
    return minmax, df


df = pd.read_csv('V2_DataSet_me_trimmed.csv')
# getting headers and setting to numbers
columns = df.columns
df.columns = [i for i in range(len(columns))]

df = toNAN(df)

df = nopen(df, 13)

df = df.dropna()

minmax, df = getmaxmin(df)

# df = filter_man_pass(df, df.columns[0])

print(minmax)
print(df)
df.to_csv('V2_DataSet.csv', index=False, header=None)
