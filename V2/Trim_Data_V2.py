import sys

import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)


# pd.set_option('display.max_rows', None)


def toNAN(data):
    """ Converts all non-numbers to Nan"""

    def is_float(x):
        try:
            return float(x)
        except:
            return None

    return data.apply(lambda x: x.apply(is_float))


def filter_man_pass(data, passing, passing_shifted='Real_Manu_Pass_shifted'):
    """ Does Man Passing change filtering and removes the lines"""

    data[passing_shifted] = data[passing].shift(-1)
    data = data.loc[data[passing] != data[passing_shifted]]
    data.pop(passing)
    shifted = data.pop(passing_shifted)
    data.insert(0, passing_shifted, shifted)
    data.drop(data.index[0], inplace=True)
    data = data.iloc[::2]
    return data


def nopen(data, index=13, amount=12):
    """ sums the last `amount` columns and sets the sum to `index` column"""
    # 'Nopen' index is 13

    cols = list(range(len(data.columns) - amount, len(data.columns)))

    # Removing all Nan is the last 12 columns
    for i in reversed(cols):
        data = data.dropna(subset=[i])

    # setting the index column with the sum
    data[index] = [sum([int(row[col]) for col in cols]) for idx, row in data.iterrows()]

    # Removing the last 12 columns
    for i in reversed(cols):
        del data[i]
    return data


def getmaxmin(data):
    """ Gets the min and max of each column """
    _minmax = {}
    for col in data.columns:
        _max = max(data[col])
        _min = min(data[col])
        _minmax.update({columns[col]: {'min': _min, 'max': _max}})
    return _minmax, data


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
