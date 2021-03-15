import os
import sys

import numpy as np
import pandas as pd
from sklearn import preprocessing

directory = 'Branch_A/data/'
directory = 'Branch_B/data/'

for filename in os.listdir(directory):
    filename = f"{directory}{filename}"
    data = pd.read_csv(filename, header=None)
    data = data.to_numpy()
    data = preprocessing.scale(data)
    data = pd.DataFrame(data)
    data.to_csv(f"{filename[:-4]}-Std.csv", index=False, header=False)
