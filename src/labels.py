import pandas as pd
import numpy as np
from os import listdir, mkdir, system
from os.path import isfile, isdir, join, exists

def get_labels():
    dir = '../afpdbCSV'
    classes = ['Normal','Atrial']
    labels = np.array([])
    records = [f for f in listdir(dir) if isfile(join(dir, f)) if(f.find('.csv') != -1)]
    records.sort()
    # records = records[0:199]
    for r in records:
        if r[0] == 'n':
            labels = np.append(labels,0)
        elif r[0] =='p':
            labels = np.append(labels,1)

    labels = labels.astype(int)

    return labels

# y = get_labels()
# print(y)
# print(y.shape)