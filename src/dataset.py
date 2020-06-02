import wfdb
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd



def load_data():
    # Create Traiining dataset
    train = '../afpdbCSV/'
    # test = 'test/'
    x_data = np.array([])
    y_data = np.array([])
    records = [f for f in listdir(train) if isfile(join(train, f)) if(f.find('.csv') != -1)]
    records.sort() 
    records = records[0:200] # exclude all the tXX.csv files
    print(records)
    counter = 0
    for r in records:
        if r[0] == 'n':
            y_data = np.append(y_data,0)
        elif r[0] =='p':
            y_data = np.append(y_data,1)

        test = pd.read_csv(train + r,index_col=0)
        test = test.values.tolist() # --> convert test dataframe to list
        x = np.array([])
        for t in test:
            # print(t[1])
            x = np.append(x,t[0])
        # print(x)
        x_data = np.append(x_data,x)
        counter = counter + 1
    x_data = x_data.reshape(-1, 1280)
    y_data = y_data.astype(int)
    # ==================================================================================
    # Create testing
    # testing = 'test/'
    # x_test = np.array([])
    # y_test = np.array([])
    # records = [f for f in listdir(testing) if isfile(join(testing, f)) if(f.find('.csv') != -1)]
    # records.sort() 
    # counter = 0
    # for r in records:
    #     if r[0] == 'n':
    #         y_test = np.append(y_test,0)
    #     elif r[0] =='p':
    #         y_test = np.append(y_test,1)

    #     test = pd.read_csv(testing + r,index_col=0)
    #     test = test.values.tolist() # --> convert test dataframe to list
    #     x = np.array([])
    #     for t in test:
    #         # print(t[1])
    #         x = np.append(x,t[0])
    #     # print(x)
    #     x_test = np.append(x_test,x)
    #     counter = counter + 1
    # x_test = x_test.reshape(-1, 1280)
    # y_test = y_test.astype(int)

    return x_data,y_data


# x_data,y_data,x_test,y_test = load_data()
# print(x_data.shape)
# print(y_data.shape)
# print(x_test.shape)
# print(y_test.shape)

# np.save('x_data.npy', x_data)
# np.save('y_data.npy', y_data)
# np.save('x_test.npy', x_test)
# np.save('y_test.npy', y_test)



