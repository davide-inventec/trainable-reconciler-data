"""
preprocess australian tourism data
"""

import pandas as pd
import numpy as np
from copy import deepcopy
import os


def build_dates():
    dates = []
    for year in range(1998,2007):
        dates += [
            str(year)+"-01-01",
            str(year)+"-04-01",
            str(year)+"-07-01",
            str(year)+"-10-01",
        ]
    dates = pd.to_datetime(dates)
    return dates

def preprocess(data):
    data = data[data.columns[::-1]]
    dates = build_dates()
    data["Date"] = dates
    data = data.set_index("Date")
    return data

def build_S(data):
    
    S = np.zeros((89,56))
    # bottom level
    for i in range(56): S[i,i] = 1
    # level 1
    for i in range(56,84): 
        for j in range(56):
            if data.columns[i] in data.columns[j]:
                S[i,j] = 1
    # level 2
    S[84,:14] = 1
    S[85,14:28] = 1
    S[86,28:42] = 1
    S[87,42:56] = 1
    # total
    S[88,:] = 1
    return S


if __name__ == "__main__":

    path = "../data/"
    save_path = "../processed_data/australian_tourism/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # load
    data = pd.read_csv(path + "hier1_with_names.csv")
    # preprocess
    data = preprocess(data)

    # split train/test
    train = data[data.index <= pd.to_datetime("2004-10-01")]
    test = data[data.index > pd.to_datetime("2004-10-01")]

    # set the hierarchy matrix
    S = build_S(data)

    # set the hierarchy list
    h_list = [[4],[7,7,7,7],[2]*28]

    # check S
    assert np.abs(data.values[:,:S.shape[1]].dot(S.transpose()) - data.values).max() == 0.0

    # scale (divide by train mean)
    m = train.mean().mean()
    print("overall average",m)
    train /= m
    test /= m
    data /= m

    # save
    data.to_csv(save_path + "data.csv")
    train.to_csv(save_path + "train.csv")
    test.to_csv(save_path + "test.csv")  
    np.save(save_path + "hierarchy_matrix.npy",S)
    pd.to_pickle(h_list,save_path + "hierarchy_list.p")

    print("\n\nAustralian Tourism preprocessed data\n")
    print(train.tail())
    print(test.head())
    print(S)