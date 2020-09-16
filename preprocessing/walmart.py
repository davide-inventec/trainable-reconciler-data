"""
preprocess Walmart data
"""

import pandas as pd
import numpy as np
from copy import deepcopy
import os


def select_item(df):
    
    tmp_df = deepcopy(df)
    
    tmp_df["tmp"] = np.quantile(tmp_df.drop(columns = "id item_id dept_id cat_id store_id state_id".split()).values, 0.1, axis = 1)
    tmp = tmp_df.groupby("item_id")["tmp"].min().sort_values(ascending=False)[:1]
    item = tmp.index[0]
    print("item_id with highest average 10% quantile:",item,"value:",tmp[0])
    return item


def prepare_data(df,dates,item):
    
    df = df[df.item_id == item].drop(columns = "id item_id dept_id cat_id".split()).reset_index(drop=True)

    # remove anomalous store for SIMPLICITY
    df = df[df.store_id!="CA_2"].reset_index(drop=True)

    # prepare total
    tot = df.drop(columns = ["store_id"]).sum()
    tot["state_id"] = "total"
    tot = pd.DataFrame(tot).transpose().rename(columns = {"state_id":"id"})

    # prepare country
    country = df.groupby("state_id").sum().reset_index().rename(columns = {"state_id":"id"})

    # prepare store
    store = df.drop(columns = "state_id").rename(columns = {"store_id":"id"})

    # concat
    df = pd.concat([store,country,tot], axis = 0)

    cols = df["id"].values

    df = df.transpose().reset_index(drop=True)
    df.columns = cols
    df = df.loc[1:]
    df["Date"] = dates
    df = df.set_index("Date")
    return df



if __name__ == "__main__":

    path = "../data/"
    save_path = "../processed_data/walmart/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # load
    dates = pd.to_datetime(pd.read_csv(path + "calendar.csv").date[:1941].values)
    df = pd.read_csv(path + "sales_train_evaluation.csv")

    # preprocess
    item = select_item(df) # extract a single item
    data = prepare_data(df, dates, item)

    # split train/test
    train = data[data.index <= pd.to_datetime("2015-05-22")]
    test = data[data.index > pd.to_datetime("2015-05-22")]

    # set the hierarchy matrix
    S = np.zeros((13,9))
    for i in range(9): S[i,i] = 1
    S[9,:3] = 1
    S[10,3:6] = 1
    S[11,6:9] = 1
    S[12,:] = 1

    # set the hierarchy list
    h_list = [[3],[3,3,3]]

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

    print("\n\nWalmart preprocessed data\n")
    print(train.tail())
    print(test.head())
    print(S)