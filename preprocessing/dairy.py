"""
preprocess dairy data
"""

import pandas as pd
import numpy as np
import os


def load_clean_data(path, files):
    prods = []
    for i,f in enumerate(files):
        if "40 Pound Block Cheddar Cheese" in f:
            prod = "cheddar_40"
        elif "Dry Whey" in f:
            prod = "whey"
        elif "500 Pound Barrel Cheddar Cheese " in f:
            prod = "cheddar_500"
        elif "Nonfat Dry Milk" in f:
            prod = "milk"
        elif "Butter" in f:
            prod = "butter"
        prods.append(prod)
    
    data = { prod : pd.read_csv(path + files[i]) for i,prod in enumerate(prods)}
 
    for p in data:
        print(p,data[p].shape)

    for k in data:

        # drop duplicates
        data[k] = data[k].drop_duplicates()

        # fix types
        data[k]["Date"] = pd.to_datetime(data[k]["Date"])
        data[k]["Sales"] = [float(s.replace(",","")) for s in data[k]["Sales"]]

        # rename column
        if "Weighted Prices" in data[k].columns:
            data[k] = data[k].rename(columns =  {"Weighted Prices":"Weighted Price"})

        # merge multiple values in same date
        data[k] = data[k].groupby("Date").mean()

        # sort 
        data[k] = data[k].sort_values(by = "Date")
        
        # fix 3 anomalous weeks (0 values, 2013)
        data[k].loc[data[k]["Sales"] == 0,"Sales"] = np.nan
        data[k].loc[data[k]["Weighted Price"] == 0,"Weighted Price"] = np.nan
        data[k] = data[k].fillna(method = "ffill")
        
        # add volume 
        data[k]["Volume"] = data[k]["Weighted Price"]*data[k]["Sales"]
    
    # group at feature level (not product)
    data = {f : pd.DataFrame({k: data[k][f] for k in data}) \
                 for f in ["Sales","Weighted Price","Volume"]}
    
    # add total to Volume
    data["Volume"]["total"] = data["Volume"].values.sum(1)
    
    return data


if __name__ == "__main__":

    path = "../data/"
    save_path = "../processed_data/dairy/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # preprocess
    files = [f for f in os.listdir(path) if "Datamart-Export_DY_WK100" in f]
    print(files)
    data = load_clean_data(path, files)
    data = data["Volume"]["cheddar_500,milk,cheddar_40,whey,butter,total".split(",")]
    data = data[data.index <= pd.to_datetime("2020-06-13")]

    # split train/test
    train = data[data.index <= pd.to_datetime("2019-06-13")]
    test = data[data.index > pd.to_datetime("2019-06-13")]

    # scale (divide by train mean)
    m = train.mean().mean()
    print("overall average",m)
    train /= m
    test /= m
    data /= m

    # set the hierarchy matrix
    S = np.array([
        1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1,
        1,1,1,1,1,
    ]).reshape(-1,5)

    # set the hierarchy list
    h_list = [[5]]

    # save
    data.to_csv(save_path + "data.csv")
    train.to_csv(save_path + "train.csv")
    test.to_csv(save_path + "test.csv")  
    np.save(save_path + "hierarchy_matrix.npy",S)
    pd.to_pickle(h_list,save_path + "hierarchy_list.p")

    print("\n\nDairy preprocessed data\n")
    print(train.tail())
    print(test.head())
    print(S)