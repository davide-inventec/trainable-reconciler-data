"""
preprocess energy data. Also, prepare tabular daata for global forecasting model.
"""

import pandas as pd
import numpy as np
from copy import deepcopy
import os
from torch.utils.data import DataLoader, Dataset


def preprocess(df,date):
    """
    preprocess raw energy data.
    """
    # create a date column
    df["Date"] = pd.to_datetime(df.year.astype(str) + "-" + df.month.astype(str),format="%Y-%m")

    # select unit
    df = df[(df.unit=="MWh") & (df.data_class == "electricity")]

    # build an ID for bottom level
    df["id"] =  df.full_fips.astype(str) + "---" + \
                df.data_field + "---" + \
                df.uer_id.astype(str) 

    # discard time series with missing values
    tmp = df.groupby("id")[["month"]].count()
    id_to_keep = tmp[tmp.month==48].index.values
    df = df[df.id.isin(id_to_keep)]

    # drop useless columns
    df = df.drop(columns=[
        "data_class",
        "state_2",
        "data_field_display_name",
        "data_stream",
        "number_of_accounts",
        "year",
        "utility_display_name",
        "month",
        "unit",
        "data_field",
        "full_fips",
        "uer_id",
        ])
    # drop absurd values
    df.loc[df.value<0,"value"] = np.nan
    # sort
    df = df.sort_values(by=["id","Date"])
    # convert to column format
    df = pd.DataFrame({ts_id : df.loc[df.id==ts_id,"value"].values \
                           for ts_id in df.id.unique()}, # NB-- unique also sorts!
                      index = df.Date.values[:48])

    # discard trivial time series [always 0/Nans]
    always_0 = df.sum(0)[df.sum(0)==0].index.values
    df = df[[c for c in df.columns if c not in always_0]]

    # fill missing values
    df = df.fillna(method="ffill").fillna(method="bfill")

    # discard time series with low variability FOR SIMPLICITY
    train = df[df.index <= pd.to_datetime(date)]
    norm_scales = np.abs(train.values[1:] - train.values[:-1]).mean(0) / (1 + train.values.mean(0))
    const_col =  df.columns[norm_scales < np.quantile(norm_scales,0.3)] 
    df = df.loc[:,[c for c in df.columns if c not in const_col]]

    return df


def build_hierarchy(df):
    """
    build hierarchy:
    return X, all_values [ts_names for upper levels in the hierarchy], h_list
    """
    n_tseries = sum([np.unique(["---".join(v.split("---")[:j]) \
                                for v in df.columns]).shape[0] for j in range(4)])
    S = np.zeros((n_tseries,df.shape[1]))

    for i in range(df.shape[1]):
        S[i,i] = 1

    levels = [2,1]
    all_values = []
    i = df.shape[1]
    for level in levels:
        values = np.unique(["---".join(col.split("---")[:level]) for col in df.columns])
        all_values.append(values)
        for v in values:
            for j,col in enumerate(df.columns):
                if "---".join(col.split("---")[:level]) == v:
                    S[i,j] = 1
            i += 1

    S[-1,:] = 1
    all_values.append(["total"])
    
    h_list = _build_h_list(df)
    
    return S,all_values,h_list


def _build_h_list(df):
    cols = df.columns.values
    h_list = []
    for level in range(3):
        upper = np.unique(["---".join(c.split("---")[:level]) for c in cols])

        level_list = []
        for u in upper:
            sel_cols = [c for c in cols if u in c]
            v = np.unique([(c.split("---")[level]) for c in sel_cols]).shape[0]
            level_list.append(v)
        h_list.append(level_list)

    # reverse element internally
    h_list = [v[::-1] for v in h_list]

    return h_list



def build_tabular_global(df,scale_n):
    """
    Prepare tab_data format for regression global model
    """
    data = []
    for col in df.columns:
        ts_data = _build_ts_tabular(df,col,scale_n)
        data.append(ts_data)

    data = pd.concat(data)
    
    for c in "ts_name level county consumption_type provider month".split():
        data[c] = data[c].astype("category")
    data = data.sort_values(by=["Date","ts_name"]).reset_index(drop=True)
    return data


def _build_ts_tabular(df,col,scale_n):

    # past and actual values
    data,y = build_X_y_regression(df[col].values,window=12,fill=False)
    
    # scale 
    scale = np.abs(df[col].values[:scale_n][1:] - df[col].values[:scale_n][:-1]).mean()  

    data /= scale
    scale = pd.DataFrame(dict(scale=[scale]*data.shape[0]))
    
    # build categorical features
    cat_feat = _build_cat_features(col)
    cat_feat = pd.concat([pd.DataFrame(cat_feat, index = [i]) for i in range(data.shape[0])])
    # month feature
    month = [v.split("-")[1] for v in df.index.values[1:]]
    month = pd.DataFrame(dict(month=month))
    # ts name
    ts_name = pd.DataFrame(dict(ts_name=[col]*data.shape[0]))
    
    data = pd.DataFrame(data, columns = ["m_" + str(j) for j in range(12,0,-1)])
    data["next_value"] = y

    data = pd.concat([
            ts_name,
            scale,
            cat_feat,
            month,
            data,
        ],
        axis = 1,
    )
    data["Date"] = df.index[1:]
    
    return data


def _build_cat_features(col):
    val = col.split("---")

    feat = dict(
        level = len(val) if val[0] != "total" else "total",
        county = val[0],
        consumption_type = val[1] if len(val) > 1 else "total",
        provider = val[2] if len(val) > 2 else "total",
    )
    
    return feat


def build_X_y_regression(ts,window=8,fill=True):
    """
    Build X,y array for one-step forecasting in a regression fashion.
    N.b: number of columns equal to len(ts)-1, since there are not past values for the first obs.
    If past values not present in a X row and fill=True, they are filled backward (last is always present).

    Parameters:
    ----------
    ts: array with time series
    window: how many past values use for regression (n. of columns in X)
    fill: bool
        if True, fill missing values

    Returns:
    --------
    X: array, row are lagged time series
    y: array, one step ahead time-series
    """
    tsdata = _TSData(ts,window,fill)
    dataloader = DataLoader(tsdata,batch_size = len(tsdata))

    for X,y in dataloader:
        X,y = X.numpy(),y.numpy()
        break

    return X,y


class _TSData(Dataset):
    """ 
    helper class for build_X_y_regression
    """
    def __init__(self,ts,window,fill):
        """
        If past values not present in X and fill is True, 
        fill them backward. Discard first observation
        """
        self.ts = ts
        self.window = window
        self.fill = fill
        
    def __getitem__(self,idx):

        x = self.ts[max(0,idx-self.window+1):idx+1]
        y = self.ts[idx+1]
        
        if len(x) < self.window: 
            if self.fill:
                x = np.concatenate([np.repeat(x[0],self.window-len(x)),x])
            else:
                x = np.concatenate([np.repeat(np.nan,self.window-len(x)),x])
        
        return x,y
    
    def __len__(self):
        return self.ts.shape[0] - 1


if __name__ == "__main__":

    path = "../data/"
    save_path = "../processed_data/energy/"
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # load
    df = pd.read_csv(path + "Utility_Energy_Registry_Monthly_County_Energy_Use__Beginning_2016.csv")
    # preprocess
    df = preprocess(df,"2018-12-01")
    # construct hierarchy
    S,all_values,h_list = build_hierarchy(df)
    col_names = np.concatenate([df.columns.values] + all_values)
    data = pd.DataFrame(df.values.dot(S.transpose()),columns=col_names, index=pd.Series(df.index,name="Date"))

    # split train/test
    train = data[data.index <= pd.to_datetime("2018-12-01")]
    test = data[data.index > pd.to_datetime("2018-12-01")]

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

    print("\n\nEnergy preprocessed data\n")
    print(train.tail())
    print(test.head())
    print(S)
    print(h_list)

    ### prepare tab_data for regression (global model)
    train = pd.read_csv(save_path + "train.csv").set_index("Date") 
    train_test = pd.read_csv(save_path + "data.csv").set_index("Date")

    # train
    tab_data_train = build_tabular_global(train,scale_n=train.shape[0])
    cat_features = ['level','county','consumption_type','provider','month','ts_name']
    # test
    tab_data_all = build_tabular_global(train_test,scale_n=train.shape[0])
    tab_data_test = tab_data_all[~tab_data_all.Date.isin(tab_data_train.Date)].reset_index(drop=True)
    # store
    pd.to_pickle((tab_data_train,cat_features),save_path + "global_tab_data_train.p")
    pd.to_pickle(tab_data_test,save_path + "global_tab_data_test.p")

    print("\n\n Tabular data for regression - global model")
    print(tab_data_train.tail())


