## Hierarchical time-series - Data download and preprocessing


#### Data download

The data can be downloaded from the following links:

- Dairy: 
    - [Cheddar 40 pounds](https://mpr.datamart.ams.usda.gov/export/Datamart-Export_DY_WK100-40%20Pound%20Block%20Cheddar%20Cheese%20Prices%20and%20Sales_20200915_121947.csv?file=1&fileType=csv)
    - [Butter](https://mpr.datamart.ams.usda.gov/export/Datamart-Export_DY_WK100-Butter%20Prices%20and%20Sales_20200915_122114.csv?file=0&fileType=csv)
    - [Cheddar 500 pounds](https://mpr.datamart.ams.usda.gov/export/Datamart-Export_DY_WK100-500%20Pound%20Barrel%20Cheddar%20Cheese%20Prices,%20Sales,%20and%20Moisture%20Content_20200915_122414.csv?file=2&fileType=csv)
    - [Dry whey](https://mpr.datamart.ams.usda.gov/export/Datamart-Export_DY_WK100-Dry%20Whey%20Prices%20and%20Sales_20200915_122414.csv?file=3&fileType=csv)
    - [Dry milk](https://mpr.datamart.ams.usda.gov/export/Datamart-Export_DY_WK100-Nonfat%20Dry%20Milk%20Prices%20and%20Sales_20200915_122414.csv?file=4&fileType=csv)
- [Australian Tourism](https://robjhyndman.com/data/hier1_with_names.csv)
- Walmart:
    - [sales](https://www.kaggle.com/c/m5-forecasting-accuracy/data?select=sales_train_validation.csv)
    - [calendar](https://www.kaggle.com/c/m5-forecasting-accuracy/data?select=calendar.csv)
- [Energy](https://data.ny.gov/api/views/47km-hhvs/rows.csv?accessType=DOWNLOAD&sorting=true)

Once completed the download, put the files under the `data` directory.

### Data Preprocessing

To preprocess the data, run:

```
bash run_preprocessing.py
```

#### Preprocessing output
For each dataset, the preprocessing outputs: 

- `data.csv` : the hierarchical time series
- `train.csv` : same as `data.csv`, only training period
- `test.csv` : same as `data.csv`, only testing period
- `hierarchy_matrix.npy`: S hierarchy matrix
- `hierarchy_list.p`: used to interface with `R` methods in [hts](https://cran.r-project.org/web/packages/hts/index.html) package, can be safely ignored. It's a list of lists, with each list representing a level in the hierarchy. Each number in the lists represents a node, and its value is equal to the number of children nodes. Values in each level follow a reverse order from the columns in `data.csv`.

For the `Energy` dataset, the ouput also includes the training and testing dataframes and labels used to learn the global forecasting model: `global_tab_data_train.p` and `global_tab_data_test.p`.

