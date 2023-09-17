import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def remove_default_cols(df):
    data_set = df[df.columns[~df.columns.isin(['jobid', 'user', 'account', 'state', 'submit', 'start', 'end', 'nodelist'])]]
    return data_set
def remove_specific_col(df, col_name):
    data_set = df[df.columns[~df.columns.isin([col_name])]]
    return data_set

def slice_df(df,lo,hi):
    return df.iloc[lo:hi,]

def split_training_test_data(X_historydata, Y_waittimedata):
        X_historydata_train_rm, X_historydata_test_rm, Y_waittimedata_train, Y_waittimedata_test = train_test_split(
        X_historydata, Y_waittimedata, test_size=0.25, random_state=42)
        print("shapes: " + "X train: " + str(X_historydata_train_rm.shape) + " Y train: " + str(Y_waittimedata_train.shape))
        print("shapes: " + "X test: " + str(X_historydata_test_rm.shape) + "Y test: " + str(Y_waittimedata_test.shape))
        return X_historydata_train_rm, X_historydata_test_rm, Y_waittimedata_train, Y_waittimedata_test

def standardization(X_data):
    ### preprocessing Standardization
    scaler = MinMaxScaler()
    print(scaler.fit(X_data))
    print("data max = " + str(scaler.data_max_))
    print("data min = " + str(scaler.data_min_))
    print("data range = " + str(scaler.data_range_))
    print("per feature scale =" + str(scaler.scale_))
    ### Transform the data
    X_data_norm = scaler.transform(X_data)
    return X_data_norm, scaler