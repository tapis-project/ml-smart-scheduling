import pandas as pd
import numpy as np
import csv


def read_data(csv_file_name, parse_dates_col):
    df = pd.read_csv(csv_file_name,parse_dates=parse_dates_col)
    return df

def add_past_k_obs_insert(past_k_obs, df,col_name, original_col_off,col_obs_off):
    PAST_K_OBS = past_k_obs
    COLUMN_OFFSET = original_col_off
    ## Length of queue_minutes array
    QM_LEN = len(df[col_name])
    print("QM_LEN" + str(QM_LEN))
    l = []

    for c in range(col_obs_off, PAST_K_OBS):
        print("c="+ str(c))
        new = []
        for i in range(0,PAST_K_OBS):
            new.append(0)
        for r in range(PAST_K_OBS, QM_LEN):
            no_qm = True
            #print("r = " + str(r))
            for i in range(r,-1,-1):
                #print("i="+ str(i))
                if df.at[i-c-1,"start"] < df.at[r-c,"submit"]:
                    #print( " start time at row (i-c-1): " + str(i-c-1)+ "  submit time at row (r-c): " + str(r-c))
                    #print("start: " + str(df.at[i-c-1,"start"]))
                    #print("submit: " + str(df.at[r-c,"submit"]))
                    #print("queue minute added: " + str(df.at[i-c-1,col_name]))
                    new.append(df.at[i-c-1,col_name])
                    no_qm = False
                    break;
            if no_qm:
                new.append(0)
        name = "qm"+str(c+1)
        df.insert(COLUMN_OFFSET+c,name,new)
    return df


def add_past_k_completed_obs_insert(past_k_obs, df, col_name, original_col_off, col_obs_off, col_index):
    PAST_K_OBS = past_k_obs
    COLUMN_OFFSET = original_col_off
    ## Length of queue_minutes array
    QM_LEN = len(df[col_name])
    print("QM_LEN" + str(QM_LEN))
    l = []
    k = 0
    for c in range(col_obs_off, PAST_K_OBS):
        print("c=" + str(c))
        new = []
        for i in range(0, PAST_K_OBS):
            new.append(0)
        for r in range(PAST_K_OBS, QM_LEN):
            no_qm = True
            # print("r = " + str(r))
            for i in range(r, -1, -1):
                # print("i="+ str(i))
                if df.at[i - c - 1, "start"] < df.at[r - c, "submit"]:
                    # print( " start time at row (i-c-1): " + str(i-c-1)+ "  submit time at row (r-c): " + str(r-c))
                    # print("start: " + str(df.at[i-c-1,"start"]))
                    # print("submit: " + str(df.at[r-c,"submit"]))
                    # print("queue minute added: " + str(df.at[i-c-1,col_name]))
                    new.append(df.at[i - c - 1, col_name])
                    no_qm = False
                    break;
            if no_qm:
                new.append(0)
        name = "qm" + str(c + 1)

        col_idx = col_index + k
        df.insert(col_idx, name, new)
        k = k + 1
    return df