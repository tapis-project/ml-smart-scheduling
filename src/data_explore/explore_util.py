import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_num_of_zeros(df, col_name):
    numof_zeros = (df[col_name] == 0).sum()
    print("Number of zeros in column: " + col_name + ": " + str(numof_zeros) )
    print("Number of elements in the column: " + str(df.shape[0]))
    numof_non_zeros = df.shape[0] - numof_zeros
    print("Non-zero elements in the column: " + str(numof_non_zeros))
    return numof_zeros,numof_non_zeros

def explore_five_plots(df,col_name_list, num_col, color_list, lo, hi):
    #color_list = []
    #color_list = ["g","r","b","c","m"]
    qm_slice_df = df.iloc[lo:hi,]
    for i in range(0,num_col):
        plt.plot(qm_slice_df[col_name_list[i]], color_list[i])

def statistics_df(y_data):
    # print mean waiting time and standard deviation
    print("mean waiting time = " + str(np.mean(y_data)))
    print("std = " + str(np.std(y_data)))
    print("variance = " + str(np.var(y_data)))
    print("median= " + str(np.median(y_data)))
    print("quantile [0,0.25,0.5, 0.75, 0.9, 0.95, 0.99, 1] = " + str(
        np.quantile(y_data, [0, 0.25, 0.5, 0.75, 0.9, 0.95, 1])))