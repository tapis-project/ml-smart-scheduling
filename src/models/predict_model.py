import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def evaluate(model,scaler,X_test_data, Y_test_data):
    result = model.evaluate(scaler.transform(X_test_data), Y_test_data)
    for i in range(len(result)):
        print(f'{model.metrics_names[i]} — -> {result[i]}')


def predict_and_compare(model,scaler, X_test_data, Y_test_data):
    y_pred = model.predict(scaler.transform(X_test_data))
    print("y_pred shape : " + str(y_pred.shape))
    ## get number of negative numbers
    print("--- inspecting prediction values --- ")
    inspect_negatives_zeros_postives(y_pred)
    print(" --- inspecting original values ---")
    inspect_negatives_zeros_postives(Y_test_data.ravel())
    ## make it positive
    y_pred_clipped = np.where(y_pred<0,0,y_pred)
    print(" ---inspecting prediction values after clipping negative values to zeros --- ")
    inspect_negatives_zeros_postives(y_pred_clipped)
    return y_pred_clipped


def inspect_test_data(X_test_data,Y_test_data,y_pred,low, hi, title):
    print("------------------" + title + "-----------------------")
    print("qm test:" + str(Y_test_data.to_numpy()[low:hi]))
    print("qm pred:" + str(y_pred.ravel()[low:hi]))
    print("X test:" + str(X_test_data.iloc[low:hi,]))
    print("------------------------------------------------------")


def compare_plots_pred_vs_true(Y_test_data, y_pred, lo, hi):
    plt.plot(Y_test_data.to_numpy()[lo:hi])
    plt.plot(y_pred[lo:hi], 'r')

def inspect_negatives_zeros_postives(y_data):
    arr = y_data
    print("number of negative numbers  :" + str(arr[np.where(arr < 0)].size))
    print("number of zeros in the data : " + str(arr[np.where(arr == 0)].size))
    print("number of positive numbers  :" + str(arr[np.where(arr > 0)].size) + "\n")

