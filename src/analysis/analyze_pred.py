import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def analysis_pred_vs_orig(lo, offset, Y_waittimedata_test, y_pred, nnodes,run_mins,provision_time=20, tolerance_factor=3, print_flag=1):
    off = offset
    #provision_time = 20
    n = tolerance_factor
    hi = lo + off
    result = ({"actual_qtime": Y_waittimedata_test.to_numpy()[lo:lo + off], "predicted_qtime": y_pred[lo:lo + off]})
    rdf = pd.DataFrame(result)
    # rdf.insert(2,"max_minutes",max_mins[lo:lo+off])
    col_index=2
    rdf.insert(col_index, "nnodes", nnodes[lo:lo + off])
    col_index += 1
    rdf.insert(col_index, "run_mins", run_mins[lo:lo + off])
    col_index += 1
    rdf.insert(col_index, "(actual-predicted)qtime", (rdf["actual_qtime"] - rdf["predicted_qtime"]))
    col_index += 1
    ### if prediction is greater than max_mins threshold that is 240 mins or 4 hrs then report
    #action = np.where(rdf["predicted_qtime"] < max_mins_threshold, 0, 1)
    #rdf.insert(5, "action", action)

    ## if the absolute difference between the actual_qtime and predicted_qtime is greater than 120 minutes
    ## report that as misprediction
    #tol = np.where((rdf["(actual-predicted)qtime"]).abs() < 120, 0, 5)
    #rdf.insert(6, "mispredict", tol)

    ## improved %
    improved_time_to_solution = provision_time + rdf["run_mins"]
    true_time_to_solution = rdf["actual_qtime"] + rdf["run_mins"]
    rdf.insert( col_index, "improved%", ((true_time_to_solution - improved_time_to_solution) / true_time_to_solution) * 100.0)
    col_index += 1
    ## if actual_qtime is less than n* provision_time, then the job would not  candidate for dynamic scheduling
    ## 0 indicates 'not a candidate'
    ## 1 indicates ' a candidate'
    have_been_scheduled = np.where(rdf["actual_qtime"] > n * provision_time, 1, 0)
    rdf.insert( col_index, "scheduled?", have_been_scheduled)
    col_index += 1
    ## if predicted_qtime is less than actual_qtime and actual_qtime is greater than
    # conditions=[((rdf["predicted_qtime"] < rdf["actual_qtime"]) & (rdf["actual_qtime"]> n*provision_time))]
    conditions = [((rdf["predicted_qtime"] < n * provision_time) & (rdf["actual_qtime"] > n * provision_time))]
    choice = ['m']
    missed = np.select(conditions, choice, 0)

    rdf.insert( col_index, "missed?", missed)
    col_index += 1



    ### Criteria 5:
    not_scheduled = np.where(rdf["actual_qtime"] <= n * provision_time, 1, 0)
    rdf.insert(col_index, "not_scheduled", not_scheduled)
    col_index += 1

    conditions = [((rdf["predicted_qtime"] > n * provision_time) & (rdf["actual_qtime"] <= n * provision_time))]
    choice = [1]
    wrongly_scheduled = np.select(conditions, choice, 0)

    rdf.insert( col_index, "wrongly_scheduled?", wrongly_scheduled)

    # rdf.insert(8,"reqcpus",reqcpus[lo:lo+200])
    # print("Criteria 1: Actions Required: predicted_queue_time>max_mins threshold (240mins): " + str(rdf['action'].value_counts()))
    # print("Criteria 2: Misprediction:abs(actual_queue_time-predicted_queue_time)>120: " + str(rdf['mispredict'].value_counts()))

    print("Criteria 3: User desires a job to be dynamically scheduled : if actual_qtime> n*provision_time where n="
          + str(n) + " \n provision_time in mins= " + str(provision_time) + ": " + str(
        rdf["scheduled?"].value_counts()))

    print(
        "Criteria 4: Missed opportunity: if (predicted_qtime < n*provision_time) and  actual_qtime > n*provision_time: " + str(
            rdf["missed?"].value_counts()))

    n_scheduled = rdf["scheduled?"].value_counts()
    n_missed = rdf["missed?"].value_counts()
    if n_missed[0] != len(rdf["missed?"]):
        diff = n_scheduled[1] - n_missed[1]
        n_miss = n_missed[1]
    else:
        diff = n_scheduled[1]
        n_miss = 0
    print("Criteria 4: Not Missed opportunity: " + str(n_scheduled[1]) + "-" + str(n_miss) + "=" + str(diff))

    #print("\n Should not be Scheduled : " + str(rdf["not_scheduled"].value_counts()))
    print("\n Criteria 5: Wrongly Scheduled : " + str(rdf["wrongly_scheduled?"].value_counts()))
    #print("\n----------------------------- Detailed Analysis--------------------------------------------------\n")
    ctol = 0
    c = 0

    missed_total = 0
    missed_list = []
    scheduled_total = 0
    scheduled_list = []
    not_missed_list = []
    not_missed_total = 0
    wrongly_scheduled_total = 0
    wrongly_scheduled_list = []
    for i in range(0, 200):
        if rdf.at[i, "missed?"] == 'm':
            missed_total = missed_total + 1
            missed_list.append(i)
        if rdf.at[i, "scheduled?"] == 1:
            scheduled_total = scheduled_total + 1
            scheduled_list.append(i)
        if (rdf.at[i, "scheduled?"] == 1) and (rdf.at[i, "missed?"] == '0'):
            not_missed_total = not_missed_total + 1
            not_missed_list.append(i)
        if (rdf.at[i, "wrongly_scheduled?"] == 1):
            wrongly_scheduled_total = wrongly_scheduled_total + 1
            wrongly_scheduled_list.append(i)
    if print_flag == 1:
            print("\n----------------------------- Detailed Analysis--------------------------------------------------\n")
            #print("\n Rows Values for which action is set to 1 based on Criteria 1:  predicted_qtime>max_mins_threshold (4 hrs) \n")
            #print("\n Improved% is calculated as : ((true_time_to_solution-improved%)/true_time_to_solution)*100.0\n")
            #print("\n true_time_to_solution = actual_qtime + run_minutes\n")
            #print("\n improved% = provision_time + run_minutes \n")
            #print(rdf.loc[la].to_string())
            #print("\n\n Number of misprediction in Criteria 1, i.e., when action is set to one: " + str(ctol)+ "\n")
            #print(rdf.loc[lt5].to_string())
            #print("\n\nNumber of misprediction when action is set to zero: " + str(c)+ "\n")
            #print(rdf.loc[lt50].to_string())
            #print ("\n \n  Number of jobs that are dynamically scheduled with new model: " + str(not_missed_total) + "\n")
            #print(rdf.loc[not_missed_list].to_string())
            print(" \n \n Number of scheduled jobs candidate " + str(scheduled_total) + "\n")
            print(rdf.loc[scheduled_list].to_string())
            print(" \n \n Number of missed opportunities " + str(missed_total) + "\n")
            print(rdf.loc[missed_list].to_string())
            print(" \n \n Wrongly scheduled jobs " + str(wrongly_scheduled_total) + "\n")
            print(rdf.loc[wrongly_scheduled_list].to_string())

def analysis_pred_vs_orig_lstm(lo, offset, Y_waittimedata_test, y_pred, nnodes,provision_time=20, tolerance_factor=3, print_flag=1):
    off = offset
    #provision_time = 20
    n = tolerance_factor
    hi = lo + off
    result = ({"actual_qtime": Y_waittimedata_test[lo:lo + off], "predicted_qtime": y_pred[lo:lo + off]})
    rdf = pd.DataFrame(result)
    # rdf.insert(2,"max_minutes",max_mins[lo:lo+off])
    col_index=2
    rdf.insert(col_index, "nnodes", nnodes[lo:lo + off])
    col_index += 1
    #rdf.insert(col_index, "run_mins", run_mins[lo:lo + off])
    #col_index += 1
    rdf.insert(col_index, "(actual-predicted)qtime", (rdf["actual_qtime"] - rdf["predicted_qtime"]))
    col_index += 1

    ## improved %
    #improved_time_to_solution = provision_time + rdf["run_mins"]
    #true_time_to_solution = rdf["actual_qtime"] + rdf["run_mins"]
    #rdf.insert( col_index, "improved%", ((true_time_to_solution - improved_time_to_solution) / true_time_to_solution) * 100.0)
    #col_index += 1
    ## if actual_qtime is less than n* provision_time, then the job would not  candidate for dynamic scheduling
    ## 0 indicates 'not a candidate'
    ## 1 indicates ' a candidate'
    have_been_scheduled = np.where(rdf["actual_qtime"] > n * provision_time, 1, 0)
    rdf.insert( col_index, "scheduled?", have_been_scheduled)
    col_index += 1
    ## if predicted_qtime is less than actual_qtime and actual_qtime is greater than
    # conditions=[((rdf["predicted_qtime"] < rdf["actual_qtime"]) & (rdf["actual_qtime"]> n*provision_time))]
    conditions = [((rdf["predicted_qtime"] < n * provision_time) & (rdf["actual_qtime"] > n * provision_time))]
    choice = ['m']
    missed = np.select(conditions, choice, 0)

    rdf.insert( col_index, "missed?", missed)
    col_index += 1



    ### Criteria 5:
    not_scheduled = np.where(rdf["actual_qtime"] <= n * provision_time, 1, 0)
    rdf.insert(col_index, "not_scheduled", not_scheduled)
    col_index += 1

    conditions = [((rdf["predicted_qtime"] > n * provision_time) & (rdf["actual_qtime"] <= n * provision_time))]
    choice = [1]
    wrongly_scheduled = np.select(conditions, choice, 0)

    rdf.insert( col_index, "wrongly_scheduled?", wrongly_scheduled)

    # rdf.insert(8,"reqcpus",reqcpus[lo:lo+200])
    # print("Criteria 1: Actions Required: predicted_queue_time>max_mins threshold (240mins): " + str(rdf['action'].value_counts()))
    # print("Criteria 2: Misprediction:abs(actual_queue_time-predicted_queue_time)>120: " + str(rdf['mispredict'].value_counts()))

    print("Criteria 3: User desires a job to be dynamically scheduled : if actual_qtime> n*provision_time where n="
          + str(n) + " \n provision_time in mins= " + str(provision_time) + ": " + str(
        rdf["scheduled?"].value_counts()))

    print(
        "Criteria 4: Missed opportunity: if (predicted_qtime < n*provision_time) and  actual_qtime > n*provision_time: " + str(
            rdf["missed?"].value_counts()))

    n_scheduled = rdf["scheduled?"].value_counts()
    n_missed = rdf["missed?"].value_counts()
    if n_scheduled[0] == offset:
        n_schedule = 0
    else:
        n_schedule=n_scheduled[1]

    if n_missed[0] != offset and n_schedule != 0:
        diff = n_scheduled[1] - n_missed[1]
        n_miss = n_missed[1]
    elif n_missed[0] != offset and (n_schedule ==0):
        diff = 0
        n_miss=0
    elif n_missed[0] == offset and (n_schedule ==0):
        diff = 0
        n_miss = 0
    else:
        diff = n_scheduled[1]
        n_miss = 0
    print("Criteria 4: Not Missed opportunity: " + str(n_schedule) + "-" + str(n_miss) + "=" + str(diff))

    #print("\n Should not be Scheduled : " + str(rdf["not_scheduled"].value_counts()))
    print("\n Criteria 5: Wrongly Scheduled : " + str(rdf["wrongly_scheduled?"].value_counts()))
    #print("\n----------------------------- Detailed Analysis--------------------------------------------------\n")
    ctol = 0
    c = 0

    missed_total = 0
    missed_list = []
    scheduled_total = 0
    scheduled_list = []
    not_missed_list = []
    not_missed_total = 0
    wrongly_scheduled_total = 0
    wrongly_scheduled_list = []
    for i in range(0, 200):
        if rdf.at[i, "missed?"] == 'm':
            missed_total = missed_total + 1
            missed_list.append(i)
        if rdf.at[i, "scheduled?"] == 1:
            scheduled_total = scheduled_total + 1
            scheduled_list.append(i)
        if (rdf.at[i, "scheduled?"] == 1) and (rdf.at[i, "missed?"] == '0'):
            not_missed_total = not_missed_total + 1
            not_missed_list.append(i)
        if (rdf.at[i, "wrongly_scheduled?"] == 1):
            wrongly_scheduled_total = wrongly_scheduled_total + 1
            wrongly_scheduled_list.append(i)
    if print_flag == 1:
            print("\n----------------------------- Detailed Analysis--------------------------------------------------\n")
            print(" \n \n Number of scheduled jobs candidate " + str(scheduled_total) + "\n")
            print(rdf.loc[scheduled_list].to_string())
            print(" \n \n Number of missed opportunities " + str(missed_total) + "\n")
            print(rdf.loc[missed_list].to_string())
            print(" \n \n Wrongly scheduled jobs " + str(wrongly_scheduled_total) + "\n")
            print(rdf.loc[wrongly_scheduled_list].to_string())
def analysis_pred(lo, offset, Y_waittimedata_test, y_pred, nnodes,run_mins,provision_time=20, tolerance_factor=3, print_flag=1):
    off = offset
    #provision_time = 20
    n = tolerance_factor
    hi = lo + off
    result = ({"actual_qtime": Y_waittimedata_test.to_numpy()[lo:lo + off], "predicted_qtime": y_pred[lo:lo + off]})
    rdf = pd.DataFrame(result)
    # rdf.insert(2,"max_minutes",max_mins[lo:lo+off])
    col_index=2
    rdf.insert(col_index, "nnodes", nnodes[lo:lo + off])
    col_index += 1
    rdf.insert(col_index, "run_mins", run_mins[lo:lo + off])
    col_index += 1
    rdf.insert(col_index, "(actual-predicted)qtime", (rdf["actual_qtime"] - rdf["predicted_qtime"]))
    col_index += 1
    ### if prediction is greater than max_mins threshold that is 240 mins or 4 hrs then report
    #action = np.where(rdf["predicted_qtime"] < max_mins_threshold, 0, 1)
    #rdf.insert(5, "action", action)

    ## if the absolute difference between the actual_qtime and predicted_qtime is greater than 120 minutes
    ## report that as misprediction
    #tol = np.where((rdf["(actual-predicted)qtime"]).abs() < 120, 0, 5)
    #rdf.insert(6, "mispredict", tol)

    ## improved %
    improved_time_to_solution = provision_time + rdf["run_mins"]
    true_time_to_solution = rdf["actual_qtime"] + rdf["run_mins"]
    rdf.insert( col_index, "improved%", ((true_time_to_solution - improved_time_to_solution) / true_time_to_solution) * 100.0)
    col_index += 1
    ## if actual_qtime is less than n* provision_time, then the job would not  candidate for dynamic scheduling
    ## 0 indicates 'not a candidate'
    ## 1 indicates ' a candidate'
    have_been_scheduled = np.where(rdf["actual_qtime"] > n * provision_time, 1, 0)
    rdf.insert( col_index, "scheduled?", have_been_scheduled)
    col_index += 1
    ## if predicted_qtime is less than actual_qtime and actual_qtime is greater than
    # conditions=[((rdf["predicted_qtime"] < rdf["actual_qtime"]) & (rdf["actual_qtime"]> n*provision_time))]
    conditions = [((rdf["predicted_qtime"] < n * provision_time) & (rdf["actual_qtime"] > n * provision_time))]
    choice = ['m']
    missed = np.select(conditions, choice, 0)

    rdf.insert( col_index, "missed?", missed)
    col_index += 1



    ### Criteria 5:
    not_scheduled = np.where(rdf["actual_qtime"] <= n * provision_time, 1, 0)
    rdf.insert(col_index, "not_scheduled", not_scheduled)
    col_index += 1

    conditions = [((rdf["predicted_qtime"] > n * provision_time) & (rdf["actual_qtime"] <= n * provision_time))]
    choice = [1]
    wrongly_scheduled = np.select(conditions, choice, 0)

    rdf.insert( col_index, "wrongly_scheduled?", wrongly_scheduled)

    #print("Criteria 3: User desires a job to be dynamically scheduled : if actual_qtime> n*provision_time where n="
    #      + str(n) + " \n provision_time in mins= " + str(provision_time) + ": " + str(
    #    rdf["scheduled?"].value_counts()))

    #print(
    #    "Criteria 4: Missed opportunity: if (predicted_qtime < n*provision_time) and  actual_qtime > n*provision_time: " + str(
    #        rdf["missed?"].value_counts()))

    n_scheduled = rdf["scheduled?"].value_counts()
    n_missed = rdf["missed?"].value_counts()

    if n_scheduled[0] == offset:
        n_schedule = 0
    else:
        n_schedule = n_scheduled[1]
    print(str(type(int(n_missed[0])))+ " " + str(n_missed[0]))
    print(str(type(offset)) +" : "+ str(offset))
    print(str(n_missed[0] != offset))
    if n_missed[0] != offset and n_schedule != 0:
        diff = n_scheduled[1] - n_missed[1]
        n_miss = n_missed[1]
    elif n_missed[0] != offset and (n_schedule ==0):
        diff = 0
        n_miss=0
    elif n_missed[0] == offset and (n_schedule ==0):
        diff = 0
        n_miss = 0
    else:
        diff = n_scheduled[1]
        n_miss = 0

    # if n_missed[0] != len(rdf["missed?"]):
    #     diff = n_scheduled[1] - n_missed[1]
    #     n_miss = n_missed[1]
    # else:
    #     diff = n_scheduled[1]
    #     n_miss = 0
    #print("Criteria 4: Not Missed opportunity: " + str(n_scheduled[1]) + "-" + str(n_miss) + "=" + str(diff))

    #print("\n Should not be Scheduled : " + str(rdf["not_scheduled"].value_counts()))
    #print("\n Criteria 5: Wrongly Scheduled : " + str(rdf["wrongly_scheduled?"].value_counts()))
    return provision_time,n_scheduled, n_miss, diff, n_schedule,offset

def analysis_short(lo, offset, Y_waittimedata_test, y_pred, provision_time=20, tolerance_factor=3):
        off = offset
        # provision_time = 20
        n = tolerance_factor
        hi = lo + off
        result = ({"actual_qtime": Y_waittimedata_test.to_numpy()[lo:lo + off], "predicted_qtime": y_pred[lo:lo + off]})
        rdf = pd.DataFrame(result)
        # rdf.insert(2,"max_minutes",max_mins[lo:lo+off])
        col_index = 2
        #rdf.insert(col_index, "nnodes", nnodes[lo:lo + off])
        #col_index += 1
        #rdf.insert(col_index, "run_mins", run_mins[lo:lo + off])
        #col_index += 1
        rdf.insert(col_index, "(actual-predicted)qtime", (rdf["actual_qtime"] - rdf["predicted_qtime"]))
        col_index += 1
        ### if prediction is greater than max_mins threshold that is 240 mins or 4 hrs then report
        # action = np.where(rdf["predicted_qtime"] < max_mins_threshold, 0, 1)
        # rdf.insert(5, "action", action)

        ## if the absolute difference between the actual_qtime and predicted_qtime is greater than 120 minutes
        ## report that as misprediction
        # tol = np.where((rdf["(actual-predicted)qtime"]).abs() < 120, 0, 5)
        # rdf.insert(6, "mispredict", tol)

        ## improved %
        #improved_time_to_solution = provision_time + rdf["run_mins"]
        #true_time_to_solution = rdf["actual_qtime"] + rdf["run_mins"]
        #rdf.insert(col_index, "improved%",
        #           ((true_time_to_solution - improved_time_to_solution) / true_time_to_solution) * 100.0)
        #col_index += 1
        ## if actual_qtime is less than n* provision_time, then the job would not  candidate for dynamic scheduling
        ## 0 indicates 'not a candidate'
        ## 1 indicates ' a candidate'
        have_been_scheduled = np.where(rdf["actual_qtime"] > n * provision_time, 1, 0)
        rdf.insert(col_index, "ideally_scheduled?", have_been_scheduled)
        col_index += 1
        ## if predicted_qtime is less than actual_qtime and actual_qtime is greater than
        # conditions=[((rdf["predicted_qtime"] < rdf["actual_qtime"]) & (rdf["actual_qtime"]> n*provision_time))]
        conditions = [((rdf["predicted_qtime"] < n * provision_time) & (rdf["actual_qtime"] > n * provision_time))]
        choice = ['m']
        missed = np.select(conditions, choice, 0)

        rdf.insert(col_index, "missed?", missed)
        col_index += 1

        ### Criteria 5:
        not_scheduled = np.where(rdf["actual_qtime"] <= n * provision_time, 1, 0)
        rdf.insert(col_index, "not_scheduled", not_scheduled)
        col_index += 1

        conditions = [((rdf["predicted_qtime"] > n * provision_time) & (rdf["actual_qtime"] <= n * provision_time))]
        choice = [1]
        wrongly_scheduled = np.select(conditions, choice, 0)

        rdf.insert(col_index, "wrongly_scheduled?", wrongly_scheduled)

        # print("Criteria 3: User desires a job to be dynamically scheduled : if actual_qtime> n*provision_time where n="
        #      + str(n) + " \n provision_time in mins= " + str(provision_time) + ": " + str(
        #    rdf["scheduled?"].value_counts()))

        # print(
        #    "Criteria 4: Missed opportunity: if (predicted_qtime < n*provision_time) and  actual_qtime > n*provision_time: " + str(
        #        rdf["missed?"].value_counts()))

        n_scheduled = rdf["ideally_scheduled?"].value_counts()
        n_missed = rdf["missed?"].value_counts()

        if n_scheduled[0] == offset:
            n_schedule = 0
        else:
            n_schedule = n_scheduled[1]
        #print(str(type(int(n_missed[0]))) + " " + str(n_missed[0]))
        #print(str(type(offset)) + " : " + str(offset))
        #print(str(n_missed[0] != offset))
        if n_missed[0] != offset and n_schedule != 0:
            diff = n_scheduled[1] - n_missed[1]
            n_miss = n_missed[1]
        elif n_missed[0] != offset and (n_schedule == 0):
            diff = 0
            n_miss = 0
        elif n_missed[0] == offset and (n_schedule == 0):
            diff = 0
            n_miss = 0
        else:
            diff = n_scheduled[1]
            n_miss = 0

        n_wrongly_scheduled = rdf["wrongly_scheduled?"].value_counts()
        if n_wrongly_scheduled[0] == offset:
            n_wrong = 0
        else:
            n_wrong = n_wrongly_scheduled[1]

        # if n_missed[0] != len(rdf["missed?"]):
        #     diff = n_scheduled[1] - n_missed[1]
        #     n_miss = n_missed[1]
        # else:
        #     diff = n_scheduled[1]
        #     n_miss = 0
        # print("Criteria 4: Not Missed opportunity: " + str(n_scheduled[1]) + "-" + str(n_miss) + "=" + str(diff))

        # print("\n Should not be Scheduled : " + str(rdf["not_scheduled"].value_counts()))
        # print("\n Criteria 5: Wrongly Scheduled : " + str(rdf["wrongly_scheduled?"].value_counts()))

        return provision_time, n_schedule, n_miss, diff, n_wrong, offset
def print_criteria():
    print("User desires a job to be dynamically scheduled : if actual_qtime> n*provision_time\n")
    print("Missed opportunity: if (predicted_qtime < n*provision_time) and  actual_qtime > n*provision_time\n")
    print("Wrongly Scheduled: if (predicted_qtime > n*provision_time) and actual_qtime < n*provision_time \n")
def add_all_to_df(columns_with_name):
    result = (columns_with_name)
    rdf = pd.DataFrame(result)
    return rdf
def add_to_df(past_k,provision_time, num_scheduled, num_missed, not_missed, wrongly_scheduled, total_num,r2_score,maes):
    result = ({"past_k":past_k,"provision_time":provision_time,"num_scheduled":num_scheduled, "num_missed":num_missed,
               "not_missed": not_missed,"wrongly_scheduled":wrongly_scheduled, "total_num":total_num, "r2_score":r2_score,
               "mae":maes})
    rdf = pd.DataFrame(result)

    return rdf
def append_to_df(df,past_k,provision_time, num_scheduled, num_missed, not_missed, wrongly_scheduled, total_num,r2_score,maes):
    result = ({"past_k":past_k,"provision_time":provision_time,"num_scheduled":num_scheduled, "num_missed":num_missed,
               "not_missed": not_missed,"wrongly_scheduled":wrongly_scheduled, "total_num":total_num, "r2_score":r2_score,
               "mae":maes})
    # Append the dictionary to the DataFrame
    df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)

    # Reset the index
    df = df.reset_index(drop=True)

    return df
def add_to_df_nodes(past_k,nodes,provision_time, num_scheduled, num_missed, not_missed, wrongly_scheduled, total_num,r2_score,maes):
    result = ({"past_k":past_k,"nodes":nodes,"provision_time":provision_time,"num_scheduled":num_scheduled, "num_missed":num_missed,
               "not_missed": not_missed,"wrongly_scheduled":wrongly_scheduled, "total_num":total_num, "r2_score":r2_score,
               "mae":maes})
    rdf = pd.DataFrame([result])

    return rdf
def plot_pred_vs_orig(lo, off, Y_waittimedata_test,y_pred):
    hi=lo+off
    fig, axs = plt.subplots(2, 2,figsize=(10, 10))
    fig.suptitle('Horizontally stacked subplots between index: ' + str(lo) + "-" + str(hi+3*off))
    axs[0, 0].plot(Y_waittimedata_test.to_numpy()[lo:hi])
    axs[0, 0].plot(y_pred[lo:hi],'r')
    axs[0, 1].plot(Y_waittimedata_test.to_numpy()[hi:hi+off])
    axs[0, 1].plot(y_pred[hi:hi+off],'r')
    axs[1, 0].plot(Y_waittimedata_test.to_numpy()[hi+off: hi+2*off])
    axs[1, 0].plot(y_pred[hi+off: hi+2*off],'r')
    axs[1, 1].plot(Y_waittimedata_test.to_numpy()[hi+2*off:hi+3*off])
    axs[1,1].plot(y_pred[hi+2*off:hi+3*off],'r')

def plot_pred_vs_orig_lstm(lo, off, Y_waittimedata_test,y_pred):
    hi=lo+off
    fig, axs = plt.subplots(2, 2,figsize=(10, 10))
    fig.suptitle('Horizontally stacked subplots between index: ' + str(lo) + "-" + str(hi+3*off))
    axs[0, 0].plot(Y_waittimedata_test[lo:hi])
    axs[0, 0].plot(y_pred[lo:hi],'r')
    axs[0, 1].plot(Y_waittimedata_test[hi:hi+off])
    axs[0, 1].plot(y_pred[hi:hi+off],'r')
    axs[1, 0].plot(Y_waittimedata_test[hi+off: hi+2*off])
    axs[1, 0].plot(y_pred[hi+off: hi+2*off],'r')
    axs[1, 1].plot(Y_waittimedata_test[hi+2*off:hi+3*off])
    axs[1,1].plot(y_pred[hi+2*off:hi+3*off],'r')

def write_result(filename,result):
    with open(filename, 'a') as f:
        ## result = ({"past_k":past_k,"nodes":nodes,"provision_time":provision_time,"num_scheduled":num_scheduled, "num_missed":num_missed,
        ##       "not_missed": not_missed,"wrongly_scheduled":wrongly_scheduled, "total_num":total_num, "r2_score":r2_score,
        ##       "mae":maes})
        f.write(f"{result['past_k']},{result['nodes']},{result['provision_time']},{result['num_scheduled']},"
                f"{result['num_missed']},{result['not_missed']}, {result['wrongly_scheduled']},{result['total_num']},"
                f"{result['r2_score']},{result['mae']}\n")
        f.flush()
        os.fsync(f.fileno())