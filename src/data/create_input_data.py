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
        print("------ column c="+ str(c))
        new = []
        for j in range(0,PAST_K_OBS):
            new.append(0)
        for r in range(PAST_K_OBS, QM_LEN):
            no_qm = True
            c_pbs = 0
            #print("row r = " + str(r))
            for i in range(r,-1,-1):
                #print("i="+ str(i))
                if df.at[i, "start"] < df.at[r, "submit"]:
                    c_pbs = c_pbs + 1
                    #print( " start time at row (i): " + str(i)+ "  submit time at row (r): " + str(r))
                    #print("start: " + str(df.at[i,"start"]))
                    #print("submit: " + str(df.at[r,"submit"]))
                    #print("count_pbs: " + str(c_pbs))
                        #if df.at[i-c-1,"start"] < df.at[r-c,"submit"]:
                        #print( " start time at row (i-c-1): " + str(i-c-1)+ "  submit time at row (r-c): " + str(r-c))
                        #print("start: " + str(df.at[i-c-1,"start"]))
                        #print("submit: " + str(df.at[r-c,"submit"]))
                        #print("queue minute added: " + str(df.at[i-c-1,col_name]))
                    if (c_pbs == (c+1)):
                        # new.append(df.at[i - c - 1, col_name])
                        #print("queue minute added: " + str(df.at[i, col_name]))
                        new.append(df.at[i, col_name])
                        if df.at[i, col_name] == 452:
                            print( " start time at row (i): " + str(i)+ "  submit time at row (r): " + str(r))
                            print("start: " + str(df.at[i,"start"]))
                            print("submit: " + str(df.at[r,"submit"]))
                            print("count_pbs: " + str(c_pbs))
                            print("queue minute added: " + str(df.at[i, col_name]))
                        no_qm = False
                        break;
                    #new.append(df.at[i-c-1,col_name])
                    #no_qm = False
                    #break;
            if no_qm:
                #print("no completed qm found, so adding 0")
                new.append(0)
        name = "qm"+str(c+1)
        df.insert(COLUMN_OFFSET+c,name,new)
    return df

def input_data_lstm(lookback, df):
    col_len = df.shape[0]
    lstm_input = []
    col_names = list(df)
    num_cols = len(col_names)
    rows_to_discard = 0
    for row in range(lookback,col_len):
        current_rows_to_discard = 0
        no_qm = True
        lookback_list = []
        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list
        for col_name in col_names:
            if col_name not in ['start', 'submit', 'end']:
                lookback_list.append([])
        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            # print("i="+ str(i))
            if df.at[i, "start"] < df.at[row, "submit"]:
                #print("start: " + str(df.at[i, "start"]))
                count_lookback = count_lookback + 1
                k = 0
                for col_name in col_names:

                    if col_name not in ['start','submit','end']:
                        #print("getting the column name:" + col_name + " value=" + str(df.at[i, col_name]))
                        lookback_list[k].append(df.at[i, col_name])
                        k = k+1
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback count never equals to  lookback. so do not consider the row " + str(current_rows_to_discard))

            rows_to_discard = current_rows_to_discard
        if current_rows_to_discard != 0:
            continue
        else:
            lstm_input.append(lookback_list)
    lstm_input_array = np.array(lstm_input)
    ## rows,lookback timesteps, features
    transposed_lstm = np.swapaxes(lstm_input_array, 1, 2)
    return transposed_lstm, rows_to_discard
def input_data_lstm_array(lookback, df,standardized_array):
    if not(df.shape[0] == standardized_array.shape[0]) :
      #print("shape does not match dataframe:"+ str(df.shape) + "  standardized array=" + str(standardized_array))
      raise Exception("shape does not match dataframe:"+ str(df.shape) + "  standardized array=" + str(standardized_array.shape))
    col_len = df.shape[0]
    lstm_input = []
    col_names = list(df)
    num_cols = len(col_names)
    rows_to_discard = 0
    for row in range(lookback,col_len):
        current_rows_to_discard = 0
        no_qm = True
        lookback_list = []
        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list
        for col_name in col_names:
            if col_name not in ['start', 'submit', 'end']:
                lookback_list.append([])
        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            # print("i="+ str(i))
            #iloc[0, lstm_df.columns.get_loc("submit")]
            #if df.at[i, "start"] < df.at[row, "submit"]:
            if df.iloc[i, df.columns.get_loc("start")] < df.iloc[row, df.columns.get_loc("submit")]:
                #print("start: " + str(df.at[i, "start"]))
                count_lookback = count_lookback + 1
                k = 0
                for col_name in col_names:

                    if col_name not in ['start','submit','end']:
                        #print("getting the column name:" + col_name + " value=" + str(df.at[i, col_name]))
                        #lookback_list[k].append(df.at[i, col_name])
                        lookback_list[k].append(standardized_array[i, k])
                        k = k+1
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback count never equals to  lookback. so do not consider the row " + str(current_rows_to_discard))

            rows_to_discard = current_rows_to_discard
        if current_rows_to_discard != 0:
            continue
        else:
            lstm_input.append(lookback_list)
    lstm_input_array = np.array(lstm_input)
    ## rows,lookback timesteps, features
    transposed_lstm = np.swapaxes(lstm_input_array, 1, 2)
    return transposed_lstm, rows_to_discard
def save_lstm_input(lstm_input_3d_array,lstm_input_txt_file_name):
    # Write the array to disk
    with open(lstm_input_txt_file_name, 'w') as outfile:
        # I'm writing a header here just for the sake of readability
        # Any line starting with "#" will be ignored by numpy.loadtxt
        outfile.write('# Array shape: {0}\n'.format(lstm_input_3d_array.shape))

        # Iterating through a ndimensional array produces slices along
        # the last axis. This is equivalent to data[i,:,:] in this case
        for data_slice in lstm_input_3d_array:
            # The formatting string indicates that I'm writing out
            # the values in left-justified columns 7 characters in width
            # with 2 decimal places.
            np.savetxt(outfile, data_slice, fmt='%-7.2f')

            # Writing out a break to indicate different slices...
            outfile.write('# New slice\n')

def optimised_add_past_k_obs_insert(lookback, df,col_name):
    ## Length of queue_minutes array
    col_len = len(df[col_name])
    rows_to_discard = 0
    lookback_list = {}
    for i in range(0, lookback):
        lookback_list['qm' + str(i + 1)] = []
    #for row in range(0,lookback):
    #    for k in range(0,lookback):
    #        lookback_list['qm' + str(k+1)].append(-200)

    #    print("appending -200 to row :" + str(row))
    #for row in range(lookback, col_len):
    for row in range(0, col_len):
        current_rows_to_discard = 0
        no_qm = True

        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list


        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            #print("i="+ str(i))
            if df.at[i, "start"] < df.at[row, "submit"]:
                #print("start: " + str(df.at[i, "start"]))

                count_lookback = count_lookback + 1
                #print("value at i="+ str(i)+ "= "+ str(df.at[i, col_name]))
                lookback_list['qm'+str(count_lookback)].append(df.at[i, col_name])
                #print("loopback dictionary after append: " + str(lookback_list))
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback_count never equals to  lookback. so do not consider the row " + str(
            current_rows_to_discard))
            rows_to_discard = current_rows_to_discard
            for k in range(count_lookback+1,lookback+1):
                lookback_list['qm' + str(k)].append(-200)
                #print("appending -200 to row :"+str(k))
    #print("loopback dictionary: " + str(lookback_list))
    #print("length: qm1 " +str(len(lookback_list['qm1'])))
    rdf=pd.DataFrame(lookback_list)
    #print(rdf.to_string())
    rdf_con = pd.concat([df, rdf],axis=1)
    return rdf_con, rows_to_discard

def optimised_add_cols_past_k_obs_insert(lookback, df,col_name,insert_col):
    ## Length of queue_minutes array
    col_len = len(df[col_name])
    rows_to_discard = 0
    lookback_list = {}
    for i in range(0, lookback):
        lookback_list['qm' + str(i + 1)] = []
    #for row in range(0,lookback):
    #    for k in range(0,lookback):
    #        lookback_list['qm' + str(k+1)].append(-200)

    #    print("appending -200 to row :" + str(row))
    #for row in range(lookback, col_len):
    for row in range(0, col_len):
        current_rows_to_discard = 0
        no_qm = True

        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list


        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            #print("i="+ str(i))
            if df.at[i, "start"] < df.at[row, "submit"]:
                #print("start: " + str(df.at[i, "start"]))

                count_lookback = count_lookback + 1
                #print("value at i="+ str(i)+ "= "+ str(df.at[i, col_name]))
                lookback_list['qm'+str(count_lookback)].append(df.at[i, col_name])
                #print("loopback dictionary after append: " + str(lookback_list))
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback_count never equals to  lookback. so do not consider the row " + str(
            current_rows_to_discard))
            rows_to_discard = current_rows_to_discard
            for k in range(count_lookback+1,lookback+1):
                lookback_list['qm' + str(k)].append(-200)
                #print("appending -200 to row :"+str(k))
    #print("loopback dictionary: " + str(lookback_list))
    #print("length: qm1 " +str(len(lookback_list['qm1'])))
    rdf=pd.DataFrame(lookback_list)
    #print(rdf.to_string())
    rdf_con = pd.concat([df, rdf],axis=1)
    return rdf_con, rows_to_discard

### Add the age of the waiting jobs and the number of nodes requested to the feature vectors
def optimised_add_age_past_k_obs_insert(lookback, df,col_name):
    ## Length of queue_minutes array
    col_len = len(df[col_name])
    rows_to_discard = 0
    lookback_list = {}
    sum_total = {}
    sum_total['nodes']=[]
    for i in range(0, lookback):
        lookback_list['age' + str(i + 1)] = []
    #for row in range(0,lookback):
    #    for k in range(0,lookback):
    #        lookback_list['qm' + str(k+1)].append(-200)

    #    print("appending -200 to row :" + str(row))
    #for row in range(lookback, col_len):
    for row in range(0, col_len):
        current_rows_to_discard = 0
        no_qm = True

        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list


        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            #print("i="+ str(i))
            sum_total_nodes = 0
            if df.at[i, "start"] > df.at[row, "submit"]:
                #print("start: " + str(df.at[i, "start"]))

                count_lookback = count_lookback + 1
                #print("value at i="+ str(i)+ "= "+ str(df.at[i, col_name]))
                age_waiting_job = pd.Timedelta(df.at[row, "submit"] - df.at[i, "submit"]).seconds/60.0
                lookback_list['age'+str(count_lookback)].append(age_waiting_job)
                sum_total_nodes = sum_total_nodes + df.at[i, "nnodes"]
                #print("loopback dictionary after append: " + str(lookback_list))
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    #sum_total['nodes'].append(sum_total_nodes)
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback_count never equals to  lookback. so do not consider the row " + str(
            current_rows_to_discard))
            rows_to_discard = current_rows_to_discard
            for k in range(count_lookback+1,lookback+1):
                lookback_list['age' + str(k)].append(0) # no job waiting
        sum_total['nodes'].append(sum_total_nodes)
                #print("appending -200 to row :"+str(k))
    #print("loopback dictionary: " + str(lookback_list))
    #print("length: qm1 " +str(len(lookback_list['qm1'])))
    rdf=pd.DataFrame(lookback_list)
    nodes=pd.DataFrame(sum_total)
    #print(rdf.to_string())
    rdf_con = pd.concat([df, rdf],axis=1)
    df_age_nodes = pd.concat([rdf_con, nodes],axis=1)
    return df_age_nodes, rows_to_discard


def optimised_add_total_age_nodes_past_k_obs_insert(lookback, df,col_name):
    ## Length of queue_minutes array
    col_len = len(df[col_name])
    rows_to_discard = 0
    lookback_list = {}
    sum_total = {}
    sum_total['nodes']=[]
    sum_total['waiting_time']=[]
    for i in range(0, lookback):
        lookback_list['age' + str(i + 1)] = []

    for row in range(0, col_len):
        current_rows_to_discard = 0
        no_qm = True

        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list


        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            #print("i="+ str(i))
            sum_total_nodes = 0
            sum_total_waiting_time=0.0
            if df.at[i, "start"] > df.at[row, "submit"]:
                #print("start: " + str(df.at[i, "start"]))

                count_lookback = count_lookback + 1
                #print("value at i="+ str(i)+ "= "+ str(df.at[i, col_name]))
                age_waiting_job = pd.Timedelta(df.at[row, "submit"] - df.at[i, "submit"]).seconds/60.0
                #lookback_list['age'+str(count_lookback)].append(age_waiting_job)
                sum_total_waiting_time = sum_total_waiting_time + age_waiting_job
                sum_total_nodes = sum_total_nodes + df.at[i, "nnodes"]
                #print("loopback dictionary after append: " + str(lookback_list))
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    #sum_total['nodes'].append(sum_total_nodes)
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback_count never equals to  lookback. so do not consider the row " + str(
            current_rows_to_discard))
            rows_to_discard = current_rows_to_discard
            #for k in range(count_lookback+1,lookback+1):
            #    lookback_list['age' + str(k)].append(0) # no job waiting
        sum_total['nodes'].append(sum_total_nodes)
        sum_total['waiting_time'].append(sum_total_waiting_time)
                #print("appending -200 to row :"+str(k))
    #print("loopback dictionary: " + str(lookback_list))
    #print("length: qm1 " +str(len(lookback_list['qm1'])))
    #rdf=pd.DataFrame(lookback_list)
    age_nodes=pd.DataFrame(sum_total)
    #print(rdf.to_string())
    rdf_con = pd.concat([df, age_nodes],axis=1)
    #df_age_nodes = pd.concat([rdf_con, nodes],axis=1)
    return rdf_con, rows_to_discard


def window_k_obs_all_vars_insert(lookback, df,col_name,vars):
    ## Length of queue_minutes array
    col_len = len(df[col_name])
    rows_to_discard = 0
    lookback_list = {}
    for i in range(0, lookback):
        lookback_list['qm' + str(i + 1)] = []
        for v in vars:
            lookback_list[v + str(i + 1)]=[]
        lookback_list["delta_t" + str(i + 1)]=[]
    #for row in range(0,lookback):
    #    for k in range(0,lookback):
    #        lookback_list['qm' + str(k+1)].append(-200)

    #    print("appending -200 to row :" + str(row))
    #for row in range(lookback, col_len):
    for row in range(0, col_len):
        current_rows_to_discard = 0
        no_qm = True

        count_lookback = 0
        #print("---- submit: " + str(df.at[row, "submit"])+ "  row = "+str(row))
        ## for each coulmn in data frame, create an empty lookback list


        ## for each row,append values to each lookback list
        for i in range(row, -1, -1):
            #print("i="+ str(i))
            if df.at[i, "start"] < df.at[row, "submit"]:
                #print("start: " + str(df.at[i, "start"]))

                count_lookback = count_lookback + 1
                #print("value at i="+ str(i)+ "= "+ str(df.at[i, col_name]))
                lookback_list['qm'+str(count_lookback)].append(df.at[i, col_name])
                for v in vars:
                    lookback_list[v +str(count_lookback)].append(df.at[i,v])
                age_waiting_job = pd.Timedelta(df.at[row, "submit"] - df.at[i, "submit"]).seconds / 60.0
                lookback_list["delta_t"+str(count_lookback)].append(age_waiting_job)
                #print("loopback dictionary after append: " + str(lookback_list))
                if (count_lookback == lookback):
                    #print("count_lookback equals to lookback at row: " + str(i))
                    no_qm = False
                    break;
        if no_qm == True:
            current_rows_to_discard = row
            print("*** loopback_count never equals to  lookback. so do not consider the row " + str(
            current_rows_to_discard))
            rows_to_discard = current_rows_to_discard
            for k in range(count_lookback+1,lookback+1):
                lookback_list['qm' + str(k)].append(-200)
                for v in vars:
                    lookback_list[v +str(k)].append(-200)
                lookback_list["delta_t" + str(k)].append(0)
                #print("appending -200 to row :"+str(k))
    #print("loopback dictionary: " + str(lookback_list))
    #print("length: qm1 " +str(len(lookback_list['qm1'])))
    rdf=pd.DataFrame(lookback_list)
    #print(rdf.to_string())
    rdf_con = pd.concat([df, rdf],axis=1)
    return rdf_con, rows_to_discard