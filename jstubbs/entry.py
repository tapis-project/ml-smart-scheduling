import json 
import os
import sys
# import various paths in the notebook environment 
sys.path.append('/home/jovyan/work/')
sys.path.append('/home/jovyan/work/src')
sys.path.append('/home/jovyan/work/src/data')

import numpy as np 
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from data import create_input_data


def get_simple_config():
    """
    An example config that exercises several of the features but doesn't require long to run. 
    """
    config = [
        {
            "source_dataset": "feb", # required; label referring to raw data file to use (TODO)
            "nbr_windows": 5, # required; number of time windows to initially divide the dataset into. 
            "bins": {
                "bin_threshold": 10,  # corresponds to threshold in google doc
                "bin_size_factor": 6, # multiplier for bin_threshold
                "nbr_bins": 5 
            },
            "outliers": {
                "drop_zero_jobs": True,
                "drop_big_jobs": True,
                "nbr_days_threshold": 2,
            },
            "current_future_split": {
                "cutoff_fraction": 0.90,
            },      
            "models": {
                # train_knn: {
                #     "param_grid": {
                #         "knn__n_neighbors": np.arange(start=1, stop=3),
                #     }
                # },
                # train_svm: {
                #     "param_grid": {
                #         "svc__kernel": ['rbf'],
                #         "svc__C": [1700],
                #         "svc__gamma": [1],
                #     },
                # },
                train_hgbc: {
                    "param_grid": {
                        "hgbc__learning_rate": [0.01, 0.1, 1],
                        "hgbc__max_iter": [10, 100, 400],
                        "hgbc__max_depth": [3, 5, 9],
                    }

                },
            }
        }
    ]
    return config


def get_window_config():
    """
    An example config that exercises using several windows across the larger dataset. 
    """
    config = [
        {
            "source_dataset": "feb", # required; label referring to raw data file to use (TODO)
            "nbr_windows": 10, # required; number of time windows to initially divide the dataset into. 
            "bins": {
                "bin_threshold": 10,
                "bin_size_factor": 6,
                "nbr_bins": 5
            },
            "outliers": {
                "drop_zero_jobs": True,
                "drop_big_jobs": True,
                "nbr_days_threshold": 2,
            },
            "current_future_split": {
                "cutoff_fraction": 0.90,
            },      
            "models": {
                train_knn: {
                    "param_grid": {
                        "knn__n_neighbors": np.arange(start=1, stop=3),
                    }
                },
                train_svm: {
                    "param_grid": {
                        "svc__kernel": ['rbf'],
                        "svc__C": [1700],
                        "svc__gamma": [1],
                    },

                }
            }

        }
    ]
    return config

def config_from_file():
    """
    Return the job config for this run from a file. By default, this
    function looks in the file job-config.json in the cwd.
    """
    # note: json approach doesn't allow for passing python objects such as the 
    #       train_* function.
    config_path = os.environ.get("JOB_CONFIG_PATH", "job-config.json")
    return json.load(open(config_path))


def get_job_config():
    """
    Return the job config for this run. The config is an array of `jobs`,
    each of which can be processed independently (e.g., in parallel).
    """
    # return get_window_config()
    return get_simple_config()


def get_raw_data(kind='jan'):
    """
    Read the raw data from a file and return a DataFrame.
    """
    # TODO: update to make this better
    if kind == 'jan':
        return create_input_data.read_data(csv_file_name="../data/processed/lookback35_anon_jan1_feb1.csv", parse_dates_col=[4,5,6])
    return create_input_data.read_data(csv_file_name="../data/processed/lookback35_anon_feb1_aug1.csv", parse_dates_col=[4,5,6])


def print_big_small_job_analysis(df, small_minutes_threshhold=5):
    """
    Function to print the number of jobs that sat in queue for 0 minutes as well 
    as the number of jobs that 
    """
    # How many jobs sat in queue for 0 minutes?
    nbr_jobs = len(df)
    nbr_zero_queue_min_jobs = len(df[df.queue_minutes == 0])
    
    nbr_small_queue_min_jobs = len(df[df.queue_minutes <= small_minutes_threshhold])
    print("Initial total number of jobs: ", nbr_jobs)
    print("Jobs with 0 queue minutes: ", nbr_zero_queue_min_jobs)
    print(f"Jobs with queue minutes leq {small_minutes_threshhold}: ", nbr_small_queue_min_jobs)
    print("Jobs with queue minutes: ", nbr_zero_queue_min_jobs)
    print("Percentage of jobs with 0 queue minutes: ", float(nbr_zero_queue_min_jobs)/nbr_jobs)


def drop_jobs_zero_minutes(df):
    """
    Drop all jobs that have a queue time of 0 minutes 
    """
    nbr_zero_min_jobs = len(df[df.queue_minutes == 0])
    print(f"Dropping {nbr_zero_min_jobs} 0 minute jobs")
    return df.drop(df[df.queue_minutes == 0].index)


def drop_high_queue_min_jobs(df, nbr_days_threshold=2):
    """
    Drop all jobs that sit in queue at least `nbr_days_threshold` (in days).
    """
    nbr_minutes_threshold = nbr_days_threshold * 24 * 60
    nbr_high_queue_min_jobs = len(df[df.queue_minutes >= nbr_minutes_threshold])
    print(f"Dropping {nbr_high_queue_min_jobs} jobs queued for greater than {nbr_minutes_threshold} minutes ({nbr_days_threshold} days)") 
    df = df.drop(df[df.queue_minutes >= nbr_minutes_threshold].index)
    return df


def create_queue_min_bins(df, 
                          bin_threshold=10,
                          bin_size_factor=6,
                          nbr_bins=5):
    """
    Add a new column, queue_minutes_bin, which is the bin of the queue_minutes for the job.
    Size and shape of bins are determined from the parameters:
      * bin_threshold: this is the `threshold` variable from the original google doc; it 
        can be thought of as the amount of time to provision the dynamic infrastructure. 
      * bin_size_factor: a scaling factor; multiplied by the bin_threshold to get the size 
        of each bin, in minutes. 
      * nbr_bins: The total number of bins. Jobs whose queue_minutes exceeds the max value 
        of the last bin will be assigned the last bin. 
    """
    # actual size of a bin, in minutes
    bin_size = bin_threshold * bin_size_factor
    # first, create the column as a float
    df['queue_minutes_bin'] = df['queue_minutes'] / bin_size
    # then, cast to int 
    df = df.astype({'queue_minutes_bin': 'int'})
    # for jobs in a bin number larger than the number of bins -1 (bins are 0-indexed), just 
    # put them in the largest bin. 
    df['queue_minutes_bin'] = np.where(df['queue_minutes_bin'] > (nbr_bins - 1), (nbr_bins - 1), df['queue_minutes_bin'])
    # print the final bin counts
    print("Job counts by bin:", df['queue_minutes_bin'].value_counts())
    return df 


def split_df_windows(df, nbr_windows):
    """
    Splits a dataframe, `df`, into `nbr_windows` dataframes of (roughly) equal size. 

    """
    dfs = []
    # the fraction of records for each dataframe is equal to 1 divided by the number of windows;
    # e.g., 3 windows means each df gets 33%; 4 windows means each df gets 25%, etc. 
    fraction = 1./nbr_windows
    for i in range(nbr_windows):
        # the ith dataframe is the set of rows between the
        lower_bound = np.quantile(df['submit'], i*fraction)
        upper_bound = np.quantile(df['submit'], (i+1)*fraction)
                                
        dfs.append(df[ (df['submit']>=lower_bound) & (df['submit']<= upper_bound) ])
    return dfs


def split_df_current_future(df, cutoff_fraction=0.75, cutoff_datetime=None):
    """
    Split a jobs dataframe into a current and future set. 
      * cutoff_fraction (float): the fraction to use for current.
      * cutoff_datetime, if supplied, should be the date_time to use as the cutoff point.
        When supplied, cutoff_fraction is ignored.
 
    Returns two dataframes, current and future
    """
    # if cutoff_datetime is provided, just use that 
    if cutoff_datetime: 
        current = df[df['submit'] <= cutoff_dateime]
        future = df[df['submit'] > cutoff_dateime]
        return current, future 
    # otherwise, we are using cutoff_fraction. 
    if not 0 < cutoff_fraction < 1:
        print("Invalid cutoff_fraction; values should be between 0 and 1.")
        return None, None 
    # the following uses np.quantile to split the df on the submit column using the cutoff_fraction
    current = df[df['submit']<=np.quantile(df['submit'], cutoff_fraction )]
    future = df[df['submit']>np.quantile(df['submit'], cutoff_fraction )]
    return current, future


def get_X_y(df, 
            X_cols=['nnodes', 
                    'max_minutes', 
                    'backlog_minutes', 
                    'backlog_num_jobs', 
                    'running_num_jobs', 
                    'running_minutes',], 
            y_col="queue_minutes_bin"):
    """
    Return the X and y objects for a dataframe, df.
    """
    X = df[X_cols]
    y = df[y_col]
    return X, y


def split_train_test(df,
                     X_cols=['nnodes', 
                              'max_minutes', 
                              'backlog_minutes', 
                              'backlog_num_jobs', 
                              'running_num_jobs', 
                              'running_minutes',], 
                     y_col="queue_minutes_bin",
                     test_size=0.2,
                     random_state=1):
    """
    Split df into training and testing sets.
    """
    X, y = get_X_y(df, X_cols, y_col)
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        test_size=test_size, 
                                                        stratify=y, 
                                                        random_state=random_state)
    return X_train, X_test, y_train, y_test


def train_knn(X_train, y_train, param_grid=None, n_neighbors=None):
    """
    Train a KNN classifier. 
    
    If param_grid is provided, this function will use CV to find 
    optimal values specified by the param_grid.
    
    Otherwise, if n_neighbors is provided, this function will train a single 
    classifier using the specified n_neighbors. 
    
    Finally, if neither param_grid nor n_neighbors is specified, 
    this function will use a pre-defined param_grid to search for an optimal 
    n_neighbors.

    """
    model = None 
    if param_grid or not n_neighbors: 
        p = pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('knn', KNeighborsClassifier()),
        ])
        if not param_grid:
            param_grid = {
                "knn__n_neighbors": np.arange(1, 100)
            }
        search = GridSearchCV(p, param_grid, n_jobs=8, refit=True)
        search.fit(X_train, y_train)
        print("Found optimal KNN: ", search.best_params_)
        print(f"Score with best parameters: {search.best_score_}")
        model = search.best_estimator_
    else: 
        p = pipeline = Pipeline([
            ('scale', StandardScaler()),
            ('knn', KNeighborsClassifier(n_neighbors=n_neighbors)),
        ])
        model = p.fit(X_train, y_train)

    return model 


def train_svm(X_train, y_train, param_grid=None):
    """
    Train a SVM classifier.

    If param_grid is provided, this function will use CV to find 
    optimal values specified by the param_grid.
    
    Otherwise, uses a pre-defined param_grid to search for optimal
    hyperparameters. 

    """

    p = Pipeline([
        ('scale', StandardScaler()),
        ('svc', SVC()),
    ])
    if not param_grid: 
        param_grid = {
            
            # kernel functions to try
            "svc__kernel": [
                # 'poly', 
                'rbf', 
                # 'sigmoid'
            ],
            
            # polynomial dregrees to try, for kernel==poly
            # "svc__degree": np.arange(start=2, stop=5),
            
            # regularization param ; bigger C tends to overfit
            # "svc__C": [0.1, 1, 10, 100, 1000],
            # "svc__C": np.arange(start=1000, stop=2000, step=100),
            "svc__C": np.arange(start=1500, stop=2000, step=100),
            
            # kernel coeff; 
            "svc__gamma": [1, 0.1, 0.01,]# 0.001, 0.0001], # ["scale", "auto"]
        }
    search = GridSearchCV(p, param_grid, n_jobs=8, refit=True, verbose=2) # verbose for progress logging
    search.fit(X_train, y_train)
    print(f"Score with best parameters: {search.best_score_}")
    print(search.best_params_)
    model = search.best_estimator_
    return model 


def train_hgbc(X_train, y_train, param_grid=None):
    """
    Train a Histogram-based Gradient Boosting Classification Tree.
    """
    p = Pipeline([
        ('scale', StandardScaler()),
        ('hgbc', HistGradientBoostingClassifier()),
    ])
    if not param_grid: 
        param_grid = {
            "hgbc__max_bins": [10, 50, 100, 200, 255, 320, 410],
        }

    search = GridSearchCV(p, 
                          param_grid, 
                          n_jobs=8, 
                          refit=True, 
                          verbose=2)
    search.fit(X_train, y_train)
    print(f"Score with best parameters: {search.best_score_}")
    print(search.best_params_)
    model = search.best_estimator_
    return model 
    
    
def compute_performance(model, X_test, y_test):
    """
    Compute the performance of a model on the test data. 
    """
    y_pred = model.predict(X_test)
    print("Accuracy on entire set: ", accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred)
    # Confusion Matrix ----
    # matrix[i, j] is nbr known to be in class i but predicted to be in class j
    matrix = confusion_matrix(y_test, y_pred)
    
    # Print accuracies if we only reschedule jobs predicted to be in the last bin -----
    # The total number of jobs rescheduled is the total number of all jobs predicted to be in 
    #     the last bin; i.e., the sum of the right-most column of the confusion matrix: 
    total_jobs_rescheduled = matrix[:, matrix.shape[1]-1].sum()
    # We can have different accuracy scores depending on which rescheduled jobs are "incorrect".
    # The worst case scenario is all other bins are considered "incorrect"
    # The best case scenario is only the 0th bin is considered "incorrect" (i.e., all bins except 
    #     bin 0 are considered "correct")
    incorrect_jobs = 0
    accuracy_scores = []
    # we iterate across rows 0, ..., shape-1 since we don't want to include the last row.
    for i in range(matrix.shape[0]-1):
        # add jobs that were predicted to be in the last bin but were actually in bin i:
        incorrect_jobs += matrix[i, matrix.shape[1]-1]
        correct_jobs = total_jobs_rescheduled - incorrect_jobs
        accuracy = correct_jobs/total_jobs_rescheduled
        accuracy_scores.append({"correct": correct_jobs, 
                                "incorrect": incorrect_jobs, 
                                "accuracy_score": accuracy})
        print(f"Accuracy for last bin scheduling assuming bins <= {i} are incorrect: {accuracy}; ({correct_jobs}/{total_jobs_rescheduled})")
    print("Confusion Matrix:")
    print(matrix)
    return report, matrix, accuracy_scores


def test_get_simple_df():
    df = get_raw_data()
    df = drop_jobs_zero_minutes(df)
    df = drop_high_queue_min_jobs(df)
    df = create_queue_min_bins(df)
    return df 


def main(): 
    config = get_job_config()
    reports = []
    matrices = []
    score_list = []
    for idx, job in enumerate(config):
        print(f"\n\n* * * * * Starting execution for job {idx} * * * * *")    
        df = get_raw_data(job["source_dataset"])
        print_big_small_job_analysis(df)
        if job["outliers"].get("drop_zero_jobs"):
            df = drop_jobs_zero_minutes(df)
        if job["outliers"].get("drop_big_jobs"):
            nbr_days_threshold = job["outliers"].get("nbr_days_threshold")
            df = drop_high_queue_min_jobs(df, nbr_days_threshold=nbr_days_threshold)
        df = create_queue_min_bins(df, 
                                   bin_threshold=job["bins"].get("bin_threshold"),
                                   bin_size_factor=job["bins"].get("bin_size_factor"),
                                   nbr_bins=job["bins"].get("nbr_bins"))
        nbr_windows = job['nbr_windows']
        # divide the dataframe into `nbr_windows` (roughly) equally sized time windows 
        # and perform the analysis on each window in isolation. 
        dfs = split_df_windows(df, nbr_windows)
        # for each df/window, we perform the following tasks:
        #   1. split the df/window into current and future 
        #   2. split the current into train and test 
        #   3. for each model in the job: 
        #      3a. train the model on window_train
        #      3b. test the model on window_test 
        #      3c. score the model on future 
        for window_nbr, df in enumerate(dfs):
            print(f"\nStarting window {window_nbr}\n*********************")
            cutoff_fraction = job["current_future_split"].get("cutoff_fraction")
            future = None 
            if cutoff_fraction:
                current, future = split_df_current_future(df, cutoff_fraction=cutoff_fraction)
            elif job["current_future_split"].get("cutoff_datetime"):
                cutoff_datetime = job["current_future_split"].get("cutoff_datetime")
                current, future = split_df_current_future(df, cutoff_datetime=cutoff_datetime)
            if future is not None:
                X_train, X_test, y_train, y_test = split_train_test(current)
            else:
                X_train, X_test, y_train, y_test = split_train_test(df)
            print(f"Window Start: {df['submit'].min()}; Window End: {df['submit'].max()}; Total Jobs in Window: {len(df)}")
            for model_type in job["models"].keys():
                kwargs = job["models"][model_type]
                model = model_type.__call__(X_train, y_train, **kwargs)
                print("Performance on TEST")
                report, matrix, scores = compute_performance(model, X_test, y_test)
                if future is not None: 
                    print("Performance on FUTURE")
                    X_future, y_future = get_X_y(future)
                    report, matrix, scores = compute_performance(model, X_future, y_future)
                    reports.append(report)
                    matrices.append(matrix)
                    score_list.append(scores)

    return reports, matrices, score_list


if __name__ == "__main__":
    reports, matrices, score_list = main()
    for i in range(len(reports)):
        print(reports[i], matrices[i], score_list[i])