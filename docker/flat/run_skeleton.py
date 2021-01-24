import sys
import time
import json

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/tf/jovyan/work')

import logging
import numpy as np
import pandas as pd
import os
# import tensorflow as tf
import numpy.random as rnd
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM, SVC

# from ad_examples.common.utils import read_csv, dataframe_to_matrix
# from ad_examples.common.gen_samples import get_synthetic_samples
# from ad_examples.common.nn_utils import AutoencoderAnomalyDetector
# from ad_examples.aad.aad_support import AadOpts, get_aad_command_args, configure_logger
# from ad_examples.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, BayesianRulesetsDescriber, get_region_memberships
# from ad_examples.aad.demo_aad import get_debug_args, detect_anomalies_and_describe

# from ad_examples.loda.loda import Loda

logger = logging.getLogger(__name__)

def convert_scores_to_classes(scores, anomaly_ratio):
    """
    Converts list of scores to flags (0/1) - top anomalies are marked as 1.
    """
    anomaly_cnt = int(len(scores) * anomaly_ratio)
    anomaly_indices = np.array(scores).argsort()[-anomaly_cnt:][::-1]
    y_pred = np.zeros(len(scores))
    np.put(y_pred, anomaly_indices, 1)
    return y_pred

def read_csv(file, header=None, sep=',', index_col=None, skiprows=None, usecols=None, encoding='utf8'):
    """Loads data from a CSV

    Returns:
        DataFrame
    """

    if header is not None and header:
        header = 0 # first row is header

    data_df = pd.read_csv(file, header=header, sep=sep, index_col=index_col, skiprows=skiprows, usecols=usecols, encoding=encoding)
    return data_df

def dataframe_to_matrix(df, labelindex=0, startcol=1):
    """ Converts a python dataframe in the expected anomaly dataset format to numpy arrays.

    The expected anomaly dataset format is a CSV with the label ('anomaly'/'nominal')
    as the first column. Other columns are numerical features.

    Note: Both 'labelindex' and 'startcol' are 0-indexed.
        This is different from the 'read_data_as_matrix()' method where
        the 'opts' parameter has same-named attributes but are 1-indexed.

    :param df: Pandas dataframe
    :param labelindex: 0-indexed column number that refers to the class label
    :param startcol: 0-indexed column number that refers to the first column in the dataframe
    :return: (np.ndarray, np.array)
    """
    cols = df.shape[1] - startcol
    x = np.zeros(shape=(df.shape[0], cols))
    for i in range(cols):
        x[:, i] = df.iloc[:, i + startcol]
    labels = np.array([True if df.iloc[i, labelindex] == "anomaly" else False for i in range(df.shape[0])], dtype=int)
    return x, labels

def load_data(input_file):
    print("loading csv...")
    # t = "ber"
    # size = "simple"
    # n = "_normalized_hours"
    # data_df = read_csv("../notebooks/data/simple.type123.csv", header=True)
    # data_df = read_csv("./data/data_parking/csv/type-ber/simple.type-ber.csv", header=True)
    #data_df = read_csv("./data/data_parking/csv" + n + "/type-" + t + "/" + size + ".type-" + t + ".csv", header=True)
    data_df = read_csv(input_file, header=True)

    # print(data_df.head())
    print("transforming data...")
    x, y = dataframe_to_matrix(data_df)
    # print(x)
    # print(y)
    return (x, y)


def slice_data(x, y, idx_from, idx_to):
    n = x.shape[0]
    return (x[idx_from:idx_to, :], y[idx_from:idx_to])


def run_ad_algorithm(algo_type, x_old, scores_old, x_new, outliers_fraction):
    seed = 42
    rnd.seed(seed)

    ad = IsolationForest(
        n_estimators=70,
        max_samples='auto',
        contamination=0.01,
        max_features=1.0,
        bootstrap=False,
        # n_jobs=-1,
        # behaviour='deprecated',
        random_state=seed,
        verbose=1,
        # warm_start=False,
    )
    # ad = OneClassSVM(
    #     kernel='rbf',
    #     degree=3,
    #     gamma='scale',
    #     coef0=0.0,
    #     tol=1e-3,
    #     nu=0.5,
    #     shrinking=True,
    #     cache_size=2048,
    #     verbose=False,
    #     max_iter=1_000
    # )
    # ad = LocalOutlierFactor(
    #     n_neighbors=20,
    #     algorithm='auto',
    #     leaf_size=30,
    #     metric='minkowski',
    #     p=2,
    #     metric_params=None,
    #     contamination=0.01,
    #     # novelty=False,
    #     # n_jobs=-1
    # )

    print("fitting...")
    print(x_old)
    ad.fit(x_old)

    print("predicting...")
    if len(scores_old) == 0:
        scores_old = (ad.predict(x_old) != 1)

    scores = (ad.predict(x_new) != 1)

    # print("Combining with historic scores and converting to classes...")
    scores_combined = np.concatenate((np.array(scores_old), np.array(scores)), 0)
    y_pred_combined = scores_combined
    y_pred = y_pred_combined[len(scores_old):]

    return (scores_combined, y_pred)

#################################################################################

args = sys.argv
print(args)
algo=args[2]
(gt_x, gt_y) = load_data(args[1])

day_rec_cnt = 24 * 12
block_size = 170 * day_rec_cnt
idx_start = 160 * day_rec_cnt
idx_curr_time = idx_start
n = gt_y.shape[0]
scores_all = np.zeros(0)
y_pred = np.zeros(0)
outlier_fraction = 0.01

t0 = time.clock()

while idx_curr_time < n :
    print(n, idx_curr_time, block_size)
    (x1, y1) = slice_data(gt_x, gt_y, 0, idx_curr_time)
    (x2, y2) = slice_data(gt_x, gt_y, idx_curr_time, idx_curr_time + block_size)

    (scores_all, y_pred_new) = run_ad_algorithm(algo, x1, scores_all, x2, outlier_fraction)
    y_pred = np.concatenate((np.array(y_pred), np.array(y_pred_new)), 0)
    # print(np.sum(y1), np.sum(y2), np.sum(y_pred))

    idx_curr_time = idx_curr_time + block_size
    y_tmp = gt_y[idx_start:idx_curr_time]
    f1 = f1_score(y_tmp, y_pred, average=None) # average='weighted')
    print(f1)

t1 = time.clock()

print("finished with training, analyzing combined output")
y = gt_y[idx_start:]

print("Elapsed time")
print(t1 -t0)

print("Calculating F1 scores...")
# print(y)
# print(y_pred)
f1 = f1_score(y, y_pred, average=None) # average='weighted')
prec2, recall2, f05, _ = precision_recall_fscore_support(y, y_pred, average=None, beta=0.5)
# print(json.dumps({ "time": t1 - t0, "f1": f1[1], "precision": prec2[1], "recall": recall2[1], "f05": f05[1] }))
print(f1)
print(prec2)
print(recall2)
print(f05)

print("Calculating F1 scores again...")
f1 = f1_score(y, y_pred, labels=[False, True]) # average='weighted')
#prec2, recall2, f05, _ = precision_recall_fscore_support(y, y_pred, beta=0.5)
prec2, recall2, f05, _ = precision_recall_fscore_support(y, y_pred)
print(f1)
print(prec2)
print(recall2)
print(f05)
