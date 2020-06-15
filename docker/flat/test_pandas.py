import logging
import numpy as np
import os
import numpy.random as rnd
from sklearn.metrics import f1_score

from ad_examples.common.utils import read_csv, dataframe_to_matrix
from ad_examples.common.gen_samples import get_synthetic_samples
from ad_examples.aad.aad_support import AadOpts, get_aad_command_args, configure_logger
from ad_examples.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, BayesianRulesetsDescriber, get_region_memberships
from ad_examples.aad.demo_aad import get_debug_args, detect_anomalies_and_describe

from ad_examples.loda.loda import Loda

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


def load_data():
    print("loading csv...")
    data_df = read_csv("../notebooks/data/simple.type123.csv", header=True)

    print("transforming data...")
    x, y = dataframe_to_matrix(data_df)
    return (x, y)


def slice_data(x, y, idx_from, idx_to):
    n = x.shape[0]
    return (x[idx_from:idx_to, :], y[idx_from:idx_to])


def run_loda(x_old, scores_old, x_new, outliers_fraction):
    rnd.seed(42)

    print("running LODA...")
    ad = Loda(mink=100, maxk=200)
    ad.fit(x_old)
    if scores_old == None:
        print("Calculating inital scores")
        scores_old = -ad.decision_function(x_old)

    print("Evaluating...")
    scores = -ad.decision_function(x_new)

    print("Combining with historic scores and converting to classes...")
    print(scores_old)
    print(scores)
    scores_combined = np.concatenate((np.array(scores_old), np.array(scores)), 0)
    y_pred_combined = convert_scores_to_classes(scores_combined, outliers_fraction)
    y_pred = y_pred_combined[len(scores_old):]

    return (scores_combined, y_pred)

#################################################################################

(gt_x, gt_y) = load_data()
# print(gt_x.shape)

day_rec_cnt = 24 * 12
block_size = 100 * day_rec_cnt
idx_start = 200 * day_rec_cnt
idx_curr_time = idx_start
n = gt_y.shape[0]
scores_all = None
y_pred = np.zeros(0)
outlier_fraction = 0.01

while idx_curr_time < n :
    print(n, idx_curr_time, block_size)
    (x1, y1) = slice_data(gt_x, gt_y, 0, idx_curr_time)
    (x2, y2) = slice_data(gt_x, gt_y, idx_curr_time, idx_curr_time + block_size)
    (scores_all, y_pred_new) = run_loda(x1, scores_all, x2, outlier_fraction)
    y_pred = np.concatenate((np.array(y_pred), np.array(y_pred_new)), 0)
    idx_curr_time = idx_curr_time + block_size

print("finished with training, analyzing combined output")
y = gt_y[idx_start:]

print("Calculating F1 scores...")
f1 = f1_score(y, y_pred, average=None) # average='weighted')

print(f1)
