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


def run_loda(x, y):
    ad_type="loda"
    data_type = "type123"
    data_size="complex"

    rnd.seed(42)

    n = x.shape[0]
    outliers_fraction = np.sum(y) / len(y)
    xx = yy = x_grid = Z = scores = None

    print("running LODA...")
    ad = Loda(mink=100, maxk=200)
    ad.fit(x)

    print("Evaluating...")
    scores = -ad.decision_function(x)

    print("Converting scores to classes...")
    y_pred = convert_scores_to_classes(scores, outliers_fraction)

    print("Calculating F1 scores...")
    f1 = f1_score(y, y_pred, average=None) # average='weighted')
    # print("F1={:f}".format(f1))
    print(f1)

# print(x)

# n = x.shape[0]
# print(x.shape)

# print(y)
# print(y.shape)

# print( y[0:3])

# print( x[1:3, 0:4])
# print( x[0:3, 0:5])

(gt_x, gt_y) = load_data()
# print(gt_x.shape)


(gt_x2, gt_y2) = slice_data(gt_x, gt_y, 0, 20000)
run_loda(gt_x2, gt_y2)

(gt_x3, gt_y3) = slice_data(gt_x, gt_y, 0, 50000)
run_loda(gt_x3, gt_y3)

# print(gt_x2.shape)
# print(gt_y2.shape)
# y1 = gt_y[0:10]
# y2 = gt_y[10:12]
# print(y1)
# print(y2)
