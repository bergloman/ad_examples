import logging
import numpy as np
import os
import numpy.random as rnd

from ad_examples.common.utils import read_csv, dataframe_to_matrix
from ad_examples.common.gen_samples import get_synthetic_samples
from ad_examples.aad.aad_support import AadOpts, get_aad_command_args, configure_logger
from ad_examples.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, BayesianRulesetsDescriber, get_region_memberships
from ad_examples.aad.demo_aad import get_debug_args, detect_anomalies_and_describe

from ad_examples.loda.loda import Loda

logger = logging.getLogger(__name__)

# # Prepare the aad arguments. It is easier to first create the parsed args and
# # then create the actual AadOpts from the args
# args = get_aad_command_args(debug=True, debug_args=get_debug_args())


# opts = AadOpts(args)
# logger.debug(opts.str_opts())
# np.random.seed(opts.randseed)
# # load synthetic (toy 2) dataset
# x, y = get_synthetic_samples(stype=2)

# # run interactive anomaly detection loop
# model, x_transformed, queried, ridxs_counts, region_extents = detect_anomalies_and_describe(x, y, opts)

def convert_scores_to_classes(scores, anomaly_ratio):
    """
    Converts list of scores to flags (0/1) - top anomalies are marked as 1.
    """
    anomaly_cnt = int(len(scores) * anomaly_ratio)
    anomaly_indices = np.array(scores).argsort()[-anomaly_cnt:][::-1]
    y_pred = np.zeros(len(scores))
    np.put(y_pred, anomaly_indices, 1)
    return y_pred

ad_type="loda"
rnd.seed(42)

print("loading csv...")
data_df = read_csv("./data/simple.type123.csv", header=True)

print("transforming data...")
x, y = dataframe_to_matrix(data_df)

n = x.shape[0]
# outliers_fraction = 0.01
xx = yy = x_grid = Z = scores = None

print("running LODA...")
ad = Loda(mink=100, maxk=200)
ad.fit(x)

print("Evaluating...")
scores = -ad.decision_function(x)
# Z = -ad.decision_function(x_grid)

print("scores:\n%s" % str(list(scores)))
top_anoms = np.argsort(-scores)[np.arange(10)]

print("top anomalies:\n%s" % str(list(top_anoms)))
