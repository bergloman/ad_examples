import sys
sys.path.insert(1, '/tf/jovyan/work')

import logging
import numpy as np
import os
import tensorflow as tf
import numpy.random as rnd
from sklearn.metrics import f1_score, precision_recall_fscore_support

from ad_examples.common.utils import read_csv, dataframe_to_matrix
from ad_examples.common.gen_samples import get_synthetic_samples
from ad_examples.common.nn_utils import AutoencoderAnomalyDetector
from ad_examples.aad.aad_support import AadOpts, get_aad_command_args, configure_logger
from ad_examples.aad.forest_description import CompactDescriber, MinimumVolumeCoverDescriber, BayesianRulesetsDescriber, get_region_memberships
from ad_examples.aad.demo_aad import get_debug_args, detect_anomalies_and_describe

from ad_examples.loda.loda import Loda

from ad_examples.glad.afss import *
from ad_examples.glad.glad_test_support import *


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


def load_data(input_file):
    print("loading csv...")
    # data_df = read_csv("../notebooks/data/simple.type123.csv", header=True)
    data_df = read_csv(input_file, header=True)

    print("transforming data...")
    x, y = dataframe_to_matrix(data_df)
    print("shape: %s" % (str(x.shape)))

    return (x, y)


def slice_data(x, y, idx_from, idx_to):
    n = x.shape[0]
    return (x[idx_from:idx_to, :], y[idx_from:idx_to])


def run_detector(x_old, scores_old, x_new, outliers_fraction):
    rnd.seed(42)

    print("running LODA...")
    ad = Loda(mink=100, maxk=200)

    # print("running auto-encoder...")
    # input_dims = x_old.shape[1]
    # ad = AutoencoderAnomalyDetector(
    #     n_inputs = input_dims,
    #     n_neurons = [2 * input_dims, round( input_dims/ 5), 2 * input_dims],
    #     normalize_scale = True,
    #     activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
    # )

    ad.fit(x_old)
    if len(scores_old) == 0:
        print("Calculating inital scores")
        scores_old = -ad.decision_function(x_old)

    print("Evaluating...")
    scores = -ad.decision_function(x_new)

    print("Combining with historic scores and converting to classes...")
    # print(scores_old)
    # print(scores)
    scores_combined = np.concatenate((np.array(scores_old), np.array(scores)), 0)
    y_pred_combined = convert_scores_to_classes(scores_combined, outliers_fraction)
    y_pred = y_pred_combined[len(scores_old):]

    return (scores_combined, y_pred)


def get_afss_modelx(opts, n_output=1):

    layer_sizes = [] # opts.afss_nodes
    if len(layer_sizes) == 0 or any(n < 1 for n in layer_sizes):
        layer_sizes = [max(50, n_output * 3)]
        logger.debug("Setting layer_sizes to [%d]" % layer_sizes[0])

    logger.debug("l2_lambda: %f" % opts.afss_l2_lambda)
    logger.debug("max_afss_epochs: %d" % opts.max_afss_epochs)

    n_neurons = layer_sizes + [n_output]

    names = ["hidden%d" % (i+1) for i in range(len(n_neurons)-1)]
    names.append("output")
    activations = []
    if len(n_neurons) > 2:
        activations = [tf.nn.leaky_relu] * (len(n_neurons) - 2)
    activations.extend([tf.nn.sigmoid, None])

    logger.debug("n_neurons (%d): %s" % (len(n_neurons), str(n_neurons)))
    logger.debug("names: %s" % str(names))

    afss = AFSS(n_neurons=n_neurons, names=names,
                activations=activations, bias_prob=opts.afss_bias_prob,
                prime=not opts.afss_no_prime, c_q_tau=opts.afss_c_tau, c_x_tau=opts.afss_c_tau,
                lambda_prior=opts.afss_lambda_prior, l2_penalty=True, l2_lambda=opts.afss_l2_lambda,
                train_batch_size=opts.train_batch_size,
                max_init_epochs=opts.n_epochs, max_afss_epochs=opts.max_afss_epochs,
                max_labeled_reps=opts.afss_max_labeled_reps)

    return afss

def afss_active_learn_ensemblex(x, y, ensemble, queried, opts):
    """ Create GLAD for given ensemble and data

        :param x: data
        :param y: anomaly indications
        :param ensemble: LODA ensemble to use
        :param queried: labeled instances
        :param opts: additional options
    """

    # populate labels as some dummy value (-1) initially
    y_labeled = np.ones(x.shape[0], dtype=int) * -1
    # find raw scores from ensemble for current data
    scores = ensemble.get_scores(x)

    # initialize GLAD network
    afss = get_afss_modelx(opts, n_output=ensemble.m)
    afss.init_network(x, prime_network=True)

    baseline_scores = afss.get_weighted_scores(x, scores)
    baseline_queried = np.argsort(-baseline_scores)
    baseline_found = np.cumsum(y[baseline_queried[np.arange(opts.budget)]])
    logger.debug("baseline found:\n%s" % (str(list(baseline_found))))

    for i in range(opts.budget):
        tm = Timer()
        a_scores = afss.get_weighted_scores(x, scores)
        ordered_indexes = np.argsort(-a_scores)
        items = get_first_vals_not_marked(ordered_indexes, queried, start=0, n=1)
        queried.extend(items)
        hf = np.array(queried, dtype=int)
        y_labeled[items] = y[items]

        afss.update_afss(x, y_labeled, hf, scores, tau=opts.afss_tau)
        logger.debug(tm.message("finished budget %d:" % (i+1)))


    afss.close_session()

    # the number of anomalies discovered within the budget while incorporating feedback
    # logger.debug("queried:\n%s" % str(queried))
    # logger.debug("y_labeled:\n%s" % str(list(y_labeled[queried])))
    found = np.cumsum(y[queried])
    logger.debug("GLAD found:\n%s" % (str(list(found))))

    # make queried indexes 1-indexed
    queried = np.array(queried, dtype=int) + 1
    baseline_queried = np.array(baseline_queried[0:opts.budget], dtype=int) + 1

    results = SequentialResults(num_seen=found, num_seen_baseline=baseline_found,
                                queried_indexes=queried,
                                queried_indexes_baseline=baseline_queried)
    return results


# run glad for new batch (e.g. week)
def run_glad(x_old, y_old, x_new, y_new, queried, opts):
    rnd.seed(42)

    # all_results = SequentialResults()

    budget = 30
    train_batch_size = 100
    afss_nodes = 0
    orig_randseed = 54536
    loda_mink = 100
    loda_maxk = 200

    print("running GLAD block ...")
    randseed = orig_randseed
    runidx = 1
    set_random_seeds(randseed, randseed + 1, randseed + 2)

    # create LODA ensemble for the initial batch
    loda_model = Loda(mink=loda_mink, maxk=loda_maxk)
    loda_model.fit(x)
    ensemble = AnomalyEnsembleLoda(loda_model)

    logger.debug("#LODA projections: %d" % ensemble.m)

    afss_active_learn_ensemblex(x, y, ensemble, opts)

    logger.debug("completed run of GLAD block")

    return queried

    # print("running auto-encoder...")
    # input_dims = x_old.shape[1]
    # ad = AutoencoderAnomalyDetector(
    #     n_inputs = input_dims,
    #     n_neurons = [2 * input_dims, round( input_dims/ 5), 2 * input_dims],
    #     normalize_scale = True,
    #     activations=[tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, None]
    # )

    # ad.fit(x_old)
    # if len(scores_old) == 0:
    #     print("Calculating inital scores")
    #     scores_old = -ad.decision_function(x_old)

    # print("Evaluating...")
    # scores = -ad.decision_function(x_new)

    # print("Combining with historic scores and converting to classes...")
    # # print(scores_old)
    # # print(scores)
    # scores_combined = np.concatenate((np.array(scores_old), np.array(scores)), 0)
    # y_pred_combined = convert_scores_to_classes(scores_combined, outliers_fraction)
    # y_pred = y_pred_combined[len(scores_old):]

    # return (scores_combined, y_pred)

#################################################################################

(gt_x, gt_y) = load_data("data/data_parking/csv/type-ber/simple.type-ber.csv")

day_rec_cnt = 24 * 12
block_size = 7 * day_rec_cnt
idx_start = 60 * day_rec_cnt
idx_curr_time = idx_start
n = gt_y.shape[0]
scores_all = np.zeros(0)
y_pred = np.zeros(0)
outlier_fraction = 0.03
queried = []

opts = {
    afss_l2_lambda: 1e-3,
    max_afss_epochs: 1,
    afss_bias_prob: 0.5,
    afss_no_prime: False,
    afss_c_tau: 1.0,
    afss_lambda_prior: 1.0,
    train_batch_size: 25,
    n_epochs: 200,
    max_afss_epochs: 1,
    afss_max_labeled_reps: 5,
    budget: 3 * 7 # number of active-learning questions allowed per block (i.e. single week)
}

# activations=activations, bias_prob=opts.afss_bias_prob,
# prime=not opts.afss_no_prime, c_q_tau=opts.afss_c_tau, c_x_tau=opts.afss_c_tau,
# lambda_prior=opts.afss_lambda_prior, l2_penalty=True, l2_lambda=opts.afss_l2_lambda,
# train_batch_size=opts.train_batch_size,
# max_init_epochs=opts.n_epochs, max_afss_epochs=opts.max_afss_epochs,
# max_labeled_reps=opts.afss_max_labeled_reps)


# create LODA ensemble for the initial batch
# loda_model = Loda(mink=mink, maxk=maxk)
# loda_model.fit(x)
# ensemble = AnomalyEnsembleLoda(loda_model)

# initialize GLAD NN + allow some AAD training
(x_init, y_init) = slice_data(gt_x, gt_y, 0, idx_curr_time)
(queried, scores_old) = run_glad([], [], x_init, y_init, [], opts)

# for each week:
#    train new GLAD + LODA on old data
#    use new GLAD + LODA on new data to assign scores
#    collect results






while idx_curr_time < n :
    print(n, idx_curr_time, block_size)
    (x1, y1) = slice_data(gt_x, gt_y, 0, idx_curr_time)
    (x2, y2) = slice_data(gt_x, gt_y, idx_curr_time, idx_curr_time + block_size)
    (scores_all, y_pred_new) = run_detector(x1, scores_all, x2, outlier_fraction)
    y_pred = np.concatenate((np.array(y_pred), np.array(y_pred_new)), 0)
    idx_curr_time = idx_curr_time + block_size

print("finished with training, analyzing combined output")
y = gt_y[idx_start:]

print("Calculating F1 scores...")
f1 = f1_score(y, y_pred, average=None) # average='weighted')
prec2, recall2, f05, _ = precision_recall_fscore_support(y_tmp, y_pred, average=None, beta=0.5)

print(f1)
print(json.dumps({ "time": t1 - t0, "f1": f1[1], "precision": prec2[1], "recall": recall2[1], "f05": f05[1] }))
