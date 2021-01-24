from typing import List, Tuple, Generator, Iterator
from enum import Enum
from os import path

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

# Set custom matplotlib themes
#try:
#    plt.style.use(['science', 'ieee'])
#except OSError:
#    pass

from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, MinMaxScaler
from sklearn.ensemble import IsolationForest, BaggingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import OneClassSVM, SVC
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, ParameterGrid

from sklearn import metrics, preprocessing, ensemble, linear_model, model_selection, neighbors, pipeline, svm, base, tree


from datasets.trace1_Rutgers.transform import get_traces, dtypes, get_traces2
from datasets.trace1_Rutgers import NOISE_SOURCES

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, backend as K, callbacks

tf.config.set_soft_device_placement(True)

import joblib
from joblib import Parallel, delayed

PROJECT_ROOT = path.dirname(path.realpath(__file__))

# Cache for faster processing
memory = joblib.Memory(location=path.join(PROJECT_ROOT, '.cache'), verbose=1, compress=True)


SEED = 0xDEADBEEF

TX_POWER = 10.0 # dBm
RSSI_BASE = -95.0 # dBm; not sure for this one. Most of the Atheros' have -95
FREQ = 2450.0 # MHz


def name2coords(name: str) -> Tuple[float, float]:
    """Name of the node contains information about it's relative position in testbed"""
    part = name[len('node'):] # extract only numeric part 'node3-2' -> '3-2'
    coords = [float(i) for i in part.split('-')]
    return tuple(coords)


#def get_rutgers_traces() -> Tuple[pd.DataFrame]:
#    def transform(df: pd.DataFrame) -> pd.DataFrame:
#        df['src'] = df['src'].apply(name2coords) # str -> coordinates
#        df['dst'] = df['dst'].apply(name2coords) # str -> coordinates
#        df['seq'] = df.index # preserve sequence number
#        df['rss'] = df.rssi + RSSI_BASE # received signal strength
#        # Calculate euclidean distance between Tx and Rx
#        return df
#
#    traces = get_traces2(n_jobs=-1)
#    output = Parallel(n_jobs=-1)(delayed(transform)(df) for df in traces)
#    return output


#def get_rutgers_datapoints() -> pd.DataFrame:
#    """Merge all dataset into single Pandas dataframe."""
#    dfs = get_rutgers_traces()
#    df = pd.concat(dfs, ignore_index=True, copy=False)
#    return df


@memory.cache
def get_initial_dataset() -> List[pd.DataFrame]:
    def process(df: pd.DataFrame):
        assert len(df.index) == 300

        if df.received.sum() != 300:
            return None

        df['src'] = df['src'].apply(name2coords) # str -> coordinates
        df['dst'] = df['dst'].apply(name2coords) # str -> coordinates
        df['seq'] = df.index # preserve sequence number
        df['rss'] = df.rssi + RSSI_BASE # received signal strength

        # Fix #1: Initial values are abnormal on every link. Replace them to avoid misclassification
        mean, stdev = np.mean(df[df.seq > 3].rss), np.std(df[df.seq > 3].rss)
        df.loc[df.seq < 3, 'rss'] = np.random.normal(mean, stdev, size=3)

        return df

    # Parallel link transformation
    traces = get_traces2(n_jobs=-1)
    dfs = Parallel(n_jobs=-1)(delayed(process)(df) for df in traces)

    # Merge them at the end
    output = pd.concat(dfs, ignore_index=True)
    return output



# @memory.cache
# def get_initial_dataset() -> pd.DataFrame:plt.style.use(['science', 'ieee'])RR should be within [0, 1] range, got {prr}'
#         df.loc[query, 'prr'] = prr

#     # Only links with 100% PRR are suitable for simulation
#     df = df[df.prr == 1.0].copy()

#     # Fix initial 3 values of the series
#     unique_pairs = df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
#     for src, dst, noise in unique_pairs:
#         query = (df.src == src) & (df.dst == dst) & (df.noise == noise)
#         mean, stdev = np.mean(df.loc[query].rss), np.std(df.loc[query].rss)
#         df.loc[query & (df.seq < 3), 'rss'] = np.random.normal(mean, stdev, size=3)

#     return df


@memory.cache
def get_unique_pairs():
    df = get_initial_dataset()
    pairs = df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
    return pairs


@memory.cache
def dummy_anomaly_injector(scaler=None, random_state=None) -> pd.DataFrame:
    # Get ready-to-use prepared dataset for anomaly injection
    df = get_initial_dataset()

    df['anomaly'] = False
    df['original'] = df.rss

    # Now, find unique pairs
    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
    n_pairs = len(unique_pairs)

    # Scale links afterwards
    if scaler:
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            df.loc[query, ['rss']] = scaler.fit_transform(df.loc[query, ['rss']])

    assert len(df.index) != 0
    return df


@memory.cache
def spike_anomaly_injector(scaler=None, random_state=None) -> pd.DataFrame:
    # Get ready-to-use prepared dataset for anomaly injection
    df = get_initial_dataset()

    # Random number generator
    rng = np.random.default_rng(random_state)

    # Now, find unique pairs
    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
    n_pairs = len(unique_pairs)

    df['anomaly'] = False
    df['original'] = df.rss

    # approx. 33.3% of links will have anomaly
    n_anomaly_pairs = int(n_pairs // 3)

    # The probability for spike anomaly to appear is:
    anomaly_probability = 3. / 300.

    # Randomly select links, which will be altered
    affected_links = rng.choice(unique_pairs, n_anomaly_pairs)

    for src, dst, noise in affected_links:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
        size = len(df[query].index)
        assert size == 300, f'something is wrong. Size {size} is not 300'
        df.loc[query, 'anomaly'] = np.random.choice([False, True], p=[1. - anomaly_probability, anomaly_probability], size=300)

        ## Now inject actual annomaly

        # get delta between median and noise floor.
        delta = abs(RSSI_BASE - df.loc[query, 'rss'].median())

        # Drop should be around 30 or less, depending on delta
        delta = delta if delta < 30 else (30 - rng.uniform(0, 3))

        # Apply drop where anomaly should have appeared
        df.loc[query & df.anomaly, 'rss'] -= delta

        # Fix values below noise floor
        df.loc[query & df.anomaly & (df['rss'] < RSSI_BASE), 'rss'] = RSSI_BASE

    assert np.any(df.original != df.rss)

    # Scale links afterwards
    if scaler:
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            df.loc[query, ['rss']] = scaler.fit_transform(df.loc[query, ['rss']])

    assert len(df.index) != 0
    return df


@memory.cache
def norecovery_anomaly_injector(scaler=None, random_state=None) -> pd.DataFrame:
    # Get ready-to-use prepared dataset for anomaly injection
    df = get_initial_dataset()

    # Random number generator
    rng = np.random.default_rng(random_state)

    # Now, find unique pairs
    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
    n_pairs = len(unique_pairs)

    df['anomaly'] = False
    df['original'] = df.rss

    # approx. 33.3% of links will have anomaly
    n_anomaly_pairs = int(n_pairs // 3)

    # No-recovery anomaly has randomized start, which will appear randomly between 200th and 280th packet
    # Let's assume there is no more the 1/3 of malicious samples.
    randomize_start = lambda: rng.integers(200, 280, endpoint=True)

    affected_links = rng.choice(unique_pairs, n_anomaly_pairs)

    for src, dst, noise in affected_links:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
        size = len(df[query].index)
        assert size == 300, f'something is wrong. Size {size} is not 300'

        start = randomize_start()
        query = query & (df.seq >= start)
        df.loc[query, 'anomaly'] = True

        ## Now inject actual annomaly

        # get delta between median and noise floor.
        delta = abs(RSSI_BASE - df.loc[query, 'rss'].median())

        # Drop should be around 30 or less, depending on delta
        delta = delta if delta < 30 else (30 - rng.uniform(0, 3))

        # Apply drop where anomaly should have appeared
        df.loc[query & df.anomaly, 'rss'] -= delta

        # Fix values below noise floor
        df.loc[query & df.anomaly & (df['rss'] < RSSI_BASE), 'rss'] = RSSI_BASE

    assert np.any(df.original != df.rss)

    # Scale links afterwards
    if scaler:
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            df.loc[query, ['rss']] = scaler.fit_transform(df.loc[query, ['rss']])

    assert len(df.index) != 0
    return df


@memory.cache
def recovery_anomaly_injector(scaler=None, random_state=None) -> pd.DataFrame:
    # Get ready-to-use prepared dataset for anomaly injection
    df = get_initial_dataset()

    # Random number generator
    rng = np.random.default_rng(random_state)

    # Now, find unique pairs
    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
    n_pairs = len(unique_pairs)

    df['anomaly'] = False
    df['original'] = df.rss

    # approx. 33.3% of links will have anomaly
    n_anomaly_pairs = int(n_pairs // 3)

    # Random drop starts between 25th and 275th packet
    randomize_start = lambda: rng.integers(25, 275, endpoint=True)

    # Randomly recovers after 2 to 20 packets
    randomize_length = lambda: rng.integers(5, 20, endpoint=True)

    # Randomly select affected links
    affected_links = rng.choice(unique_pairs, n_anomaly_pairs)

    for src, dst, noise in affected_links:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
        size = len(df[query].index)
        assert size == 300, f'something is wrong. Size {size} is not 300'

        start = randomize_start()
        stop = start + randomize_length()
        query = query & (df.seq >= start) & (df.seq <= stop)
        df.loc[query, 'anomaly'] = True

        ## Now actually inject anomalies

        # get delta between median and noise floor.
        delta = abs(RSSI_BASE - df.loc[query, 'rss'].median())

        # Drop should be around 30 or less, depending on delta
        delta = delta if delta < 30 else (30 - rng.uniform(0, 3))

        # Apply drop where anomaly should have appeared
        df.loc[query & df.anomaly, 'rss'] -= delta

        # Fix values below noise floor
        df.loc[query & df.anomaly & (df['rss'] < RSSI_BASE), 'rss'] = RSSI_BASE

    assert np.any(df.original != df.rss)

    # Scale links afterwards
    if scaler:
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            df.loc[query, ['rss']] = scaler.fit_transform(df.loc[query, ['rss']])

    assert len(df.index) != 0
    return df


@memory.cache
def slow_anomaly_injector(scaler=None, random_state=None) -> pd.DataFrame:
    # Get ready-to-use prepared dataset for anomaly injection
    df = get_initial_dataset()

    # Random number generator
    rng = np.random.default_rng(random_state)

    # Now, find unique pairs
    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
    n_pairs = len(unique_pairs)

    df['anomaly'] = False
    df['original'] = df.rss

    # approx. 33.3% of links will have anomaly
    n_anomaly_pairs = int(n_pairs // 3)

    # randomize start of slow degradation between 10th and 200th packet
    rand_start = lambda: rng.integers(0, 20, endpoint=True)
    rand_duration = lambda: rng.integers(150, 280, endpoint=True)
    rand_rate = lambda: rng.uniform(0.5, 1.5, size=300) / -300.0

    def slope(seq, rate, start):
        # General curve
        curve = rate * (seq - start)
        curve[curve > 0] = 0  # Correction #1: Remove values above 0
        #curve[curve < (rate * (end - start))] = rate * (end - start) # Correction #2: stop falling after "end" value

        # Sanity checks
        assert np.all(rate < 0), f'Rate should have been negative. Got {rate}'

        return curve

    # Randomly select affected links
    affected_links = rng.choice(unique_pairs, n_anomaly_pairs)

    for src, dst, noise in affected_links:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
        size = len(df[query].index)
        assert size == 300, f'something is wrong. Size {size} is not 300'

        start = rand_start()
        end = start + rand_duration()
        rate = rand_rate()

        df.loc[query & (df.seq >= start), 'anomaly'] = True
        df.loc[query, 'rss'] += slope(seq=df.loc[query].seq, rate=rate, start=start)
        #df.loc[query, 'rss'] += factor(seq=df.loc[query,:].seq, offset=start)

    assert np.any(df.original != df.rss)

    # Scale links afterwards
    if scaler:
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            df.loc[query, ['rss']] = scaler.fit_transform(df.loc[query, ['rss']])

    assert len(df.index) != 0
    return df


@memory.cache
def get_agg_features_dataset(anomaly:str, scaler=None, random_state=None) -> pd.DataFrame:
    datasets = {
        'dummy': dummy_anomaly_injector,
        'spikes': spike_anomaly_injector,
        'norecovery': norecovery_anomaly_injector,
        'recovery': recovery_anomaly_injector,
        'slow': slow_anomaly_injector
    }

    assert anomaly in datasets.keys(), f'Invalid anomaly type: {anomaly}'
    df = datasets[anomaly](scaler=scaler, random_state=random_state)

    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()

    features = []

    for src, dst, noise in unique_pairs:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)

        view = df[query]

        mean = view.rss.mean()
        stddev = view.rss.std()
        median = view.rss.median()
        q25 = view.rss.quantile(.25)
        q50 = view.rss.quantile(.5) # this is mediana
        q75 = view.rss.quantile(.75)
        rmin = view.rss.min()
        rmax = view.rss.max()

        anomaly = view.anomaly.sum() > 0

        value = (mean, stddev, median, rmin, q25, q50, q75, rmax, anomaly)
        features.append(value)

    df = pd.DataFrame(features, columns=['MEAN', 'STD', 'MEDIAN', 'MIN', 'Q25', 'Q50', 'Q75', 'MAX', 'anomaly'])
    return df


@memory.cache
def get_agg_features_dataset_plus(anomaly:str, scaler=None, random_state=None) -> pd.DataFrame:
    datasets = {
        'dummy': dummy_anomaly_injector,
        'spikes': spike_anomaly_injector,
        'norecovery': norecovery_anomaly_injector,
        'recovery': recovery_anomaly_injector,
        'slow': slow_anomaly_injector
    }

    assert anomaly in datasets.keys(), f'Invalid anomaly type: {anomaly}'
    df = datasets[anomaly](scaler=scaler, random_state=random_state)

    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()

    features = []

    for src, dst, noise in unique_pairs:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)

        view = df[query]

        mean = view.rss.mean()
        stddev = view.rss.std()
        median = view.rss.median()
        q25 = view.rss.quantile(.25)
        q50 = view.rss.quantile(.5) # this is mediana
        q75 = view.rss.quantile(.75)
        rmin = view.rss.min()
        rmax = view.rss.max()

        skewness = sp.stats.skew(view.rss)
        kurtosis = sp.stats.kurtosis(view.rss)

        # TODO: Add min, max of first derivative

        anomaly = view.anomaly.sum() > 0

        value = (mean, stddev, median, rmin, q25, q50, q75, rmax, skewness, kurtosis, anomaly)
        features.append(value)

    df = pd.DataFrame(features, columns=['MEAN', 'STD', 'MEDIAN', 'MIN', 'Q25', 'Q50', 'Q75', 'MAX', 'SKEW', 'KURTOSIS', 'anomaly'])
    return df


@memory.cache
def get_histogram_features_dataset(anomaly:str, scaler=None, random_state=None) -> pd.DataFrame:
    datasets = {
        'dummy': dummy_anomaly_injector,
        'spikes': spike_anomaly_injector,
        'norecovery': norecovery_anomaly_injector,
        'recovery': recovery_anomaly_injector,
        'slow': slow_anomaly_injector,
    }

    assert anomaly in datasets.keys(), f'Invalid anomaly type: {anomaly}'
    df = datasets[anomaly](scaler=scaler, random_state=random_state)

    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()

    features = []
    hasAnomalies = []

    hmin, hmax = df.rss.min(), df.rss.max()

    for src, dst, noise in unique_pairs:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
        view = df[query]

        count, _ = np.histogram(view.rss.ravel(), bins=10, range=(hmin, hmax), density=False)
        features.append(count)

        anomalies_count = view.anomaly.sum()
        hasAnomalies.append(anomalies_count > 0)

    df = pd.DataFrame(data=features, columns=[f'bin{i+1}' for i in range(10)])
    df['anomaly'] = hasAnomalies

    return df


@memory.cache
def get_fft_features_dataset(anomaly:str, scaler=None, random_state=None) -> pd.DataFrame:
    datasets = {
        'dummy': dummy_anomaly_injector,
        'spikes': spike_anomaly_injector,
        'norecovery': norecovery_anomaly_injector,
        'recovery': recovery_anomaly_injector,
        'slow': slow_anomaly_injector,
    }

    assert anomaly in datasets.keys(), f'Invalid anomaly type: {anomaly}'
    df = datasets[anomaly](scaler=scaler, random_state=random_state)

    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()

    features = []
    hasAnomalies = []

    for src, dst, noise in unique_pairs:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)

        view = df[query]

        fft = np.abs(np.fft.fft(view.rss.ravel()))

        magnitude = np.abs(fft)[:150]
        phase = np.arctan2(np.imag(fft), np.real(fft))[:150]

        feature = np.concatenate([magnitude, phase])

        features.append(feature)

        anomalies_count = view.anomaly.sum()
        hasAnomalies.append(anomalies_count > 0)



    n = fft.size
    timestep = 30 / 300 # 300 packets in 30 secs
    freq = np.fft.fftfreq(n, d=timestep)[:150]

    df = pd.DataFrame(data=features, columns=[f'Mag@{i:.2f}Hz' for i in freq] + [f'Phase@{i:.2f}Hz' for i in freq])
    df['anomaly'] = hasAnomalies

    return df


@memory.cache
def get_ts_features_vector(anomaly:str, scaler=None, random_state=None) -> pd.DataFrame:
    datasets = {
        'dummy': dummy_anomaly_injector,
        'spikes': spike_anomaly_injector,
        'norecovery': norecovery_anomaly_injector,
        'recovery': recovery_anomaly_injector,
        'slow': slow_anomaly_injector,
    }

    assert anomaly in datasets.keys(), f'Invalid anomaly type: {anomaly}'
    df = datasets[anomaly](scaler=scaler, random_state=random_state)

    unique_pairs = get_unique_pairs() #df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()

    features = []
    hasAnomalies = []

    for src, dst, noise in unique_pairs:
        query = (df.src==src) & (df.dst==dst) & (df.noise==noise)

        view = df[query]

        features.append(view.rss.values)

        anomalies_count = view.anomaly.sum()
        hasAnomalies.append(anomalies_count > 0)

    df = pd.DataFrame(data=features)
    df.columns = [f'$X_{{{c}}}$' for c in df.columns]
    df['anomaly'] = hasAnomalies

    return df


@memory.cache
def unsupervised_isolation_forest_classifier(X, y, random_state=None) -> Tuple:
    # Grid of parameters to search for solution
    param_grid = dict(
        n_estimators=[10, 20, 30, 40, 50, 70, 100],
    )

    # Classification algorithm with sane defaults (in most cases they are the same as scikit-learn's)
    clf = ensemble.IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination='auto',
        max_features=1.0,
        bootstrap=False,
        n_jobs=-1,
        behaviour='deprecated',
        random_state=random_state,
        verbose=1,
        warm_start=False,
    )

    output = []
    for params in model_selection.ParameterGrid(param_grid):
        clf.set_params(**params)
        # Unsupervised approaches can directly perform prediction (without split into train-test)
        y_pred = clf.fit_predict(X) != 1 # Returns -1 for outliers and 1 for inliers.

        result = (y, y_pred, params)
        output.append(result)  # Appends results

    return output


@memory.cache
def supervised_random_forest_classifier(X, y, random_state=None):
    param_grid = dict(
        n_estimators=[10, 20, 30, 40, 50, 70, 100],
    )

    clf = BaggingClassifier(
        base_estimator=None,
        n_estimators=10,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        n_jobs=-1,
        random_state=random_state,
        verbose=1
    )

    # Supervised algorithms require train-test split. However, to preserve
    # dataset size, we will use StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    output = []
    for params in model_selection.ParameterGrid(param_grid):
        clf.set_params(**params)
        # Unsupervised approaches can directly perform prediction (without split into train-test)
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1, verbose=1)

        # Appends results
        result = (y, y_pred, params)
        output.append(result)

    return output


@memory.cache
def unsupervised_oneclass_svm_classifier(X, y, random_state=None):
    # Grid of parameters to search for solution
    param_grid = dict(
        kernel=['rbf', 'linear'],
        nu=[0.10, 0.3, 0.5, 0.70, 0.90, 1.0],
        gamma=('auto', 'scale'),
    )

    # Classification algorithm with sane defaults (in most cases they are the same as scikit-learn's)
    clf = svm.OneClassSVM(
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        tol=1e-3,
        nu=0.5,
        shrinking=True,
        cache_size=2048,
        verbose=False,
        max_iter=1_000
    )

    output = []
    for params in model_selection.ParameterGrid(param_grid):
        clf.set_params(**params)
        # Unsupervised approaches can directly perform prediction (without split into train-test)
        y_pred = clf.fit_predict(X) != 1 # Returns -1 for outliers and 1 for inliers.

        result = (y, y_pred, params)
        output.append(result)  # Appends results

    return output


@memory.cache
def supervised_svm_classifier(X, y, random_state=None):
    param_grid = dict(
        C=(1e-3, 1e-2, 1e-1, 1.0, 10., 100.),
        kernel=('linear', 'rbf'),
        gamma=('auto', 'scale'),
    )

    clf = SVC(
        C=1.0,
        kernel='rbf',
        degree=3,
        gamma='scale',
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=True,
        max_iter=1_000,
        decision_function_shape='ovr',
        break_ties=False,
        random_state=random_state
    )

    # Supervised algorithms require train-test split. However, to preserve
    # dataset size, we will use StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    output = []
    for params in model_selection.ParameterGrid(param_grid):
        clf.set_params(**params)
        # Unsupervised approaches can directly perform prediction (without split into train-test)
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1, verbose=1)

        # Appends results
        result = (y, y_pred, params)
        output.append(result)

    return output


@memory.cache
def unsupervised_local_outlier_factor_classifier(X, y, random_state=None):
    # Grid of parameters to search for solution
    param_grid = dict(
        n_neighbors=[5, 10, 20, 40, 50, 80],
        leaf_size=[10, 30, 50, 80],
        algorithm=['ball_tree', 'kd_tree', 'brute'],
        p=[1, 2]
    )

    # Classification algorithm with sane defaults (in most cases they are the same as scikit-learn's)
    clf = neighbors.LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        contamination="auto",
        novelty=False,
        n_jobs=-1
    )

    output = []
    for params in model_selection.ParameterGrid(param_grid):
        clf.set_params(**params)
        # Unsupervised approaches can directly perform prediction (without split into train-test)
        y_pred = clf.fit_predict(X) != 1 # Returns -1 for outliers and 1 for inliers.

        result = (y, y_pred, params)
        output.append(result)  # Appends results

    return output


def build_autoencoder_model(input_features, latent_conn=4, filters=[128, 64, 32]):
    # Building a model will be slower, but regularization will try to avoid extreme weights (and overfit on train data).
    kernel_regularizer = keras.regularizers.l2(l=0.01)

    x = inputs = layers.Input(shape=(input_features,))
    for f in filters:
        x = layers.Dense(f, kernel_regularizer=kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)


    latent_outputs = layers.Dense(latent_conn)(x)

    x = latent_inputs = layers.Input(shape=(latent_conn, ))
    for f in filters[::-1]:
        x = layers.Dense(f, kernel_regularizer=kernel_regularizer)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    decoder_outputs = layers.Dense(input_features)(x)

    # Encoder model
    encoder = keras.Model(inputs=inputs, outputs=latent_outputs, name='encoder')

    # Decoder model
    decoder = keras.Model(inputs=latent_inputs, outputs=decoder_outputs, name='decoder')

    # Combine encoder --> decoder into autoencoder
    autoencoder = keras.Model(inputs=inputs, outputs=decoder(encoder(inputs)), name='autoencoder')

    # Compile graph for TensorFlow 2.x
    autoencoder.compile(
        optimizer='adam',
        loss='MSE',
        metrics=['MSE'],
    )

    return encoder, decoder, autoencoder


@memory.cache
def encode_features(X, y, random_state=None):
    input_features = X.shape[1] # Figure out how many input features are there

    # get automatically build models for training
    encoder, decoder, autoencoder = build_autoencoder_model(
        input_features=input_features,
        latent_conn=4,
        filters=[128, 64, 32],
    )

    autoencoder.fit(X, y,
        epochs=1_000,
        shuffle=True,
        callbacks=[
            callbacks.EarlyStopping(patience=10, verbose=1, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.1, patience=3, verbose=1),
        ],
        validation_split=0.33,
        use_multiprocessing=True,
    )

    # Freeze autoencoder layers
    # TODO: Is freeze necessery when predict is performed
    for layer in autoencoder.layers:
        layer.trainable = False

    # TODO: Should I use more conservative approach and throw away AE's train data
    X_encoded = encoder.predict(X)

    return X_encoded, y


class AutoencoderTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, filters=(128, 64, 32), latent_conn=4):
        self.filters = filters
        self.latent_conn = latent_conn
        super(AutoencoderTransformer, self).__init__()

    def fit(self, X, y=None):
        # Figure out how many input features are there
        input_features = X.shape[1]

        encoder, decoder, autoencoder = build_autoencoder_model(
            input_features=input_features,
            latent_conn=self.latent_conn,
            filters=self.filters,
        )

        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder

        self.autoencoder.fit(X, y,
            epochs=1_000,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(monitor='loss', patience=15, verbose=1, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1),
            ],
            validation_split=0.33,
            use_multiprocessing=True,
        )

        # Freeze autoencoder layers
        # TODO: Is freeze necessery when predict is performed?
        for layer in self.autoencoder.layers:
            layer.trainable = False

        return self

    def transform(self, X, y=None):
        X_encoded = self.encoder.predict(X)
        return X_encoded




@memory.cache
def supervised_logistic_regression_classifier(X, y, random_state=None):
    param_grid = dict(
        C=(1e-3, 1e-2, 1e-1, 1.0, 10., 100.),
    )

    clf = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-4,
        C=1e-3,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=random_state,
        solver='lbfgs',
        max_iter=10_000,
        multi_class='auto',
        verbose=1,
        warm_start=False,
        n_jobs=-1,
        l1_ratio=None
    )

    # Supervised algorithms require train-test split. However, to preserve
    # dataset size, we will use StratifiedKFold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    output = []
    for params in model_selection.ParameterGrid(param_grid):
        clf.set_params(**params)
        # Unsupervised approaches can directly perform prediction (without split into train-test)
        y_pred = cross_val_predict(clf, X, y, cv=cv, n_jobs=-1, verbose=1)

        # Appends results
        result = (y, y_pred, params)
        output.append(result)

    return output


def threshold_strategies(random_state=None):
    """Plan (threshold):
        - [x] aggregated features: (abs(mean - median) < 3dBm) || (2*stdev(x) < 8dBm)
        - [x] histogram: x < 85dBm
        - [ ] timeseries batch: p < 10**-3
    """

    dummy = lambda: dummy_anomaly_injector(scaler=None, random_state=random_state)
    spikes = lambda: spike_anomaly_injector(scaler=None, random_state=random_state)
    norecovery = lambda: norecovery_anomaly_injector(scaler=None, random_state=random_state)
    recovery = lambda: recovery_anomaly_injector(scaler=None, random_state=random_state)
    slow = lambda: slow_anomaly_injector(scaler=None, random_state=random_state)

    datasets = {
        #'dummy': dummy,
        'norecovery': norecovery,
        'recovery': recovery,
        'spikes': spikes,
        'slow': slow,
    }

    lin2log = lambda x: 10 * np.log10(x)

    def aggr_approach(df, offset_threshold=3.0, stdev_threshold=2.5):
        y_true, y_pred = [], []
        unique_pairs = df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            view = df[query]
            x = view.rss.ravel()

            # Difference between mean and median has to be lower than threshold
            criteria1 = np.abs(np.mean(x) - np.median(x)) > offset_threshold

            # Deviation has to be lower than threshold
            criteria2 = 2 * np.std(x) > stdev_threshold
            #criteria2 = (np.mean(x) + 2*np.std()

            result = np.any(criteria1 | criteria2)
            #print(criteria1 + criteria2)

            y_pred.append(result)
            y_true.append(np.any(view['anomaly']))

            #print(np.any(view['anomaly']), 2*np.std(x))
            #print()

        return y_true, y_pred

    def hist_approach(df, threshold=-85.0):
        y_true, y_pred = [], []
        unique_pairs = df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            view = df[query]
            x = view.rss.ravel()

            result = np.any(x < threshold)

            y_pred.append(result)
            y_true.append(np.any(view['anomaly']))

        return y_true, y_pred

    def ts_as_feature_vector(df, alpha=1e-3):
        y_true, y_pred = [], []
        unique_pairs = df.apply(lambda row: (row['src'], row['dst'], row['noise']), axis=1).unique()
        for src, dst, noise in unique_pairs:
            query = (df.src==src) & (df.dst==dst) & (df.noise==noise)
            view = df[query]
            x = view.rss.ravel()

            k2, p = sp.stats.normaltest(x)
            result = (p < alpha) # if p < alpha, it is not normal distribution, therefore anomaly

            y_pred.append(result)
            y_true.append(np.any(view['anomaly']))

        return y_true, y_pred

    #def fft_approach(x):
    #    freq = np.abs(np.fft.fft(x))
    #    freq_db = lin2log(freq)
    #    # [N/2] - 1; frequency for each sample is i * samplingFrequency / N; 10Hz / (300 / 2 - 1)
    #    ratio = 300 // 5
    #    return (np.sum(freq_db[:ratio] > -20.0) > ratio // 2)


    #df = norecovery_anomaly_injector(scaler=None, random_state=SEED)
    for dataset_name, dataset in datasets.items():
        string = f'% computer generated: anomaly={dataset_name}\nBaseline & Threshold (Tab.~\\ref{{tab:threshold-config}})'
        for name, func in (('time-value', ts_as_feature_vector), ('aggr', aggr_approach), ('hist', hist_approach)):
            df = dataset()
            y_true, y_pred = func(df)
            #print(metrics.classification_report(y_true=y_true, y_pred=y_pred))
            prec = metrics.precision_score(y_true, y_pred, labels=[False, True])
            rec = metrics.recall_score(y_true, y_pred, labels=[False, True])
            f1 = metrics.f1_score(y_true, y_pred, labels=[False, True])

            string += f'\t& {prec:.2f} & {rec:.2f} & {f1:.2f}\\tnote{{1}} &'

        string = string + '& - & - ' + '\\\\'
        print(string)



def research_explain():
    from sklearn.tree import DecisionTreeClassifier

    scalers = {
        'None': StandardScaler(with_mean=False, with_std=False),
        'StdScalerMean': StandardScaler(with_mean=True, with_std=False),
        'StdScalerAll': StandardScaler(with_mean=True, with_std=True),
        #'MinMaxScaler': MinMaxScaler(feature_range=(0, 1)),
        'RobustScalerMean': RobustScaler(with_centering=True, with_scaling=False),
        'RobustScalerAll': RobustScaler(with_centering=True, with_scaling=True),
    }

    features = {
        'aggr': get_agg_features_dataset,
        #'ts': get_ts_features_vector,
        #'fft': get_fft_features_dataset,
        'hist': get_histogram_features_dataset,
    }

    get_initial_dataset()
    get_unique_pairs()

    # Start parallel "dummy" calculation of generate datasets
    tasks = []
    for scaler in scalers.values():
        for anomaly_name in ['spikes', 'norecovery', 'recovery']:
            for featureset in features.values():
                task = delayed(featureset)(anomaly_name, scaler, random_state=SEED)
                tasks.append(task)

    # Run, run, run
    Parallel(n_jobs=-1)(tasks)

    param_grid = dict(
        max_depth=(1, 3, 5, 7, 9, None),
        min_samples_split=(30, 40, 50, 70, 100),
        #min_samples_leaf=(30, 40, 50, 70, 100),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=SEED,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        class_weight=None,
        presort='deprecated',
        ccp_alpha=0.0
    )

    param_grid = {
        'C': (1e-3, 1e-2, 1e-1, 1.0, 10., 100.),
    }

    clf = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=1e-4,
        C=1e-3,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight=None,
        random_state=SEED,
        solver='lbfgs',
        max_iter=1_000,
        multi_class='auto',
        verbose=0,
        warm_start=False,
        n_jobs=-1,
        l1_ratio=None
    )

    results = []

    for anomaly_name in ['spikes', 'norecovery', 'recovery']:
        for featureset_name, featureset in features.items():

            best_f1 = None
            best_params = None
            best_scaler = None
            for scaler_name, scaler in scalers.items():
                model = model_selection.GridSearchCV(
                    estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
                )

                df = featureset(anomaly_name, scaler, random_state=SEED)
                X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()

                model.fit(X, y)
                best_model = model.best_estimator_

                y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1, verbose=1)
                f1 = metrics.f1_score(y, y_pred, labels=[False, True])
                if best_f1 is None or best_f1 < f1:
                    best_f1 = f1
                    best_params = model.best_params_
                    best_scaler = scaler_name

            results.append((anomaly_name, featureset_name, best_scaler, best_f1, best_params))

    results = pd.DataFrame(results, columns=('anomaly', 'features', 'scaler', 'F1', 'params'))
    print(results)




def main_logatec():
    from datasets.trace6_logatec.transform import get_traces as get_logatec_traces
    import matplotlib.pyplot as plt
    plt.style.use(['science', 'ieee'])
    pre_encoder_scaler = per_link_scaler = dataset_scaler = encoder = None


    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    def random_forest():
        param_grid = dict(
            #n_estimators=(50, 75, 100, 200, 500),
            criterion=('gini', 'entropy'),
            max_depth=(1, 3, 5, 7, 9, None),
            min_samples_split=(30, 40, 50, 70, 100),
            min_samples_leaf=(30, 40, 50, 70, 100),
            ccp_alpha=np.arange(0, 1.0, step=0.2),
        )


        clf = ensemble.RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True, oob_score=False,
            n_jobs=-1,
            random_state=SEED,
            verbose=1,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None
        )

        model = model_selection.GridSearchCV(
            estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
        )

        return model


    def decision_tree():

        param_grid = dict(
            #n_estimators=(50, 75, 100, 200, 500),
            criterion=('gini', 'entropy'),
            max_depth=(1, 3, 5, 7, 9, None),
            min_samples_split=(10, 30, 40, 50, 70, 100),
            min_samples_leaf=(15, 30, 40, 50, 70, 100),
            ccp_alpha=np.arange(0, 1.0, step=0.2),
        )

        clf = tree.DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features=None,
            random_state=SEED,
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            class_weight=None,
            presort='deprecated',
            ccp_alpha=0.0
        )

        model = model_selection.GridSearchCV(
            estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
        )

        return model

    def get_svm_model():
        param_grid = dict(
            C=(1e-3, 1e-2, 1e-1, 1.0, 10., 100.),
            kernel=('linear', 'rbf'),
            gamma=('auto', 'scale'),
        )

        clf = SVC(
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=True,
            max_iter=10_000,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=SEED
        )

        model = model_selection.GridSearchCV(
            estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
        )

        return model


    def logistic_regression():
        param_grid = dict(
            C=(1e-3, 1e-2, 1e-1, 1.0, 10., 100.),
        )

        clf = LogisticRegression(
            penalty='l2',
            dual=False,
            tol=1e-4,
            C=1e-3,
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            random_state=SEED,
            solver='lbfgs',
            max_iter=10_000,
            multi_class='auto',
            verbose=1,
            warm_start=False,
            n_jobs=-1,
            l1_ratio=None
        )


        model = model_selection.GridSearchCV(
            estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
        )

        return model


    def get_iforest_model():
        param_grid = dict(
            n_estimators=[10, 20, 30, 40, 50, 70, 100],
        )

        # Classification algorithm with sane defaults (in most cases they are the same as scikit-learn's)
        clf = ensemble.IsolationForest(
            n_estimators=100,
            max_samples='auto',
            contamination='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            behaviour='deprecated',
            random_state=SEED,
            verbose=1,
            warm_start=False,
        )


        model = model_selection.GridSearchCV(
            estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
        )

        return model

    #per_link_scaler = preprocessing.StandardScaler(with_mean=True, with_std=False)
    #per_link_scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=False)
    #per_link_scaler = preprocessing.MinMaxScaler()
    #pre_encoder_scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=True)
    #encoder = AutoencoderTransformer()
    dataset_scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=False)
    #dataset_scaler = preprocessing.MinMaxScaler()
    model = logistic_regression()
    #model = random_forest()
    #model = get_svm_model()
    #model = get_iforest_model()

    df = get_ts_features_vector(anomaly='norecovery', scaler=per_link_scaler, random_state=SEED)
    X_train, y_train = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()


    if encoder:
        if pre_encoder_scaler:
            pre_encoder_scaler.fit_transform(X_train)

        #encoder = AutoencoderTransformer()
        encoder.fit(X_train)
        X_train = encoder.transform(X_train)

    if dataset_scaler:
        X_train = dataset_scaler.fit_transform(X_train)


    model.fit(X_train, y_train)


    # Limit the size of input to 300, which is size of Rutgers tracesets
    traces = get_logatec_traces(n_jobs=-1)

    X_test = []
    for trace in traces:
        if len(trace.index) < 300:
            continue

        _x = trace['rss_avg'].iloc[:300].to_numpy().reshape(1, -1)
        assert _x.shape == (1, 300), f'(1, 300) != {_x.shape}'
        if per_link_scaler:
            _x = per_link_scaler.fit_transform(_x)
        X_test.append(_x)

    X_test = np.concatenate(X_test)
    assert X_test.shape == (11, 300), f'(11, 300) != {X_test.shape}'

    if encoder:
        if pre_encoder_scaler:
            pre_encoder_scaler.transform(X_test)

        X_test = encoder.transform(X_test)

    if dataset_scaler:
        X_test = dataset_scaler.transform(X_test)

    y_pred = model.predict(X_test)


    ## Plot method
    for trace, is_anomaly in zip(traces, y_pred):
        f, ax = plt.subplots()
        src, dst = trace['src'].iloc[0], trace['dst'].iloc[0] # extract SRC and DST
        title = f'N{src} $\\rightarrow$ N{dst}'
        c = 'g' # default is green
        if is_anomaly:
            c = 'r'
            title += ' is anomalous!'

        #ax.set_title(title)
        ax.set(xlim=(0, 300), ylim=(-95, 0), xlabel='packet sequence', ylabel='RSS [dBm]')
        ax.plot(trace.index, trace['rss_avg']) # c=c
        f.savefig(f'./tomazs-detection-n{src}-n{dst}.png')

    # f, axes = plt.subplots(nrows=3, ncols=4, figsize=(16,9))
    # for ax, trace, is_anomaly in zip(axes.flatten(), traces, y_pred):
    #     src, dst = trace['src'].iloc[0], trace['dst'].iloc[0]
    #     title = f'Link {src} --> {dst}'

    #     c = 'g'
    #     if is_anomaly:
    #         c = 'r'
    #         title += ' (ANOMALOUS!)'

    #     ax.set_title(title)
    #     ax.set_ylim(-95, 0)

    #     ax.plot(trace.index, trace['rss_avg'], c=c)

    # f.savefig('./tomazs_iforest.png')
    # plt.show()



def visualize_autoencoder_feature_space():
    import seaborn as sns
    plt.style.use(['science', 'ieee'])

    for anomaly in ('slow', 'spikes', 'recovery', 'norecovery'):
    #for anomaly in ('spikes', ):

        scaler = None
        scaler = RobustScaler(with_centering=True, with_scaling=True)
        #scaler = StandardScaler(with_mean=True, with_std=False)
        df = get_ts_features_vector(anomaly=anomaly, scaler=scaler, random_state=SEED)

        input_features = len(df.columns) - 1

        encoder, decoder, autoencoder = build_autoencoder_model(
            input_features=input_features, latent_conn=4, filters=[64, 32, 16]
        )

        X, y = df.drop('anomaly', axis=1).to_numpy(), df.anomaly.ravel()
        #X = RobustScaler(with_centering=True, with_scaling=True).fit_transform(X)
        #X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

        autoencoder.fit(X, X,
            epochs=1000,
            shuffle=True,
            callbacks=[
                callbacks.EarlyStopping(patience=10, verbose=0, restore_best_weights=True),
                callbacks.ReduceLROnPlateau(patience=3, verbose=0),
            ],
            validation_split=0.33,
            use_multiprocessing=True,
        )

        columns = list(df.columns)[:-1]
        X = df[df.anomaly==False][columns].iloc[2].to_numpy()
        X_ = np.expand_dims(X, axis=0)

        X_enc = encoder.predict(X_)

        rng = np.random.default_rng(seed=SEED)

        if anomaly == 'spikes':
            # The probability for spike anomaly to appear is:
            anomaly_probability = 3. / 300.

            mask = np.random.choice([False, True], p=[1. - anomaly_probability, anomaly_probability], size=300)

            delta = abs(RSSI_BASE - np.median(X))
            delta = delta if delta < 30 else (30 - rng.uniform(0, 3))
            X[mask] -= delta

        elif anomaly == 'norecovery':
            start = rng.integers(200, 280, endpoint=True)

            delta = abs(RSSI_BASE - np.median(X))
            delta = delta if delta < 30 else (30 - rng.uniform(0, 3))
            X[start:] -= delta

        elif anomaly == 'recovery':
            start = rng.integers(25, 275, endpoint=True)
            duration = rng.integers(5, 20, endpoint=True)

            delta = abs(RSSI_BASE - np.median(X))
            delta = delta if delta < 30 else (30 - rng.uniform(0, 3))
            X[start:start+duration] -= delta


        elif anomaly == 'slow':
            # randomize start of slow degradation between 10th and 200th packet
            start = rng.integers(0, 20, endpoint=True)
            duration = rng.integers(150, 280, endpoint=True)
            rate = rng.uniform(0.5, 1.5, size=300) / -300.0

            def slope(seq, rate, start):
                # General curve
                curve = rate * (seq - start)
                curve[curve > 0] = 0  # Correction #1: Remove values above 0
                #curve[curve < (rate * (end - start))] = rate * (end - start) # Correction #2: stop falling after "end" value

                # Sanity checks
                assert np.all(rate < 0), f'Rate should have been negative. Got {rate}'

                return curve

            X = X + slope(np.array(range(300)), rate, start)
        else:
            raise ValueError('Invald anomaly')


        X[X<RSSI_BASE] = RSSI_BASE
        Xanom_ = np.expand_dims(X, axis=0)

        Xanom_enc = encoder.predict(Xanom_)
        #ax.plot(X_enc[0], alpha=0.75, label='True')



        f, ax = plt.subplots()

        names = ['\\#1','\\#2', '\\#3', '\\#4']

        ax.plot(names, X_enc[0], marker='x', linestyle=':', label='False')
        ax.plot(names, Xanom_enc[0], marker='o', linestyle=':', label='True')
        #ax.set_xticks([1,2,3,4])

        ax.set_xlabel('Encoded features (\\#)')
        ax.legend(title='Anomalous')

        f.savefig(f'./../figures/anomalies/autoencoder-featurespace-{anomaly}-ts.pdf')

        #plt.show()

    #y_pred = IsolationForest(n_estimators=20, n_jobs=-1, random_state=SEED).fit_predict(X_enc, y) != 1
    #print(
    #    metrics.precision_score(y, y_pred),
    #    metrics.recall_score(y, y_pred),
    #    metrics.f1_score(y, y_pred),
    #    metrics.roc_auc_score(y, y_pred),
    #)

    #y_pred = model_selection.cross_val_predict(
    #    estimator=SVC(gamma='auto', random_state=SEED, C=10.0),
    #    X=X_enc,
    #    y=y,
    #    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    #)

    #y_pred = SVC(random_state=SEED).fit(x_train, y_train).predict(x_test)
    #print(
    #    metrics.precision_score(y, y_pred),
    #    metrics.recall_score(y, y_pred),
    #    metrics.f1_score(y, y_pred),
    #    metrics.roc_auc_score(y, y_pred),
    #)


    #data = []
    #for i in range(X_enc.shape[0]):
    #    for j in range(4):
    #        data.append([f'\#{j+1}', X_enc[i,j], y[i]])

    #data = pd.DataFrame(data, columns=['feature space', 'value', 'Anomalous'])

    #f, ax = plt.subplots()
    #sns.violinplot(data=data, x='feature space', y='value', hue='Anomalous', ax=ax)
    #for i in range(X.shape[0]):
    #    ax.plot(X_enc[i,:], c=('r' if y[i] else 'g'), linestyle='-', alpha=0.1)
    #f.savefig(f'./../figures/anomalies/autoencoder-featurespace-{anomaly}-fft.pdf')
    #plt.show(f)





def visualize_feature_space():
    import seaborn as sns
    plt.style.use(['science', 'ieee'])

    anomaly='slow'

    scaler = None
    df = get_ts_features_vector(anomaly=anomaly, scaler=scaler, random_state=SEED)

    input_features = len(df.columns) - 1

    rng = np.random.default_rng(seed=SEED)
    # randomize start of slow degradation between 10th and 200th packet
    rand_start = lambda: rng.integers(0, 20, endpoint=True)
    rand_duration = lambda: rng.integers(150, 280, endpoint=True)
    rand_rate = lambda: rng.uniform(0.5, 1.5, size=300) / -300.0

    def slope(seq, rate, start):
        # General curve
        curve = rate * (seq - start)
        curve[curve > 0] = 0  # Correction #1: Remove values above 0
        #curve[curve < (rate * (end - start))] = rate * (end - start) # Correction #2: stop falling after "end" value

        # Sanity checks
        assert np.all(rate < 0), f'Rate should have been negative. Got {rate}'

        return curve


    f, ax = plt.subplots()

    columns = list(df.columns)[:-1]
    X = df[df.anomaly==False][columns].iloc[1].to_numpy()
    Xanom = X + slope(np.array(range(300)), rand_rate(), rand_start())
    ax.plot(X, alpha=0.75, label='False')
    ax.plot(Xanom, alpha=0.75, label='True')

    ax.set_ylabel('RSS [dBm]')
    ax.set_xlabel('packet ID')
    ax.legend(title='Anomaly')


    #f, ax = plt.subplots()
    #sns.violinplot(data=data, x='feature space', y='value', hue='Anomalous', ax=ax)
    #for i in range(X.shape[0]):
    #    ax.plot(X_enc[i,:], c=('r' if y[i] else 'g'), linestyle='-', alpha=0.1)
    f.savefig(f'./../figures/anomalies/featurespace-{anomaly}-ts.pdf')
    #plt.show(f)



    F = np.abs(np.fft.fft(X))
    Fanom = np.abs(np.fft.fft(Xanom))

    f, ax = plt.subplots()

    columns = list(df.columns)[:-1]
    X = df[df.anomaly==False][columns].iloc[1].to_numpy()
    ax.magnitude_spectrum(X, Fs=10, scale='dB', alpha=0.75, label='False')
    ax.magnitude_spectrum(Xanom, Fs=10, scale='dB', alpha=0.75, label='True')
    #ax.plot(X+slope(np.array(range(300)), rand_rate(), rand_start()), alpha=0.75)

    ax.set_ylabel('Magnitude [dB]')
    ax.set_xlabel('Frequency [Hz]')
    ax.legend(title='Anomaly')


    #f, ax = plt.subplots()
    #sns.violinplot(data=data, x='feature space', y='value', hue='Anomalous', ax=ax)
    #for i in range(X.shape[0]):
    #    ax.plot(X_enc[i,:], c=('r' if y[i] else 'g'), linestyle='-', alpha=0.1)
    f.savefig(f'./../figures/anomalies/featurespace-{anomaly}-fft.pdf')
    plt.show(f)


def explore_decision():
    #import seaborn as sns
    from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
    import graphviz

    #plt.style.use(['science', 'ieee', 'no-latex'])
    #plt.style.use(['ieee', 'no-latex'])

    param_grid = dict(
        criterion=('gini', 'entropy'),
        max_depth=(1, 3, 5, 7, 9, None),
        min_samples_split=(30, 40, 50, 70, 100),
        min_samples_leaf=(30, 40, 50, 70, 100),
        ccp_alpha=np.arange(0, 1.0, step=0.1),
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    clf = DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=SEED,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        class_weight=None,
        presort='deprecated',
        ccp_alpha=0.0
    )

    anomaly_type = 'norecovery'

    per_link_scaler = None
    dataset_scaler = None

    per_link_scaler = RobustScaler(with_centering=True, with_scaling=False)

    dataset_scaler = RobustScaler(with_centering=True, with_scaling=False)
    #dataset_scaler = RobustScaler(with_centering=True, with_scaling=False)

    df = get_ts_features_vector(anomaly=anomaly_type, scaler=per_link_scaler, random_state=SEED)
    #df = get_agg_features_dataset(anomaly=anomaly_type, scaler=None, random_state=SEED)
    #df = get_fft_features_dataset(anomaly=anomaly_type, scaler=None, random_state=SEED)
    #df = get_histogram_features_dataset(anomaly=anomaly_type, scaler=None, random_state=SEED)

    X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()
    # Encode or not ...
    #X, y = encode_features(X, y, random_state=SEED)
    #X = pd.DataFrame(X, columns=('#1', '#2', '#3', '#4'))

    #columns = list(X.columns)
    #if isinstance(columns[0], (float)) or 'MIN' in columns:
    #    columns = [f'$X_{{{c}}}$' for c in columns]

    columns = X.columns

    if dataset_scaler:
        X = dataset_scaler.fit_transform(X)

    model = model_selection.GridSearchCV(
        estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv
    )
    model.fit(X, y)
    best_model = model.best_estimator_
    print(best_model.get_params())

    y_pred = cross_val_predict(best_model, X, y, cv=cv, n_jobs=-1, verbose=1)

    f1 = metrics.f1_score(y, y_pred, labels=['No', 'Yes'])
    print(f'F1 = {f1:.2f}')

    f, ax = plt.subplots(figsize=(8, 8), dpi=192)
    plot_tree(best_model, feature_names=columns, class_names=['No', 'Yes'], max_depth=3, filled=True, ax=ax)
    #ax.set_title(f'Is it anomaly?', fontsize=10)
    #f.tight_layout()
    plt.show()
    #f.savefig(f'../figures/agg_{anomaly_type}_dtree_explain.png')


    # Filter incorrectly classified samples
    #df = X.copy()
    #df['true'] = y
    #df['pred'] = y_pred
    #df = df[~df['true'] & df['pred']]


    #f, ax = plt.subplots(dpi=192)
    #
    #for idx, row in df[df['true'] & ~df['pred']].head(1).iterrows():
    #    x = [row[i] for i in range(300)]
    #    ax.plot(x, alpha=0.4)

    #plt.show()


def research_to_explain():
    import graphviz
    from dtreeviz.trees import dtreeviz

    #plt.style.use(['science', 'ieee', 'no-latex'])
    #plt.style.use(['ieee', 'no-latex'])
    cv = model_selection.StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

    param_grid = dict(
        criterion=('gini', 'entropy'),
        max_depth=(1, 3, 5, 7, 9, None),
        min_samples_split=(30, 40, 50, 70, 100),
        min_samples_leaf=(30, 40, 50, 70, 100),
        ccp_alpha=np.arange(0, 1.0, step=0.2),
    )


    clf = tree.DecisionTreeClassifier(
        criterion="gini",
        splitter="best",
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.,
        max_features=None,
        random_state=SEED,
        max_leaf_nodes=None,
        min_impurity_decrease=0.,
        min_impurity_split=None,
        class_weight=None,
        presort='deprecated',
        ccp_alpha=0.0
    )

    link_scaler = preprocessing.RobustScaler(with_centering=True, with_scaling=False)
    scaler = preprocessing.StandardScaler(with_mean=True, with_std=True)

    features = {
        'aggr': get_agg_features_dataset,
        'ts': get_ts_features_vector,
        'fft': get_fft_features_dataset,
        'hist': get_histogram_features_dataset,
    }



    YELLOW = '#fefecd'
    GREEN = '#cfe2d4'
    DARKBLUE = '#313695'
    BLUE = '#4575b4'
    DARKGREEN = '#006400'
    LIGHTORANGE = '#fee090'
    LIGHTBLUE = '#a6bddb'
    GREY = '#444443'
    WEDGE_COLOR = GREY

    HIGHLIGHT_COLOR = '#D67C03'

    colors = [
        None,  # 0 classes
        None,  # 1 class
        #['#47a2e6', '#E4803B'],  # 2 classes
        ['#0c7f00', '#ff0000'],
        ['#FEFEBB', '#D9E6F5', '#a1dab4'],  # 3 classes
        ['#FEFEBB', '#D9E6F5', '#a1dab4', LIGHTORANGE],  # 4
        ['#FEFEBB', '#D9E6F5', '#a1dab4', '#41b6c4', LIGHTORANGE],  # 5
        ['#FEFEBB', '#c7e9b4', '#41b6c4', '#2c7fb8', LIGHTORANGE, '#f46d43'],  # 6
        ['#FEFEBB', '#c7e9b4', '#7fcdbb', '#41b6c4', '#225ea8', '#fdae61', '#f46d43'],  # 7
        ['#FEFEBB', '#edf8b1', '#c7e9b4', '#7fcdbb', '#1d91c0', '#225ea8', '#fdae61', '#f46d43'],  # 8
        ['#FEFEBB', '#c7e9b4', '#41b6c4', '#74add1', BLUE, DARKBLUE, LIGHTORANGE, '#fdae61', '#f46d43'],  # 9
        ['#FEFEBB', '#c7e9b4', '#41b6c4', '#74add1', BLUE, DARKBLUE, LIGHTORANGE, '#fdae61', '#f46d43', '#d73027']  # 10
    ]


    for anomaly_name in ('norecovery', 'recovery', 'spikes', 'slow'):
        for features_name, featureset in features.items():
            df = featureset(anomaly=anomaly_name, scaler=link_scaler, random_state=SEED)
            X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()

            columns = X.columns

            if scaler:
                X = scaler.fit_transform(X)

            model = model_selection.GridSearchCV(
                estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv, verbose=1,
            )

            model.fit(X, y)
            best_model = model.best_estimator_
            print(best_model.get_params())

            y_pred = model_selection.cross_val_predict(
                best_model, X, y, cv=cv, n_jobs=-1, verbose=1
            )

            f1 = metrics.f1_score(y, y_pred, labels=['No', 'Yes'])
            print(f'F1 = {f1:.2f}')

            #f, ax = plt.subplots(figsize=(12,12), dpi=192)
            #tree.plot_tree(best_model, feature_names=columns, class_names=['No', 'Yes'], max_depth=3, filled=True, ax=ax)
            #ax.set_title(f'Settings: {anomaly_name}; {features_name}; Scaled: Yes; F1={f1:.2f}', fontsize=10)
            #f.tight_layout()
            #f.savefig(f'./figures/explain/{anomaly_name}-{features_name}-scaled.png')
            dot_data = tree.export_graphviz(best_model, out_file=None, feature_names=columns, class_names=['No', 'Yes'], max_depth=3, filled=True)
            graph = graphviz.Source(dot_data)
            graph.render(directory='./figures/explain/', filename=f'{anomaly_name}-{features_name}-scaled', format='pdf')

            viz = dtreeviz(best_model, X, y, target_name='Anomaly?', feature_names=columns, class_names=['No', 'Yes'], histtype='strip', colors=dict(classes=colors))
            viz.save(f'./figures/explain/adv-{anomaly_name}-{features_name}-scaled.svg')





            df = featureset(anomaly=anomaly_name, scaler=None, random_state=SEED)
            X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()

            columns = X.columns

            model = model_selection.GridSearchCV(
                estimator=clf, param_grid=param_grid, scoring='f1', n_jobs=-1, cv=cv, verbose=1,
            )

            model.fit(X, y)
            best_model = model.best_estimator_
            print(best_model.get_params())

            y_pred = model_selection.cross_val_predict(
                best_model, X, y, cv=cv, n_jobs=-1, verbose=1
            )

            f1 = metrics.f1_score(y, y_pred, labels=['No', 'Yes'])
            print(f'F1 = {f1:.2f}')

            #f, ax = plt.subplots(figsize=(12,12), dpi=192)
            #tree.plot_tree(best_model, feature_names=columns, class_names=['No', 'Yes'], max_depth=3, filled=True, ax=ax)
            #ax.set_title(f'Settings: {anomaly_name}; {features_name}; Scaled: Yes; F1={f1:.2f}', fontsize=10)
            #f.tight_layout()
            #f.savefig(f'./figures/explain/{anomaly_name}-{features_name}-scaled.png')
            dot_data = tree.export_graphviz(best_model, out_file=None, feature_names=columns, class_names=['No', 'Yes'], max_depth=3, filled=True)
            graph = graphviz.Source(dot_data)
            graph.render(directory='./figures/explain/', filename=f'{anomaly_name}-{features_name}', format='pdf')

            viz = dtreeviz(best_model, X, y, target_name='Anomaly?', feature_names=columns, class_names=['No', 'Yes'], histtype='strip', colors=dict(classes=colors))
            viz.save(f'./figures/explain/adv-{anomaly_name}-{features_name}.svg')





def main():
    scalers = {
        'None': lambda: StandardScaler(with_mean=False, with_std=False),
        'StdScalerMean': lambda: StandardScaler(with_mean=True, with_std=False),
        'StdScalerAll': lambda: StandardScaler(with_mean=True, with_std=True),
        'MinMaxScaler': lambda: MinMaxScaler(feature_range=(0, 1)),
        'RobustScalerMean': lambda: RobustScaler(with_centering=True, with_scaling=False),
        'RobustScalerAll': lambda: RobustScaler(with_centering=True, with_scaling=True),
    }

    features = {
        'aggr': get_agg_features_dataset,
        'ts': get_ts_features_vector,
        'fft': get_fft_features_dataset,
        'hist': get_histogram_features_dataset,
    }

    classifiers = {
        'RForest': supervised_random_forest_classifier,
        'SVM': supervised_svm_classifier,
        'logreg': supervised_logistic_regression_classifier,
        'OCSVM': unsupervised_oneclass_svm_classifier,
        'IForest': unsupervised_isolation_forest_classifier,
        'LOF': unsupervised_local_outlier_factor_classifier,
    }

    # # REMOVE
    # import matplotlib.pyplot as plt
    # df = get_ts_features_vector('slow', None, random_state=SEED)
    # for idx, sample in df.iterrows():
    #     if sample['anomaly']:
    #         f, ax = plt.subplots()
    #         ax.plot([sample[i] for i in range(300)], label=f'sample #{idx}')
    #         f.savefig(f'./bak/slow-sample{idx}.png')
    #         print([sample[i] for i in range(300)][:5])

    #     if idx > 30:
    #         break

    # raise

    #print(df.columns)
    #X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()
    #X, y = encode_features(X, y, random_state=SEED)
    #print(X.columns)
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y)
    #model = ensemble.RandomForestClassifier(max_depth=None, min_samples_leaf=30, n_estimators=30, random_state=SEED)
    #model.fit(X_train, y_train)
    #y_pred = model.predict(X_test)
    #
    #print('F1', metrics.f1_score(y_test, y_pred))
    #from sklearn.tree import export_graphviz
    #from subprocess import call
    #export_graphviz(model.estimators_[0], out_file='tree.dot',
    #            rounded = True, proportion = False,
    #            precision = 2, filled = True)
    #call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    #raise

    # Warm-up caches

    # These two will prevent inefficiency where every process want to make cache for their outputs (only at cold-start)
    get_initial_dataset()
    get_unique_pairs()

    # Start parallel "dummy" calculation of generate datasets
    tasks = []
    for scaler in scalers.values():
        for anomaly_name in ['spikes', 'norecovery', 'recovery', 'slow']:
            for featureset in features.values():
                task = delayed(featureset)(anomaly_name, scaler(), random_state=SEED)
                tasks.append(task)

    # Run, run, run
    Parallel(n_jobs=-1)(tasks)


    inputs = []
    tasks = []
    results = []

    for link_scaler_name, link_scaler in scalers.items():
        for scaler_name, scaler in scalers.items():
            for anomaly_name in ['spikes', 'norecovery', 'recovery', 'slow']:
                for featureset_name, featureset in features.items():
                    for classifier_name, classifier in classifiers.items():
                        # Without autoencoders
                        input = dict(
                            name=classifier_name,
                            scaler=scaler_name,
                            anomaly=anomaly_name,
                            features=featureset_name,
                            link_scaler=link_scaler_name,
                        )
                        inputs.append(input)

                        df = featureset(anomaly_name, link_scaler(), random_state=SEED)
                        X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()
                        X = scaler().fit_transform(X)

                        task = delayed(classifier)(X, y, random_state=SEED)
                        tasks.append(task)


                        # With autoencoders
                        input = dict(
                            name='encoder+'+classifier_name,
                            scaler=scaler_name,
                            anomaly=anomaly_name,
                            features=featureset_name,
                            link_scaler=link_scaler_name,
                        )
                        inputs.append(input)

                        df = featureset(anomaly_name, link_scaler(), random_state=SEED)
                        X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()
                        X = scaler().fit_transform(X)
                        X, y = encode_features(X, y, random_state=SEED)

                        task = delayed(classifier)(X, y, random_state=SEED)
                        tasks.append(task)


    # Run all tasks in parallel
    interm = Parallel(n_jobs=-1)(tasks)
    for input, task in zip(inputs, interm):
        for (y_true, y_pred, params) in task:
            scores = dict(
                precision = metrics.precision_score(y_true, y_pred, labels=[False, True]),
                recall = metrics.recall_score(y_true, y_pred, labels=[False, True]),
                f1 = metrics.f1_score(y_true, y_pred, labels=[False, True]),
            )

            result = {**input, **scores, **params}
            results.append(result)


    results = pd.DataFrame(results)
    results.to_parquet('./20200907-anomaly_results.parquet', compression='gzip')
    #results.to_csv('./anomaly_results.csv')


def plot_random_things():
    plt.style.use(['science', 'ieee', 'no-latex'])

    df = get_ts_features_vector('norecovery', scaler=None, random_state=SEED)
    df = df[df.anomaly]
    X, y = df.drop('anomaly', axis=1).copy(), df.anomaly.ravel()


    f, ax = plt.subplots()
    ax.axvline(x=200, linestyle=':', color='r', linewidth=1, zorder=100, alpha=0.5)

    for i, row in X.iterrows():
        ax.plot(np.arange(300), row[:], alpha=.1, c='black', linestyle='-', linewidth=0.8)

    ax.set(xlim=(0,300), xlabel='packet number', ylabel='RSS [dBm]')

    f.savefig('./sample.png')
    plt.show()




if __name__ == "__main__":
    #threshold_strategies(random_state=SEED)
    #raise

    #main()

    main_logatec()

    #research_explain()

    #explore_decision()

    #research_to_explain()

    #plot_random_things()

    #visualize_autoencoder_feature_space()
    #visualize_feature_space()
