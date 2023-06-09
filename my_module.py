import pandas as pd
import numpy as np
import scipy.stats as stats
import logging
import unittest

from matplotlib import pyplot as plt
from sklearn import metrics

# CHANGE DATA


def delete_nan_column(df: pd.DataFrame, threshold: float, NaN: str = np.nan):
    """
    Delete all the columns in which the number of nan values
    more than the threshold (in per cent)

    Args:
        df: data frame containing the features
        threshold: value from 0 to 1
        NaN: string nan value, if it is not np.nan
    Return:
        df_new: df without the deleted columns
    """
    df = df.apply(lambda x: np.NaN if str(x) == NaN else x)
    df_nulls = df.isnull().sum()
    limit = df.shape[0] * (1 - threshold)
    cols_to_delete = []
    for col in df_nulls.index:
        if df_nulls[col] > limit:
            cols_to_delete.append(col)

    return df.drop(cols_to_delete, axis=1)


def get_corr_features(df: pd.DataFrame, threshold: float):
    """
    Find all features with absolute correlation more than threshold

    Args:
        df: Series with correlation values
        threshold: correlation (value between 0 and 1)
    Return:
        cols: list with features
    """
    cols = []

    for feature in df.index:
        if abs(df.loc[feature]) > threshold:
            cols.append(feature)
    return cols


def check_distribution(
    train: pd.DataFrame, test: pd.DataFrame, p_threshold: float = 0.05
):
    """
    Function compares the distributions of the train and test datasets
    using ttest (import stats from scipy required)

    Args:
        train: DataFrame
        test: DataFrame
    Return:
        res: DataFrame with bool values. True - ditribution is same
    """
    res = pd.Series()
    for col in train.columns:
        st, p = stats.ttest_ind(train[col], test[col])
        res[col] = p >= p_threshold

    return res


# PLOT DATA


def print_evaluate_regression(true, predicted):
    mse = metrics.mean_squared_error(true, predicted)
    rmsle = metrics.mean_squared_log_error(true, predicted, squared=False)
    r2 = metrics.r2_score(true, predicted)
    print("MSE:", mse)
    print("RMSLE:", rmsle)
    print("R2:", r2)
    print("______")


def showRegressionResults(trainHistory):
    """Function to:
     * Print final loss.
     * Plot loss curves.

    Args:
      trainHistory: object returned by model.fit
    """

    # Print final loss
    print("Final training loss: " + str(trainHistory.history["loss"][-1]))
    print("Final Validation loss: " + str(trainHistory.history["val_loss"][-1]))

    # Plot loss curves
    plt.plot(trainHistory.history["loss"])
    plt.plot(trainHistory.history["val_loss"])
    plt.legend(["Training loss", "Validation loss"], loc="best")
    plt.title("Loss Curves")


# TESTING


def test_nulls(df):
    s_null = df.isna().sum()
    nans = []
    for feature in df.isna().sum().index:
        if s_null[feature] != 0:
            nans.append(feature)
    assert len(nans) == 0, f"NaNs in columns:\n {nans}"
    print("Engineered features do not contain nulls.")


# LOGGING


def log_evaluate_regression(true, predicted):
    mse = metrics.mean_squared_error(true, predicted)
    rmsle = metrics.mean_squared_log_error(true, predicted, squared=False)
    r2 = metrics.r2_score(true, predicted)
    logging.info(f"MSE:, {mse}")
    logging.info(f"RMSLE:, {rmsle}")
    logging.info(f"R2:, {r2}")
    logging.info("______")
