"""
This module contains utility functions for the analysis notebook.
"""

import json
import pathlib
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.utils import parallel_backend

# catch warnings
warnings.filterwarnings("ignore")

# setting global seed
np.random.seed(0)

# PROJECT DIR PATH
PROJECT_DIR_PATH = pathlib.Path(__file__).parent.parent


def load_json_file(fpath: str | pathlib.Path) -> dict:
    """Wrapper function that loads in a json file

    Parameters
    ----------
    fpath : pathlib.Path
        path to json file

    Returns
    -------
    dict
        contents of the json file

    Raises
    ------
    TypeError
        Raised if pathlib.Path or str types are not passed
    FileNotFoundError
        Raised if the file path provides does not exist
    """

    # type checking
    if isinstance(fpath, str):
        fpath = pathlib.Path(fpath).resolve(strict=True)
    if not isinstance(fpath, pathlib.Path):
        raise TypeError("'fpath' must be a pathlib.Path or str object")

    # loading json file
    with open(fpath, mode="r") as contents:
        return json.load(contents)


def drop_na_samples(
    profile: pd.DataFrame, features: list[str], cut_off: Optional[float] = 0
):
    """Drops rows from a profile based on the number of NaN values allowed per
    row.

    Parameters
    ----------
    profile : pd.DataFrame
        Profile containing the samples that may have rows with NaN values that
        will be dropped.
    features: list[str]
        list of feature to count number of NaNs
    cut_off : Optional[float], optional
        The maximum proportion of NaN values allowed in a row. The cut_off
        values ranges between 0 to 1.0. If set to 0.0, will drop all rows if
        it has at least 1 NaN.  Default is 0.

    Returns
    -------
    pd.DataFrame
        Porfile with rows dropped if they exceed the specified cut-off of NaN
        values.

    Raises
    ------
    TypeError
        If 'profile' is not a pandas DataFrame object.
        If 'features' is not a list or if any element in 'features' is not a string.
        If 'cut_off' is not a float.
    ValueError
        If 'cut_off' is not between 0.0 and 1.0.
    """

    # type checking
    if not isinstance(profile, pd.DataFrame):
        raise TypeError("'profile must be a DataFrame'")
    if not isinstance(features, list):
        raise TypeError("'features' must be a list")
    if not all([isinstance(feat, str) for feat in features]):
        raise TypeError("elements within the feats must be a string type")
    if isinstance(cut_off, int):
        cut_off = float(cut_off)
    if not isinstance(cut_off, float):
        raise TypeError("'float' must be a float type")
    if isinstance(cut_off, float) and (cut_off > 1.0 or cut_off < 0):
        raise ValueError("'cut_off' must be between a float between 0 <= cut_off >= 1")

    # creating profiles
    meta_cols = list(set(profile.columns.tolist()) - set(features))
    meta_profile = profile[meta_cols]
    profile = profile[features]

    # if cut_off is None, Drop all rows that has at least 1 NaN
    if cut_off == 0.0:
        profile = profile.dropna()
    else:
        # Remove the entries based on the frequency of NaN values found per sample.
        # This is done by creating a boolean mask where True indicates that the number
        # of NaNs is less than the cutoff (max_na).
        n_samples = profile.shape[0]  # rows == number of samples
        max_na = round(n_samples * cut_off)
        bool_mask = (profile.isna().sum(axis=1) < max_na).values

        # updating profile with accepted samples
        profile = profile[bool_mask]

    # Merge the metdata with
    profile = meta_profile.merge(
        profile, left_index=True, right_index=True
    ).reset_index(drop=True)

    return profile


def shuffle_features(feature_vals: np.array, seed: Optional[int] = 0) -> np.array:
    """Shuffles all values within feature space

    Parameters
    ----------
    feature_vals : np.array
        Values to be shuffled.

    seed : Optional[int]
        setting random seed

    Returns
    -------
    np.array
        Returns shuffled values within the feature space

    Raises
    ------
    TypeError
        Raised if a numpy array is not provided
    """
    # setting seed
    np.random.seed(seed)

    # shuffle given array
    if not isinstance(feature_vals, np.ndarray):
        raise TypeError("'feature_vals' must be a numpy array")
    if feature_vals.ndim != 2:
        raise TypeError("'feature_vals' must be a 2x2 matrix")

    # shuffling feature space
    n_cols = feature_vals.shape[1]
    for col_idx in range(0, n_cols):
        # selecting column, shuffle, and update:
        feature_vals[:, col_idx] = np.random.permutation(feature_vals[:, col_idx])

    return feature_vals


def train_multiclass(
    X_train: np.ndarray, y_train: np.ndarray, param_grid: dict, seed: Optional[int] = 0
) -> BaseEstimator:
    """This approach utilizes RandomizedSearchCV to explore a range of parameters
    specified in the param_grid, ultimately identifying the most suitable model
    configuration

    This function will return the best model.

    Parameters
    ----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    param_grid : dict
        parameters to tune
    seed : Optional[int]
        set random seed, default = 0

    Returns
    -------
    BaseEstimator
        Best model
    """
    # setting seed:
    np.random.seed(seed)

    # create a Logistic regression model with One vs Rest scheme (ovr)
    logistic_regression_model = LogisticRegression(class_weight="balanced")
    ovr_model = OneVsRestClassifier(logistic_regression_model)

    # next is to use RandomizedSearchCV for hyper parameter turning
    with parallel_backend("multiprocessing"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

            # execute RandomizedResearchCV
            random_search = RandomizedSearchCV(
                estimator=ovr_model,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                random_state=seed,
                n_jobs=-1,
            )

            # fit with training data
            random_search.fit(X_train, y_train)

    # get the best model
    best_model = random_search.best_estimator_
    return best_model


def evaluate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    dataset: str,
    shuffled: bool,
    seed: Optional[int] = 0,
) -> tuple[pd.DataFrame]:
    """calculates the precision/recall and f1

    Parameters
    ----------
    model : BaseEstimator
        best model
    X : np.ndarray
        features
    y : np.ndarray
        labels
    shuffled : bool
        Flag indicating if the data has been shuffled
    seed : Optional[int], optional
        _description_, by default 0

    Returns
    -------
    tuple[pd.DataFrame]
        returns a tuple that contains the f1 scores and precision/recall scores
        in a dataframe
    """

    # setting seed
    np.random.seed(seed)

    # number of classes
    n_classes = len(np.unique(y, axis=0))

    # loading in injury_codes
    injury_code_path = (
        PROJECT_DIR_PATH / "results/1.data_splits/injury_codes.json"
    ).resolve(strict=True)
    injury_codes = load_json_file(injury_code_path)

    # making predictions
    predictions = model.predict(X)
    probability = model.predict_proba(X)

    # computing and collecting  precision and recall curve
    precision_recall_scores = []
    for i in range(n_classes):
        # precision_recall_curve calculation
        precision, recall, _ = precision_recall_curve(y[:, i], probability[:, i])

        # iterate all scores and save all data into a list
        for i in range(len(precision)):
            precision_recall_scores.append([dataset, shuffled, precision[i], recall[i]])

    # creating scores df
    precision_recall_scores = pd.DataFrame(
        precision_recall_scores, columns=["dataset", "shuffled", "precision", "recall"]
    )

    # Compute F1 score
    f1_scores = []
    for i in range(n_classes):
        y_true = y[:, i]
        y_pred = predictions[:, i]
        f1 = f1_score(y_true, y_pred)
        f1_scores.append([dataset, shuffled, injury_codes["decoder"][str(i)], f1])

    # convert to data frame and display
    f1_scores = pd.DataFrame(
        f1_scores, columns=["data_set", "shuffled", "class", "f1_score"]
    )

    return (precision_recall_scores, f1_scores)


def check_feature_order(ref_feat_order: list[str], input_feat_order: list[str]) -> bool:
    """Check if the input feature order follows the same sequence as the reference feature order.

    Parameters
    ----------
    ref_feat_order : list[str]
        The reference feature space defining the expected order of features.
    input_feat_order : list[str]
        The feature space to be checked against the reference feature order.

    Returns
    -------
    bool
        True if the inputted feature space has the same order as the reference feature
        space, else False.

    Raises
    ------
    ValueError
        When the feature spaces are not of the same size.
    """

    # Check if both feature spaces have the same size
    if len(ref_feat_order) != len(input_feat_order):
        raise ValueError("Feature spaces are not of the same size")

    # Check if the order is the same
    for idx in range(len(ref_feat_order)):
        # If there's a mismatch, return False
        if ref_feat_order[idx] != input_feat_order[idx]:
            return False

    # Return True if the order is identical
    return True
