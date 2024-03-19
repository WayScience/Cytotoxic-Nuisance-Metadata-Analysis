"""
This module contains utility functions for the analysis notebook.
"""

from typing import Optional

import numpy as np
import pandas as pd


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

    # creating a copy for feature values to prevent overwriting of global variables
    feature_vals = np.copy(feature_vals)

    # shuffling feature space
    n_cols = feature_vals.shape[1]
    for col_idx in range(0, n_cols):
        # selecting column, shuffle, and update:
        feature_vals[:, col_idx] = np.random.permutation(feature_vals[:, col_idx])

    return feature_vals
