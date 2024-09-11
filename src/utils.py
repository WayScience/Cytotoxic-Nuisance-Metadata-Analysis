"""
This module contains utility functions for the analysis notebook.
"""

import json
import pathlib
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pycytominer.cyto_utils import infer_cp_features
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import label_binarize
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


def shuffle_features(
    profile: pd.DataFrame, features: list[str], seed: Optional[int] = 0
) -> pd.DataFrame:
    """Shuffles the feature space of the given profile. The shuffling process is done by independently
    taking each feature, shuffling the values within the feature space, and updating the provided profile.
    As a result, a DataFrame with a shuffled feature space is returned.

    Parameters
    ----------
    profile : pd.DataFrame
        DataFrame to be shuffled.

    features : list[str]
        List of features to select for shuffling

    seed : Optional[int]
        setting random seed

    Returns
    -------
    pd.DataFrame
        Returns shuffled values within the selected feature space

    Raises
    ------
    TypeError
        Raised if a pandas DataFrame is not provded
        Raised if a non-int value is provided for 'seed'
    """

    # shuffle given array
    if not isinstance(profile, pd.DataFrame):
        raise TypeError("'feature_vals' must be a pandas DataFrame")
    if not isinstance(seed, int):
        raise TypeError("'seed' must be an int")

    # make a copy of the DataFrame to prevent over writing the original DataFrame
    profile = profile.copy(deep=True)[features]

    # how shuffle everything per row
    for col in profile.columns:
        profile[col] = (
            profile[col].sample(frac=1, random_state=seed).reset_index(drop=True)
        )

    return profile


def train_multiclass(
    X_train: np.ndarray,
    y_train: np.ndarray,
    param_grid: dict,
    cv_results_outpath: pathlib.Path,
    seed: Optional[int] = 0,
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
    cv_results_outpath: pathlib.Path
        path were to save the model cross validation parameter
        search scores
    seed : Optional[int]
        set random seed, default = 0

    Returns
    -------
    BaseEstimator
        Best model
    """
    # setting seed:
    np.random.seed(seed)

    # create a Logistic regression model with multi_
    lr_model = LogisticRegression(multi_class="multinomial", class_weight="balanced")

    # next is to use RandomizedSearchCV for hyper parameter turning
    with parallel_backend("multiprocessing"):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, module="sklearn"
            )
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
            # execute RandomizedResearchCV
            random_search = RandomizedSearchCV(
                lr_model,
                param_distributions=param_grid,
                n_iter=10,
                cv=5,
                verbose=0,
                random_state=0,
                n_jobs=-1,
            )

            # fit with training data
            random_search.fit(X_train, y_train)

    # save the cv results search results
    cv_results_df = pd.DataFrame(random_search.cv_results_)
    cv_results_df.to_csv(cv_results_outpath, index=False)

    # get the best model
    best_model = random_search.best_estimator_

    return best_model


def calculate_multi_class_pr_curve(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    dataset_type: str,
    shuffled: bool,
) -> pd.DataFrame:
    """Calculates precision and recall from multi-class model.

    This code was heavily influenced by @roshankern:
    https://github.com/WayScience/phenotypic_profiling/blob/main/utils/evaluate_utils.py

    Parameters
    ----------
    model : BaseEstimator
        mutli-class model to evaluate
    X : np.ndarray
        Feature dataset
    y : np.ndarray
        associated labels
    dataset_type : str
        label indicating what type of data you are evaluating on the model
    shuffled : bool
        label indicating wether the data is shuffled or not
    """

    # get injury_type classes
    injury_types = model.classes_

    # binarize labels
    y_binarized = label_binarize(y, classes=injury_types)

    # predict the probability scores
    y_scores = model.predict_proba(X)

    # loading in injury_codes
    injury_code_path = (
        PROJECT_DIR_PATH / "results/0.feature_selection/injury_codes.json"
    ).resolve(strict=True)
    injury_codes = load_json_file(injury_code_path)["decoder"]

    # storing all dataframes containing pr scores per class
    pr_data = []

    for injury_type in injury_types:
        # get pr scores
        precision, recall, _ = precision_recall_curve(
            y_binarized[:, injury_type], y_scores[:, injury_type]
        )

        # store data
        pr_data.append(
            pd.DataFrame(
                {
                    "dataset_type": dataset_type,
                    "shuffled": shuffled,
                    "injury_type": injury_codes[str(injury_type)],
                    "precision": precision,
                    "recall": recall,
                }
            )
        )

    # return the scores in tidy long format
    return pd.concat(pr_data, axis=0).reset_index(drop=True)


def calculate_multi_class_f1score(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    dataset_type: str,
    shuffled: bool,
) -> pd.DataFrame:
    """Calculate weighted and individual class f1 scores.

    This code was heavily influenced by @roshankern:
    https://github.com/WayScience/phenotypic_profiling/blob/main/utils/evaluate_utils.py

    Parameters
    ----------
    model : BaseEstimator
        mutli-class model to evaluate
    X : np.ndarray
        Feature dataset
    y : np.ndarray
        associated labels
    dataset_type : str
        label indicating what type of data you are evaluating on the model
    shuffled : bool
        label indicating wether the data is shuffled or not
    """

    # loading injury codes
    injury_code_path = (
        PROJECT_DIR_PATH / "results/0.feature_selection/injury_codes.json"
    ).resolve(strict=True)
    injury_codes = load_json_file(injury_code_path)["decoder"]

    # injury classes
    injury_labels = model.classes_
    injury_types = [injury_codes[str(injury_code)] for injury_code in injury_labels]

    # prediction
    y_pred = model.predict(X)

    # calculate f1 score per injust and weighted scores across all injuries
    scores = f1_score(y, y_pred, average=None, labels=injury_labels, zero_division=0)

    # add scores into a data frame
    scores = pd.DataFrame(scores).transpose()
    scores.columns = injury_types
    scores = scores.transpose().reset_index()
    scores.columns = ["injury_type", "f1_score"]

    # inserting data info columns
    scores.insert(0, "dataset_type", dataset_type)
    scores.insert(1, "shuffled", shuffled)

    return scores


def generate_confusion_matrix_tl(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    shuffled: bool,
    dataset_type: str,
) -> pd.DataFrame:
    """Generates a multi-class confusion matrix with given predicted and true labels

    This also provides additional metadata: recall score, dataset type, and shuffled
    label.

    Format of this confusion matrix is tidy long.

    Parameters
    ----------
    mode : BaseEstimator
        model to measure performance
    X : np.ndarray
        features
    y : np.ndarray
        true labels
    shuffled : str
        label in dicating weather the data split shuffled or not
    dataset_type : str
        label indicating

    Returns
    -------
    pd.DataFrame
        confusion matrix with recall score
    """

    # extracting all classes the models has learned from
    class_labels = model.classes_

    # loading in injury_codes
    injury_code_path = (
        PROJECT_DIR_PATH / "results/0.feature_selection/injury_codes.json"
    ).resolve(strict=True)
    injury_codes = load_json_file(injury_code_path)["decoder"]

    # predicting labels with given X values
    predictions = model.predict(X)

    # generate confusing matrix and convert to pandas dataframe
    cm_df = pd.DataFrame(
        data=confusion_matrix(y_true=y, y_pred=predictions, labels=class_labels)
    )

    # calculate recall score
    recall_per_all_classes = []
    for i in range(len(class_labels)):
        for j in range(len(class_labels)):
            count = cm_df.iloc[j, i]
            total_count = sum(cm_df.iloc[:, i])
            recall = count / total_count

            # some treatments were not found in the holdout sets
            if np.isnan(recall):
                recall = 0.0

            recall_per_all_classes.append(recall)

    # update the data frame by replace injury codes with injury name
    cm_df.columns = [
        injury_codes[str(injury_code)] for injury_code in cm_df.columns.tolist()
    ]
    cm_df.index = [
        injury_codes[str(injury_code)] for injury_code in cm_df.index.tolist()
    ]

    cm_df = cm_df.reset_index()
    cm_df = cm_df.rename(columns={"index": "true_labels"})

    # insert data type name in the matrix
    cm_df.insert(0, "dataset_type", dataset_type)

    # insert shuffled label
    cm_df.insert(1, "shuffled_model", shuffled)

    # make the confusion matrix tidy long format
    cm_df = pd.melt(
        cm_df,
        id_vars=["dataset_type", "shuffled_model", "true_labels"],
        var_name="predicted_labels",
        value_name="count",
    )

    # insert recall data
    cm_df["recall"] = recall_per_all_classes

    return cm_df


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


def split_meta_and_features(
    profile: pd.DataFrame,
    compartments=["Nuclei", "Cells", "Cytoplasm"],
    metadata_tag: Optional[bool] = False,
) -> Tuple[list[str], list[str]]:
    """Splits metadata and feature column names

    Parameters
    ----------
    profile : pd.DataFrame
        image-based profile
    compartments : list, optional
        compartments used to generated image-based profiles, by default
        ["Nuclei", "Cells", "Cytoplasm"]
    metadata_tag : Optional[bool], optional
        indicating if the profiles have metadata columns tagged with 'Metadata_'
        , by default False

    Returns
    -------
    tuple[list[str], list[str]]
        Tuple containing metadata and feature column names
    """

    # identify features names
    features_cols = infer_cp_features(profile, compartments=compartments)

    # iteratively search metadata features and retain order if the Metadata tag is not added
    if metadata_tag is False:
        meta_cols = [
            colname
            for colname in profile.columns.tolist()
            if colname not in features_cols
        ]
    else:
        meta_cols = infer_cp_features(profile, metadata=metadata_tag)

    return (meta_cols, features_cols)


def evaluate_model_performance(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    shuffled: bool,
    dataset_type: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Wrapper function that evaluates the performance of the model by
    calculating precision, recall, and F1 scores per class.

    Parameters
    ----------
    model : BaseEstimator
        The model to evaluate.
    X : np.ndarray
        The feature array.
    y : np.ndarray
        The labels array.
    shuffled : bool
        Indicates whether the model is shuffled or not.
    dataset_type : str
        Flag indicating the data split being used.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        A tuple of DataFrames containing the precision-recall and F1 scores
        respectively.
    """

    return (
        calculate_multi_class_pr_curve(
            model=model, X=X, y=y, shuffled=shuffled, dataset_type=dataset_type
        ),
        calculate_multi_class_f1score(
            model=model, X=X, y=y, shuffled=shuffled, dataset_type=dataset_type
        ),
    )


# this needs to be a function
def get_injury_treatment_info(profile: pd.DataFrame, groupby_key: str):
    # checking
    if not isinstance(profile, pd.DataFrame):
        raise TypeError("'profile' must be a pandas data frame object")
    if not isinstance(groupby_key, str):
        raise TypeError("'groupby_key' ust be a string")
    if groupby_key not in profile.columns.tolist():
        raise ValueError("'grouby_key' column does not in data frame column")

    # Showing the amount of data we have after removing the holdout data
    meta_injury = []
    for injury_type, df in profile.groupby("injury_type"):
        # extract n_wells, n_compounds and unique compounds per injury_type
        n_wells = df.shape[0]
        injury_code = df["injury_code"].unique()[0]
        unique_compounds = list(df["Compound Name"].unique())
        n_compounds = len(unique_compounds)

        # store information
        meta_injury.append(
            [injury_type, injury_code, n_wells, n_compounds, unique_compounds]
        )

    # creating data frame
    injury_meta_df = pd.DataFrame(
        meta_injury,
        columns=[
            "injury_type",
            "injury_code",
            "n_wells",
            "n_compounds",
            "compound_list",
        ],
    ).sort_values("n_wells", ascending=False)

    # display
    return injury_meta_df
