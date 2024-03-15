#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import pathlib
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.utils import parallel_backend

# catch warnings
warnings.filterwarnings("ignore")

#
sys.path.append("../../")
from src.utils import shuffle_features  # noqa

# ## Helper functions

# In[2]:


def train_multiclass(
    X_train: np.ndarray, y_train: np.ndarray, param_grid: dict, seed: Optional[int] = 0
) -> BaseEstimator:
    """Develops a Logistic Regression model employing the One vs Rest training
    scheme and incorporates RandomizedSearchCV for optimal parameter selection.
    This approach utilizes RandomizedSearchCV to explore a range of parameters
    specified in the param_grid, ultimately identifying the most suitable model
    configuration

    This function will return best model.

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


# In[3]:


# setting paths and parameters
results_dir = pathlib.Path("../../results").resolve(strict=True)
data_splits_dir = (results_dir / "1.data_splits").resolve(strict=True)

# setting path for training dataset
training_dataset_path = (data_splits_dir / "training_data.csv.gz").resolve(strict=True)

# holdout paths
plate_holdout_path = (data_splits_dir / "plate_holdout.csv.gz").resolve(strict=True)
treatment_holdout_path = (data_splits_dir / "treatment_holdout.csv.gz").resolve(
    strict=True
)
wells_holdout_path = (data_splits_dir / "wells_holdout.csv.gz").resolve(strict=True)

# setting output paths
modeling_dir = (results_dir / "2.modeling").resolve()
modeling_dir.mkdir(exist_ok=True)

# ml parameters to hyperparameterization tuning
param_grid = {
    "estimator__C": uniform(0.1, 10),
    "estimator__solver": ["newton-cg", "liblinear", "sag", "saga"],
    "estimator__penalty": ["l1", "l2", "elasticnet"],
    "estimator__l1_ratio": uniform(0, 1),
}


# In[4]:


# loading injury codes
with open(data_splits_dir / "injury_codes.json") as json_file:
    injury_codes = json.load(json_file)

# loading in the dataset
training_df = pd.read_csv(training_dataset_path)

# display data
print("Shape: ", training_df.shape)
training_df.head()


# In[5]:


# setting random seeds
seed = 0
np.random.seed(seed)

# setting a one hot encoder


# splitting between meta and feature columns
meta_cols = training_df.columns[:33]
feat_cols = training_df.columns[33:]

# Splitting the data where y = injury_types and X = morphology features
X = training_df[feat_cols].values
y_labels = training_df["injury_code"]

# since this is a multi-class problem and in order for precision and recalls to work
# we need to binarize it to different classes
# source: https://stackoverflow.com/questions/56090541/how-to-plot-precision-and-recall-of-multiclass-classifier
n_classes = len(np.unique(y_labels.values))
y = label_binarize(y_labels, classes=[*range(n_classes)])

# then we can split the data set with are newly binarized labels
# we made sure to use stratify to ensure proportionality within training and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, stratify=y)


# In[6]:


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


# ## Training and Evaluating Multi-class Logistic Model with original dataset split
#

# In[7]:


# train and get the best_model
best_model = train_multiclass(X_train, y_train, param_grid=param_grid, seed=0)


# In[ ]:


test_precision_recall_df, test_f1_score_df = evaluate(
    model=best_model, X=X_test, y=y_test, dataset="test", shuffled=False, seed=0
)
train_precision_recall_df, train_f1_score_df = evaluate(
    model=best_model, X=X_train, y=y_train, dataset="train", shuffled=False, seed=0
)


# ## Training and Evaluating Multi-class Logistic Model with shuffled dataset split
#

# In[ ]:


# setting random seed
seed = 0
np.random.seed(seed)

# shuffle feature space
shuffled_X_train = shuffle_features(X_train, seed=seed)


# In[ ]:


shuffled_best_model = train_multiclass(
    shuffled_X_train, y_train, param_grid=param_grid, seed=seed
)


# In[ ]:


shuffle_test_precision_recall_df, shuffle_test_f1_score_df = evaluate(
    model=best_model, X=X_test, y=y_test, dataset="test", shuffled=True, seed=0
)
shuffle_train_precision_recall_df, shuffle_train_f1_score_df = evaluate(
    model=best_model, X=X_train, y=y_train, dataset="train", shuffled=True, seed=0
)


# ## Evaluating Multi-class model with holdout data

# In[ ]:


# loading in holdout data
# setting seed
seed = 0
np.random.seed(seed)

# loading all holdouts
plate_holdout_df = pd.read_csv(plate_holdout_path)
treatment_holdout_df = pd.read_csv(treatment_holdout_path)
well_holdout_df = pd.read_csv(wells_holdout_path)

# splitting the dataset into
X_plate_holdout = plate_holdout_df[feat_cols]
y_plate_holout = label_binarize(
    y=plate_holdout_df["injury_code"],
    classes=[*range(len(plate_holdout_df["injury_code"].unique()))],
)

X_treatment_holdout = treatment_holdout_df[feat_cols]
y_treatment_holout = label_binarize(
    y=treatment_holdout_df["injury_code"],
    classes=[*range(len(treatment_holdout_df["injury_code"].unique()))],
)

X_well_holdout = well_holdout_df[feat_cols]
y_well_holout = label_binarize(
    y=well_holdout_df["injury_code"],
    classes=[*range(len(well_holdout_df["injury_code"].unique()))],
)


# ### Evaluating Multi-class model trained with original split with holdout data

# In[ ]:


# evaluating with plate holdout
plate_ho_precision_recall_df, plate_ho_f1_score_df = evaluate(
    model=best_model,
    X=X_plate_holdout,
    y=y_plate_holout,
    dataset="plate_holdout",
    shuffled=False,
    seed=0,
)
plate_ho_shuffle_precision_recall_df, plate_ho_shuffle_train_f1_score_df = evaluate(
    model=shuffled_best_model,
    X=X_plate_holdout,
    y=y_plate_holout,
    dataset="plate_holdout",
    shuffled=True,
    seed=0,
)

# evaluating with treatment holdout
treatment_ho_precision_recall_df, treatment_ho_f1_score_df = evaluate(
    model=best_model,
    X=X_treatment_holdout,
    y=y_treatment_holout,
    dataset="treatment_holdout",
    shuffled=False,
    seed=0,
)
treatment_ho_shuffle_precision_recall_df, treatment_ho_shuffle_train_f1_score_df = (
    evaluate(
        model=shuffled_best_model,
        X=X_treatment_holdout,
        y=y_treatment_holout,
        dataset="treatment_holdout",
        shuffled=True,
        seed=0,
    )
)

# evaluating with treatment holdout
well_ho_precision_recall_df, well_ho_test_f1_score_df = evaluate(
    model=best_model,
    X=X_well_holdout,
    y=y_well_holout,
    dataset="well_holdout",
    shuffled=False,
    seed=0,
)
well_ho_shuffle_precision_recall_df, well_ho_shuffle_train_f1_score_df = evaluate(
    model=shuffled_best_model,
    X=X_well_holdout,
    y=y_well_holout,
    dataset="well_holdout",
    shuffled=True,
    seed=0,
)


# In[ ]:


# storing all f1 scores
all_f1_scores = pd.concat(
    [
        test_f1_score_df,
        train_f1_score_df,
        shuffle_test_f1_score_df,
        shuffle_train_f1_score_df,
        plate_ho_f1_score_df,
        plate_ho_shuffle_train_f1_score_df,
        treatment_ho_f1_score_df,
        treatment_ho_shuffle_train_f1_score_df,
        well_ho_test_f1_score_df,
        well_ho_shuffle_train_f1_score_df,
    ]
)
all_f1_scores.to_csv(
    modeling_dir / "all_f1_scores.csv.gz", index=False, compression="gzip"
)

# storing pr scores
all_pr_scores = pd.concat(
    [
        shuffle_test_precision_recall_df,
        shuffle_train_precision_recall_df,
        shuffle_test_precision_recall_df,
        shuffle_train_precision_recall_df,
        plate_ho_precision_recall_df,
        plate_ho_shuffle_precision_recall_df,
        treatment_ho_precision_recall_df,
        treatment_ho_shuffle_precision_recall_df,
        well_ho_precision_recall_df,
        well_ho_shuffle_precision_recall_df,
    ]
)
all_pr_scores.to_csv(
    modeling_dir / "precision_recall_scores.csv.gz", index=False, compression="gzip"
)
