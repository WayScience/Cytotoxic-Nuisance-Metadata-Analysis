#!/usr/bin/env python
# coding: utf-8

# # Module 2: Modeling
#
# In this notebook, we focus on developing and training a Multi-class logistic regression model, employing Randomized Cross-Validation (CV) for hyperparameter tuning to address our classification task. The dataset is split into an 80% training set and a 20% testing set. To evaluate the performance of our model during training, we used performance evaluation metrics such as precision, recall, and F1 scores. Additionally, we extend our evaluation by testing our model on a holdout dataset, which includes plate, treatment, and well information, providing a comprehensive assessment of its real-world performance.

# In[1]:


import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

# import local modules
sys.path.append("../../")
from src.utils import (
    check_feature_order,
    evaluate_model_performance,
    generate_confusion_matrix_tl,
    load_json_file,
    shuffle_features,
    split_meta_and_features,
    train_multiclass,
)

# In[2]:


# setting random seeds varaibles
seed = 0
np.random.seed(seed)

# setting paths and parameters
results_dir = pathlib.Path("../../results").resolve(strict=True)
feature_dir = (results_dir / "0.feature_selection/").resolve(strict=True)
data_splits_dir = (results_dir / "1.data_splits").resolve(strict=True)

# test and train data paths
X_train_path = (data_splits_dir / "X_train.csv.gz").resolve(strict=True)
X_test_path = (data_splits_dir / "X_test.csv.gz").resolve(strict=True)
y_train_path = (data_splits_dir / "y_train.csv.gz").resolve(strict=True)
y_test_path = (data_splits_dir / "y_test.csv.gz").resolve(strict=True)

# shared feature space path
feature_space_path = (feature_dir / "cell_injury_shared_feature_space.json").resolve(
    strict=True
)

# holdout paths
plate_holdout_path = (data_splits_dir / "plate_holdout.csv.gz").resolve(strict=True)
treatment_holdout_path = (data_splits_dir / "treatment_holdout.csv.gz").resolve(
    strict=True
)
wells_holdout_path = (data_splits_dir / "wells_holdout.csv.gz").resolve(strict=True)

# setting output paths
modeling_dir = (results_dir / "2.modeling").resolve()
modeling_dir.mkdir(exist_ok=True)


# Below are the paramters used:
#
# - **penalty**: Specifies the type of penalty (regularization) applied during logistic regression. It can be 'l1' for L1 regularization, 'l2' for L2 regularization, or 'elasticnet' for a combination of both.
# - **C**: Inverse of regularization strength; smaller values specify stronger regularization. Controls the trade-off between fitting the training data and preventing overfitting.
# - **max_iter**: Maximum number of iterations for the optimization algorithm to converge.
# - **tol**: Tolerance for the stopping criterion during optimization. It represents the minimum change in coefficients between iterations that indicates convergence.
# - **l1_ratio**: The mixing parameter for elastic net regularization. It determines the balance between L1 and L2 penalties in the regularization term. A value of 1 corresponds to pure L1 (Lasso) penalty, while a value of 0 corresponds to pure L2 (Ridge) penalty
# - **solver**: Optimization algorithms to be explored during hyperparameter tuning for logistic regression
#

# In[3]:


# Parameters
param_grid = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "max_iter": np.arange(100, 1100, 100),
    "tol": np.arange(1e-6, 1e-3, 1e-6),
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}


# Loading training data splits
#

# In[4]:


# loading injury codes
injury_codes = load_json_file(feature_dir / "injury_codes.json")

# load share feature space data
feature_space = load_json_file(feature_space_path)
shared_features = feature_space["features"]

# loading training data splits
X_train = pd.read_csv(X_train_path)
X_test = pd.read_csv(X_test_path)
y_train = pd.read_csv(y_train_path)
y_test = pd.read_csv(y_test_path)


# spliting meta and feature column names
_, feat_cols = split_meta_and_features(X_train)

# checking if the feature space are identical (also looks for feature space order)
assert check_feature_order(
    ref_feat_order=shared_features, input_feat_order=X_test.columns.tolist()
), "Feature space are not identical"

# display data split sizes
print("X training size", X_train.shape)
print("X testing size", X_test.shape)
print("y training size", y_train.shape)
print("y testing size", y_test.shape)  #


# ## Training and Evaluating Multi-class Logistic Model with original dataset split
#

# In[5]:


# setting model path
model_path = modeling_dir / "multi_class_model.joblib"

# if trained model exists, skip training
if model_path.exists():
    best_model = joblib.load(model_path)

# train model and save
else:
    best_model = train_multiclass(X_train, y_train, param_grid=param_grid, seed=seed)
    joblib.dump(best_model, model_path)


# In[6]:


# evaluating model on train dataset
train_precision_recall_df, train_f1_score_df = evaluate_model_performance(
    model=best_model, X=X_train, y=y_train, shuffled=False, dataset_type="Train"
)

# evaluating model on test dataset
test_precision_recall_df, test_f1_score_df = evaluate_model_performance(
    model=best_model, X=X_test, y=y_test, shuffled=False, dataset_type="Test"
)


# In[7]:


# creating confusion matrix for both train and test set on non-shuffled model
cm_train_df = generate_confusion_matrix_tl(
    model=best_model, X=X_train, y=y_train, shuffled=False, dataset_type="Train"
)
cm_test_df = generate_confusion_matrix_tl(
    model=best_model, X=X_test, y=y_test, shuffled=False, dataset_type="Test"
)


# ## Training and Evaluating Multi-class Logistic Model with shuffled dataset split
#

# In[8]:


# shuffle feature space
shuffled_X_train = shuffle_features(X_train, features=shared_features, seed=seed)

# checking if the shuffled and original feature space are the same
assert not X_train.equals(shuffled_X_train), "DataFrames are the same!"


# In[9]:


# setting model path
shuffled_model_path = modeling_dir / "shuffled_multi_class_model.joblib"

# if trained model exists, skip training
if shuffled_model_path.exists():
    shuffled_best_model = joblib.load(shuffled_model_path)

# train model and save
else:
    shuffled_best_model = train_multiclass(
        shuffled_X_train, y_train, param_grid=param_grid, seed=seed
    )
    joblib.dump(shuffled_best_model, shuffled_model_path)


# In[10]:


# evaluating shuffled model on train dataset
shuffle_train_precision_recall_df, shuffle_train_f1_score_df = (
    evaluate_model_performance(
        model=shuffled_best_model,
        X=shuffled_X_train,
        y=y_train,
        shuffled=True,
        dataset_type="Train",
    )
)

# valuating shuffled model on test dataset
shuffle_test_precision_recall_df, shuffle_test_f1_score_df = evaluate_model_performance(
    model=shuffled_best_model, X=X_test, y=y_test, shuffled=True, dataset_type="Test"
)


# In[11]:


# creating confusion matrix for shuffled model
shuffled_cm_train_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=shuffled_X_train,
    y=y_train,
    shuffled=True,
    dataset_type="Train",
)
shuffled_cm_test_df = generate_confusion_matrix_tl(
    model=shuffled_best_model, X=X_test, y=y_test, shuffled=True, dataset_type="Test"
)


# ## Evaluating Multi-class model with holdout data
#

# Loading in all the hold out data
#

# In[12]:


# loading all holdouts
plate_holdout_df = pd.read_csv(plate_holdout_path)
treatment_holdout_df = pd.read_csv(treatment_holdout_path)
well_holdout_df = pd.read_csv(wells_holdout_path)

# splitting the dataset into X = features , y = injury_types
X_plate_holdout = plate_holdout_df[feat_cols]
y_plate_holdout = plate_holdout_df["injury_code"]

X_treatment_holdout = treatment_holdout_df[feat_cols]
y_treatment_holdout = treatment_holdout_df["injury_code"]

X_well_holdout = well_holdout_df[feat_cols]
y_well_holdout = well_holdout_df["injury_code"]


# ### Evaluating Multi-class model trained with original split with holdout data
#

# In[13]:


# evaluating plate holdout data with both trained original and shuffled model
plate_ho_precision_recall_df, plate_ho_f1_score_df = evaluate_model_performance(
    model=best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=False,
    dataset_type="Plate Holdout",
)

plate_ho_shuffle_precision_recall_df, plate_ho_shuffle_f1_score_df = (
    evaluate_model_performance(
        model=shuffled_best_model,
        X=X_plate_holdout,
        y=y_plate_holdout,
        shuffled=True,
        dataset_type="Plate Holdout",
    )
)


# evaluating treatment holdout data with both trained original and shuffled model
treatment_ho_precision_recall_df, treatment_ho_f1_score_df = evaluate_model_performance(
    model=best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=False,
    dataset_type="Treatment Holdout",
)

treatment_ho_shuffle_precision_recall_df, treatment_ho_shuffle_f1_score_df = (
    evaluate_model_performance(
        model=shuffled_best_model,
        X=X_treatment_holdout,
        y=y_treatment_holdout,
        shuffled=True,
        dataset_type="Treatment Holdout",
    )
)


# evaluating well holdout data with both trained original and shuffled model
well_ho_precision_recall_df, well_ho_f1_score_df = evaluate_model_performance(
    model=best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=False,
    dataset_type="Well Holdout",
)

well_ho_shuffle_precision_recall_df, well_ho_shuffle_f1_score_df = (
    evaluate_model_performance(
        model=shuffled_best_model,
        X=X_well_holdout,
        y=y_well_holdout,
        shuffled=True,
        dataset_type="Well Holdout",
    )
)


# In[14]:


# creating confusion matrix with plate holdout (shuffled and not shuffled)
plate_ho_cm_df = generate_confusion_matrix_tl(
    model=best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=False,
    dataset_type="Plate Holdout",
)
shuffled_plate_ho_cm_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=True,
    dataset_type="Plate Holdout",
)

# creating confusion matrix with treatment holdout (shuffled and not shuffled)
treatment_ho_cm_df = generate_confusion_matrix_tl(
    model=best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=False,
    dataset_type="Treatment Holdout",
)
shuffled_treatment_ho_cm_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=True,
    dataset_type="Treatment Holdout",
)

# creating confusion matrix with plate_hold (shuffled and not shuffled)
well_ho_cm_df = generate_confusion_matrix_tl(
    model=best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=False,
    dataset_type="Well Holdout",
)
shuffled_well_ho_cm_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=True,
    dataset_type="Well Holdout",
)


# Storing all f1 and pr scores
#

# In[15]:


# storing all f1 scores
all_f1_scores = pd.concat(
    [
        # original split
        test_f1_score_df,
        train_f1_score_df,
        # shuffle split
        shuffle_test_f1_score_df,
        shuffle_train_f1_score_df,
        # plate holdout
        plate_ho_f1_score_df,
        plate_ho_shuffle_f1_score_df,
        # treatment holdout
        treatment_ho_f1_score_df,
        treatment_ho_shuffle_f1_score_df,
        # well holdout
        well_ho_f1_score_df,
        well_ho_shuffle_f1_score_df,
    ]
)

# saving all f1 scores
all_f1_scores.to_csv(
    modeling_dir / "all_f1_scores.csv.gz", index=False, compression="gzip"
)


# In[16]:


# storing pr scores
all_pr_scores = pd.concat(
    [
        # original split
        test_precision_recall_df,
        train_precision_recall_df,
        # shuffled split
        shuffle_test_precision_recall_df,
        shuffle_train_precision_recall_df,
        # plate holdout
        plate_ho_precision_recall_df,
        plate_ho_shuffle_precision_recall_df,
        # treatment holdout
        treatment_ho_precision_recall_df,
        treatment_ho_shuffle_precision_recall_df,
        # well holdout
        well_ho_precision_recall_df,
        well_ho_shuffle_precision_recall_df,
    ]
)

# saving pr scores
all_pr_scores.to_csv(
    modeling_dir / "precision_recall_scores.csv.gz", index=False, compression="gzip"
)


# In[17]:


all_cm_dfs = pd.concat(
    [
        # original split
        cm_train_df,
        cm_test_df,
        # shuffled split
        shuffled_cm_train_df,
        shuffled_cm_test_df,
        # plate holdout
        plate_ho_cm_df,
        shuffled_plate_ho_cm_df,
        # treatment holdout
        treatment_ho_cm_df,
        shuffled_treatment_ho_cm_df,
        # well holdout
        well_ho_cm_df,
        shuffled_well_ho_cm_df,
    ]
)


# saving pr scores
all_cm_dfs.to_csv(
    modeling_dir / "confusion_matrix.csv.gz", index=False, compression="gzip"
)
