#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import local modules
sys.path.append("../../")
from src.utils import (
    calculate_multi_class_f1score,
    calculate_multi_class_pr_curve,
    generate_confusion_matrix_tl,
    load_json_file,
    shuffle_features,
    train_multiclass,
)

# In[2]:


# setting random seeds varaibles
seed = 0
np.random.seed(seed)

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


# Below are the paramters used:
#
# - **penalty**: Specifies the type of penalty (regularization) applied during logistic regression. It can be 'l1' for L1 regularization, 'l2' for L2 regularization, or 'elasticnet' for a combination of both.
# - **C**: Inverse of regularization strength; smaller values specify stronger regularization. Controls the trade-off between fitting the training data and preventing overfitting.
# - **max_iter**: Maximum number of iterations for the optimization algorithm to converge.
# - **tol**: Tolerance for the stopping criterion during optimization. It represents the minimum change in coefficients between iterations that indicates convergence.
# - **l1_ratio**: The mixing parameter for elastic net regularization. It determines the balance between L1 and L2 penalties in the regularization term. A value of 1 corresponds to pure L1 (Lasso) penalty, while a value of 0 corresponds to pure L2 (Ridge) penalty
# - **solver**: Optimization algorithms to be explored during hyperparameter tuning for logistic regression

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


# In[4]:


# loading injury codes
injury_codes = load_json_file(data_splits_dir / "injury_codes.json")

# loading in the dataset
training_df = pd.read_csv(training_dataset_path)

# display data
print("Shape: ", training_df.shape)
training_df.head()


# Splitting the dataset into training and testing subsets involves getting 80% of the data to the training set and 20% to the test set.

# In[5]:


# splitting between meta and feature columns
meta_cols = training_df.columns[:33]
feat_cols = training_df.columns[33:]

# Splitting the data where y = injury_types and X = morphology features
X = training_df[feat_cols].values
y = training_df["injury_code"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, random_state=seed, stratify=y
)


# ## Training and Evaluating Multi-class Logistic Model with original dataset split
#

# In[6]:


# train and get the best_model
best_model = train_multiclass(X_train, y_train, param_grid=param_grid, seed=seed)

# save model
joblib.dump(best_model, modeling_dir / "multi_class_model.joblib")


# In[7]:


# evaluating model on train dataset
train_precision_recall_df = calculate_multi_class_pr_curve(
    model=best_model, X=X_train, y=y_train, shuffled=False, dataset_type="train"
)
train_f1_score_df = calculate_multi_class_f1score(
    model=best_model, X=X_train, y=y_train, shuffled=False, dataset_type="train"
)

# evaluating model on test dataset
test_precision_recall_df = calculate_multi_class_pr_curve(
    model=best_model, X=X_test, y=y_test, shuffled=False, dataset_type="test"
)
test_f1_score_df = calculate_multi_class_f1score(
    model=best_model, X=X_test, y=y_test, shuffled=False, dataset_type="test"
)


# In[8]:


# creating confusing matrix for both train and test set on non-shuffled model
cm_train_df = generate_confusion_matrix_tl(
    model=best_model, X=X_train, y=y_train, shuffled=False, dataset_type="train"
)
cm_test_df = generate_confusion_matrix_tl(
    model=best_model, X=X_test, y=y_test, shuffled=False, dataset_type="test"
)


# ## Training and Evaluating Multi-class Logistic Model with shuffled dataset split
#

# In[9]:


# shuffle feature space
shuffled_X_train = shuffle_features(X_train, seed=seed)


# In[10]:


shuffled_best_model = train_multiclass(
    shuffled_X_train, y_train, param_grid=param_grid, seed=seed
)
joblib.dump(shuffled_best_model, modeling_dir / "shuffled_multi_class_model.joblib")


# In[11]:


# evaluating shuffled model on train dataset
shuffle_train_precision_recall_df = calculate_multi_class_pr_curve(
    model=shuffled_best_model,
    X=shuffled_X_train,
    y=y_train,
    shuffled=True,
    dataset_type="train",
)
shuffle_train_f1_score_df = calculate_multi_class_f1score(
    model=shuffled_best_model,
    X=shuffled_X_train,
    y=y_train,
    shuffled=True,
    dataset_type="train",
)

# evaluating shuffled model on test dataset
shuffle_test_precision_recall_df = calculate_multi_class_pr_curve(
    model=shuffled_best_model, X=X_test, y=y_test, shuffled=True, dataset_type="test"
)
shuffle_test_f1_score_df = calculate_multi_class_f1score(
    model=shuffled_best_model, X=X_test, y=y_test, shuffled=True, dataset_type="test"
)


# In[12]:


shuffled_cm_train_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=shuffled_X_train,
    y=y_train,
    shuffled=True,
    dataset_type="train",
)
shuffled_cm_test_df = generate_confusion_matrix_tl(
    model=shuffled_best_model, X=X_test, y=y_test, shuffled=True, dataset_type="test"
)


# ## Evaluating Multi-class model with holdout data

# Loading in all the hold out data

# In[13]:


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

# In[14]:


# evaluating plate holdout data with both trained original and shuffled model
plate_ho_precision_recall_df = calculate_multi_class_pr_curve(
    model=best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=False,
    dataset_type="plate_holdout",
)
plate_ho_f1_score_df = calculate_multi_class_f1score(
    model=best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=False,
    dataset_type="plate_holdout",
)


plate_ho_shuffle_precision_recall_df = calculate_multi_class_pr_curve(
    model=shuffled_best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=True,
    dataset_type="plate_holdout",
)
plate_ho_shuffle_f1_score_df = calculate_multi_class_f1score(
    model=shuffled_best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=True,
    dataset_type="plate_holdout",
)

# evaluating treatment holdout data with both trained original and shuffled model
treatment_ho_precision_recall_df = calculate_multi_class_pr_curve(
    model=best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=False,
    dataset_type="treatment_holdout",
)
treatment_ho_f1_score_df = calculate_multi_class_f1score(
    model=best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=False,
    dataset_type="treatment_holdout",
)
treatment_ho_shuffle_precision_recall_df = calculate_multi_class_pr_curve(
    model=shuffled_best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=True,
    dataset_type="treatment_holdout",
)
treatment_ho_shuffle_f1_score_df = calculate_multi_class_f1score(
    model=shuffled_best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=True,
    dataset_type="treatment_holdout",
)

# evaluating well holdout data with both trained original and shuffled model
well_ho_precision_recall_df = calculate_multi_class_pr_curve(
    model=best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=False,
    dataset_type="well_holdout",
)
well_ho_f1_score_df = calculate_multi_class_f1score(
    model=best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=False,
    dataset_type="well_holdout",
)
well_ho_shuffle_precision_recall_df = calculate_multi_class_pr_curve(
    model=shuffled_best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=True,
    dataset_type="well_holdout",
)
well_ho_shuffle_f1_score_df = calculate_multi_class_f1score(
    model=shuffled_best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=True,
    dataset_type="well_holdout",
)


# In[15]:


# creating confusing matrix with plate holdout (shuffled and not snuffled)
plate_ho_cm_df = generate_confusion_matrix_tl(
    model=best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=False,
    dataset_type="plate_holdout",
)
shuffled_plate_ho_cm_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=X_plate_holdout,
    y=y_plate_holdout,
    shuffled=True,
    dataset_type="plate_holdout",
)

# creating confusing matrix with treatment holdout (shuffled and not snuffled)
treatment_ho_cm_df = generate_confusion_matrix_tl(
    model=best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=False,
    dataset_type="treatment_holdout",
)
shuffled_treatment_ho_cm_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=X_treatment_holdout,
    y=y_treatment_holdout,
    shuffled=True,
    dataset_type="treatment_holdout",
)

# creating confusing matrix with plate_hold (shuffled and not snuffled)
well_ho_cm_df = generate_confusion_matrix_tl(
    model=best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=False,
    dataset_type="well_holdout",
)
shuffled_well_ho_cm_df = generate_confusion_matrix_tl(
    model=shuffled_best_model,
    X=X_well_holdout,
    y=y_well_holdout,
    shuffled=True,
    dataset_type="well_holdout",
)


# Storing all f1 and pr scores

# In[16]:


# storing all f1 scores
all_f1_scores = pd.concat(
    [
        test_f1_score_df,
        train_f1_score_df,
        shuffle_test_f1_score_df,
        shuffle_train_f1_score_df,
        plate_ho_f1_score_df,
        plate_ho_shuffle_f1_score_df,
        treatment_ho_f1_score_df,
        treatment_ho_shuffle_f1_score_df,
        well_ho_f1_score_df,
        well_ho_shuffle_f1_score_df,
    ]
)

# saving all f1 scores
all_f1_scores.to_csv(
    modeling_dir / "all_f1_scores.csv.gz", index=False, compression="gzip"
)


# In[17]:


# storing pr scores
all_pr_scores = pd.concat(
    [
        test_precision_recall_df,
        train_precision_recall_df,
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

# saving pr scores
all_pr_scores.to_csv(
    modeling_dir / "precision_recall_scores.csv.gz", index=False, compression="gzip"
)


# In[18]:


all_cm_dfs = pd.concat(
    [
        cm_train_df,
        cm_test_df,
        shuffled_cm_train_df,
        shuffled_cm_test_df,
        well_ho_cm_df,
        shuffled_well_ho_cm_df,
        treatment_ho_cm_df,
        shuffled_treatment_ho_cm_df,
        well_ho_cm_df,
        shuffled_well_ho_cm_df,
    ]
)


# saving pr scores
all_cm_dfs.to_csv(
    modeling_dir / "confusion_matrix.csv.gz", index=False, compression="gzip"
)


# In[19]:


all_cm_dfs
