#!/usr/bin/env python
# coding: utf-8

# # Module 2: Modeling
#
# In this notebook, we focus on developing and training a Multi-class logistic regression model, employing Randomized Cross-Validation (CV) for hyperparameter tuning to address our classification task. The dataset is split into an 80% training set and a 20% testing set. To evaluate the performance of our model during training, we used performance evaluation metrics such as precision, recall, and F1 scores. Additionally, we extend our evaluation by testing our model on a holdout dataset, which includes plate, treatment, and well information, providing a comprehensive assessment of its real-world performance.
#
# In this notebook, we will train four models:
#
# - A model using feature-selected cell injury profiles
# - A model using shuffled feature-selected cell injury profiles
# - A model using JUMP-aligned feature-selected cell injury profiles
# - A model using shuffled JUMP-aligned feature-selected cell injury profiles

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
    get_coeff_scores,
    load_json_file,
    shuffle_features,
    train_multiclass,
)

# Setting parameters and paths

# In[2]:


# setting random seeds variables
seed = 0
np.random.seed(seed)


# In[3]:


# setting directory paths
results_dir = pathlib.Path("../../results").resolve(strict=True)
feature_dir = (results_dir / "0.feature_selection/").resolve(strict=True)
data_splits_dir = (results_dir / "1.data_splits").resolve(strict=True)

# setting test and train data paths
aligned_X_train_path = (data_splits_dir / "aligned_X_train.csv.gz").resolve(strict=True)
aligned_X_test_path = (data_splits_dir / "aligned_X_test.csv.gz").resolve(strict=True)
aligned_y_train_path = (data_splits_dir / "aligned_y_train.csv.gz").resolve(strict=True)
aligned_y_test_path = (data_splits_dir / "aligned_y_test.csv.gz").resolve(strict=True)

fs_X_train_path = (data_splits_dir / "fs_X_train.csv.gz").resolve(strict=True)
fs_X_test_path = (data_splits_dir / "fs_X_test.csv.gz").resolve(strict=True)
fs_y_train_path = (data_splits_dir / "fs_y_train.csv.gz").resolve(strict=True)
fs_y_test_path = (data_splits_dir / "fs_y_test.csv.gz").resolve(strict=True)

# setting feature selected holdouts data paths
fs_plate_holdout_path = (data_splits_dir / "fs_plate_holdout.csv.gz").resolve(
    strict=True
)
fs_treatment_holdout_path = (data_splits_dir / "fs_treatment_holdout.csv.gz").resolve(
    strict=True
)
fs_well_holdout_path = (data_splits_dir / "fs_well_holdout.csv.gz").resolve(strict=True)

# set injury codes path
injury_codes_path = (feature_dir / "injury_codes.json").resolve(strict=True)

# setting feature spaces paths
fs_feature_space_path = (
    feature_dir / "fs_cell_injury_only_feature_space.json"
).resolve(strict=True)
aligned_feature_space_path = (
    feature_dir / "aligned_cell_injury_shared_feature_space.json"
).resolve(strict=True)

# setting output paths
modeling_dir = (results_dir / "2.modeling").resolve()
modeling_dir.mkdir(exist_ok=True)

# setting model paths
fs_model_path = modeling_dir / "fs_multi_class_model.joblib"
fs_shuffled_model_path = modeling_dir / "fs_shuffled_multi_class_model.joblib"

aligned_model_path = modeling_dir / "aligned_multi_class_model.joblib"
aligned_shuffled_model_path = modeling_dir / "aligned_shuffled_multi_class_model.joblib"

# setting cross-validations scores paths
fs_model_cv_results_path = modeling_dir / "fs_multi_class_cv_results.csv"
fs_shuffled_model_cv_results_path = (
    modeling_dir / "fs_shuffled_multi_class_cv_results.csv"
)

aligned_model_cv_results_path = modeling_dir / "aligned_multi_class_cv_results.csv"
aligned_shuffled_model_cv_results_path = (
    modeling_dir / "aligned_shuffled_multi_class_cv_results.csv"
)


# In[4]:


# loading data splits
aligned_X_train_df = pd.read_csv(aligned_X_train_path)
aligned_X_test_df = pd.read_csv(aligned_X_test_path)
aligned_y_train_df = pd.read_csv(aligned_y_train_path)
aligned_y_test_df = pd.read_csv(aligned_y_test_path)

fs_X_train_df = pd.read_csv(fs_X_train_path)
fs_X_test_df = pd.read_csv(fs_X_test_path)
fs_y_train_df = pd.read_csv(fs_y_train_path)
fs_y_test_df = pd.read_csv(fs_y_test_path)

# loading fs_holdouts
fs_plate_holdout_df = pd.read_csv(fs_plate_holdout_path)
fs_treatment_holdout_df = pd.read_csv(fs_treatment_holdout_path)
fs_well_holdout_df = pd.read_csv(fs_well_holdout_path)

# load injury codes
injury_codes = load_json_file(injury_codes_path)

# loading feature spaces
fs_feature_space = load_json_file(fs_feature_space_path)
aligned_feature_space = load_json_file(aligned_feature_space_path)

fs_meta = fs_feature_space["meta_features"]
fs_feats = fs_feature_space["features"]

aligned_aligned = aligned_feature_space["meta_features"]
aligned_feats = aligned_feature_space["features"]

# checking if the feature space are identical (also looks for feature space order)
assert check_feature_order(
    ref_feat_order=aligned_feats, input_feat_order=aligned_X_test_df.columns.tolist()
), "Feature space are not identical"


# ## Training and evaluating Multi-class Logistic model with feature selected cell injury profiles (not jump aligned)

# Below are the parameters used:
#
# - **penalty**: Specifies the type of penalty (regularization) applied during logistic regression. It can be 'l1' for L1 regularization, 'l2' for L2 regularization, or 'elasticnet' for a combination of both.
# - **C**: Inverse of regularization strength; smaller values specify stronger regularization. Controls the trade-off between fitting the training data and preventing overfitting.
# - **max_iter**: Maximum number of iterations for the optimization algorithm to converge.
# - **tol**: Tolerance for the stopping criterion during optimization. It represents the minimum change in coefficients between iterations that indicates convergence.
# - **l1_ratio**: The mixing parameter for elastic net regularization. It determines the balance between L1 and L2 penalties in the regularization term. A value of 1 corresponds to pure L1 (Lasso) penalty, while a value of 0 corresponds to pure L2 (Ridge) penalty
# - **solver**: Optimization algorithms to be explored during hyperparameter tuning for logistic regression

# In[5]:


param_grid = {
    "penalty": ["l1", "l2", "elasticnet"],
    "C": [0.001, 0.01, 0.1, 1, 10, 100],
    "max_iter": np.arange(100, 1100, 100),
    "tol": np.arange(1e-6, 1e-3, 1e-6),
    "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
    "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
}


# ### Training both non shuffled and shuffled models with feature selected cell injury profiles (not aligned with JUMP)

# In[6]:


# if trained model exists, skip training
if fs_model_path.exists():
    fs_best_model = joblib.load(fs_model_path)

# train model and save
else:
    fs_best_model = train_multiclass(
        fs_X_train_df,
        fs_y_train_df,
        param_grid=param_grid,
        seed=seed,
        cv_results_outpath=fs_model_cv_results_path,
    )
    joblib.dump(fs_best_model, fs_model_path)


# Training shuffled model
# shuffle feature space
fs_shuffled_X_train = shuffle_features(fs_X_train_df, features=fs_feats, seed=seed)

# checking if the shuffled and original feature space are the same
assert not fs_shuffled_X_train.equals(fs_X_train_df), "DataFrames are the same!"

if fs_shuffled_X_train.equals(fs_X_train_df):
    raise ValueError("Error: DataFrames are the same")

# if trained model exists, skip training
if fs_shuffled_model_path.exists():
    fs_shuffled_best_model = joblib.load(fs_shuffled_model_path)


# train model and save
else:
    fs_shuffled_best_model = train_multiclass(
        fs_shuffled_X_train,
        fs_y_train_df,
        param_grid=param_grid,
        seed=seed,
        cv_results_outpath=fs_shuffled_model_cv_results_path,
    )
    joblib.dump(fs_shuffled_best_model, fs_shuffled_model_path)


# ### Evaluating both shuffled and non shuffled models with original data split

# In[7]:


# evaluating model on train dataset
train_precision_recall_df, train_f1_score_df = evaluate_model_performance(
    model=fs_best_model,
    X=fs_X_train_df,
    y=fs_y_train_df,
    shuffled=False,
    dataset_type="Train",
)

# evaluating model on test dataset
test_precision_recall_df, test_f1_score_df = evaluate_model_performance(
    model=fs_best_model,
    X=fs_X_test_df,
    y=fs_y_test_df,
    shuffled=False,
    dataset_type="Test",
)


# In[8]:


# evaluating shuffled model on train dataset
shuffle_train_precision_recall_df, shuffle_train_f1_score_df = (
    evaluate_model_performance(
        model=fs_shuffled_best_model,
        X=fs_shuffled_X_train,
        y=fs_y_train_df,
        shuffled=True,
        dataset_type="Train",
    )
)

# evaluating shuffled model on test dataset
shuffle_test_precision_recall_df, shuffle_test_f1_score_df = evaluate_model_performance(
    model=fs_shuffled_best_model,
    X=fs_X_test_df,
    y=fs_y_test_df,
    shuffled=True,
    dataset_type="Test",
)


# ### Creating confusion matrix with both shuffle and non shuffled models with original data split

# In[9]:


# creating confusion matrix for both train and test set on non-shuffled model
cm_train_df = generate_confusion_matrix_tl(
    model=fs_best_model,
    X=fs_X_train_df,
    y=fs_y_train_df,
    shuffled=False,
    dataset_type="Train",
)
cm_test_df = generate_confusion_matrix_tl(
    model=fs_best_model,
    X=fs_X_test_df,
    y=fs_y_test_df,
    shuffled=False,
    dataset_type="Test",
)


# In[10]:


# creating confusion matrix for shuffled model
shuffled_cm_train_df = generate_confusion_matrix_tl(
    model=fs_shuffled_best_model,
    X=fs_shuffled_X_train,
    y=fs_y_train_df,
    shuffled=True,
    dataset_type="Train",
)
shuffled_cm_test_df = generate_confusion_matrix_tl(
    model=fs_shuffled_best_model,
    X=fs_X_test_df,
    y=fs_y_test_df,
    shuffled=True,
    dataset_type="Test",
)


# ### Evaluating both shuffled and non Multi-class model with holdout data

# In[11]:


# evaluating plate holdout data with both trained original and shuffled model
plate_ho_precision_recall_df, plate_ho_f1_score_df = evaluate_model_performance(
    model=fs_best_model,
    X=fs_plate_holdout_df[fs_feats],
    y=fs_plate_holdout_df["injury_code"],
    shuffled=False,
    dataset_type="Plate Holdout",
)

plate_ho_shuffle_precision_recall_df, plate_ho_shuffle_f1_score_df = (
    evaluate_model_performance(
        model=fs_shuffled_best_model,
        X=fs_plate_holdout_df[fs_feats],
        y=fs_plate_holdout_df["injury_code"],
        shuffled=True,
        dataset_type="Plate Holdout",
    )
)


# evaluating treatment holdout data with both trained original and shuffled model
treatment_ho_precision_recall_df, treatment_ho_f1_score_df = evaluate_model_performance(
    model=fs_best_model,
    X=fs_treatment_holdout_df[fs_feats],
    y=fs_treatment_holdout_df["injury_code"],
    shuffled=False,
    dataset_type="Treatment Holdout",
)

treatment_ho_shuffle_precision_recall_df, treatment_ho_shuffle_f1_score_df = (
    evaluate_model_performance(
        model=fs_shuffled_best_model,
        X=fs_treatment_holdout_df[fs_feats],
        y=fs_treatment_holdout_df["injury_code"],
        shuffled=True,
        dataset_type="Treatment Holdout",
    )
)

# evaluating well holdout data with both trained original and shuffled model
well_ho_precision_recall_df, well_ho_f1_score_df = evaluate_model_performance(
    model=fs_best_model,
    X=fs_well_holdout_df[fs_feats],
    y=fs_well_holdout_df["injury_code"],
    shuffled=False,
    dataset_type="Well Holdout",
)

well_ho_shuffle_precision_recall_df, well_ho_shuffle_f1_score_df = (
    evaluate_model_performance(
        model=fs_shuffled_best_model,
        X=fs_well_holdout_df[fs_feats],
        y=fs_well_holdout_df["injury_code"],
        shuffled=True,
        dataset_type="Well Holdout",
    )
)


# ### Creating confusion matrix with both shuffle and non shuffled models with holdout data

# In[12]:


# creating confusion matrix with plate holdout (shuffled and not shuffled)
plate_ho_cm_df = generate_confusion_matrix_tl(
    model=fs_best_model,
    X=fs_plate_holdout_df[fs_feats],
    y=fs_plate_holdout_df["injury_code"],
    shuffled=False,
    dataset_type="Plate Holdout",
)
shuffled_plate_ho_cm_df = generate_confusion_matrix_tl(
    model=fs_shuffled_best_model,
    X=fs_plate_holdout_df[fs_feats],
    y=fs_plate_holdout_df["injury_code"],
    shuffled=True,
    dataset_type="Plate Holdout",
)

# creating confusion matrix with treatment holdout (shuffled and not shuffled)
treatment_ho_cm_df = generate_confusion_matrix_tl(
    model=fs_best_model,
    X=fs_treatment_holdout_df[fs_feats],
    y=fs_treatment_holdout_df["injury_code"],
    shuffled=False,
    dataset_type="Treatment Holdout",
)
shuffled_treatment_ho_cm_df = generate_confusion_matrix_tl(
    model=fs_shuffled_best_model,
    X=fs_treatment_holdout_df[fs_feats],
    y=fs_treatment_holdout_df["injury_code"],
    shuffled=True,
    dataset_type="Treatment Holdout",
)

# creating confusion matrix with plate_hold (shuffled and not shuffled)
well_ho_cm_df = generate_confusion_matrix_tl(
    model=fs_best_model,
    X=fs_well_holdout_df[fs_feats],
    y=fs_well_holdout_df["injury_code"],
    shuffled=False,
    dataset_type="Well Holdout",
)
shuffled_well_ho_cm_df = generate_confusion_matrix_tl(
    model=fs_shuffled_best_model,
    X=fs_well_holdout_df[fs_feats],
    y=fs_well_holdout_df["injury_code"],
    shuffled=True,
    dataset_type="Well Holdout",
)


# ### Saving all model evaluations

# #### Storing all f1_scores

# In[13]:


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
    modeling_dir / "fs_all_f1_scores.csv.gz", index=False, compression="gzip"
)


# #### Saving all precision and recall scores

# In[14]:


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
    modeling_dir / "fs_precision_recall_scores.csv.gz", index=False, compression="gzip"
)


# ### Saving all confusion matrices

# In[15]:


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
    modeling_dir / "fs_confusion_matrix.csv.gz", index=False, compression="gzip"
)


# ### Training model with JUMP aligned feature selected cell injury profiles

# In[16]:


# if trained aligned model exists, skip training
if aligned_model_path.exists():
    aligned_best_model = joblib.load(aligned_model_path)

# train model with aligned cell injury profiles and save
else:
    aligned_best_model = train_multiclass(
        aligned_X_train_df[aligned_feats],
        aligned_y_train_df["injury_code"],
        param_grid=param_grid,
        seed=seed,
        cv_results_outpath=aligned_model_cv_results_path,
    )
    joblib.dump(aligned_best_model, aligned_model_path)


# Training shuffled model
# shuffle feature space
aligned_shuffled_X_train = shuffle_features(
    aligned_X_train_df, features=aligned_feats, seed=seed
)

# checking if the shuffled and original feature space are the same
if aligned_X_train_df.equals(aligned_shuffled_X_train):
    raise ValueError("Error: DataFrames are the same")

# if trained shuffled aligned model exists, skip training
if aligned_shuffled_model_path.exists():
    aligned_shuffled_best_model = joblib.load(aligned_shuffled_model_path)

# train model with shuffled aligned cell injury data and save
else:
    aligned_shuffled_best_model = train_multiclass(
        aligned_shuffled_X_train[aligned_feats],
        aligned_y_train_df["injury_code"],
        param_grid=param_grid,
        seed=seed,
        cv_results_outpath=aligned_shuffled_model_cv_results_path,
    )
    joblib.dump(aligned_shuffled_best_model, aligned_shuffled_model_path)


# ## Extracting coefficient scores
#
# Next, we will extract the coefficient scores for all morphological features for each class and save the results into a single CSV file.
#
# The generated CSV file will include the following columns:
# - **injury_id**: A numeric identifier assigned to each injury type.
# - **injury_name**: The name of the injury type.
# - **feature**: The name of the morphological feature.
# - **coefficient**: The coefficient score associated with the morphological feature for the specific injury class.
# - **model_name**: The name of the model that generated the scores, good for tracking.
#
# This file will allow for easy examination of the importance of each feature in predicting different injury types, facilitating a deeper understanding of the model's behavior.

# In[17]:


# Extracting coefficient scores from both models
fs_coeff_df = get_coeff_scores(
    best_model=fs_best_model,
    features=fs_feats,
    injury_codes=injury_codes,
    model_name="fs_model",
)
aligned_coeff_df = get_coeff_scores(
    best_model=aligned_best_model,
    features=aligned_feats,
    injury_codes=injury_codes,
    model_name="JUMP_aligned_model",
)

# concatenating the coefficient scores
all_coeff_scores = pd.concat(
    [
        fs_coeff_df,
        aligned_coeff_df,
    ]
).reset_index(drop=True)

# save
all_coeff_scores.to_csv(modeling_dir / "all_model_coeff_scores.csv", index=False)

# display
all_coeff_scores.head()
