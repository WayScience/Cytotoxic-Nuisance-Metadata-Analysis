#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import uniform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# import local modules
sys.path.append("../../")
from src.utils import evaluate, load_json_file, shuffle_features, train_multiclass

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

# ml parameters to hyperparameterization tuning
param_grid = {
    "estimator__C": uniform(0.1, 10),
    "estimator__solver": ["newton-cg", "liblinear", "sag", "saga"],
    "estimator__penalty": ["l1", "l2", "elasticnet"],
    "estimator__l1_ratio": uniform(0, 1),
}


# In[3]:


# loading injurt codes
injury_codes = load_json_file(data_splits_dir / "injury_codes.json")

# loading in the dataset
training_df = pd.read_csv(training_dataset_path)

# display data
print("Shape: ", training_df.shape)
training_df.head()


# In[4]:


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


# ## Training and Evaluating Multi-class Logistic Model with original dataset split
#

# In[5]:


# train and get the best_model
best_model = train_multiclass(X_train, y_train, param_grid=param_grid, seed=0)


# In[6]:


test_precision_recall_df, test_f1_score_df = evaluate(
    model=best_model, X=X_test, y=y_test, dataset="test", shuffled=False, seed=0
)
train_precision_recall_df, train_f1_score_df = evaluate(
    model=best_model, X=X_train, y=y_train, dataset="train", shuffled=False, seed=0
)


# ## Training and Evaluating Multi-class Logistic Model with shuffled dataset split
#

# In[7]:


# shuffle feature space
shuffled_X_train = shuffle_features(X_train, seed=seed)


# In[8]:


shuffled_best_model = train_multiclass(
    shuffled_X_train, y_train, param_grid=param_grid, seed=seed
)


# In[9]:


shuffle_test_precision_recall_df, shuffle_test_f1_score_df = evaluate(
    model=best_model, X=X_test, y=y_test, dataset="test", shuffled=True, seed=0
)
shuffle_train_precision_recall_df, shuffle_train_f1_score_df = evaluate(
    model=best_model, X=X_train, y=y_train, dataset="train", shuffled=True, seed=0
)


# ## Evaluating Multi-class model with holdout data

# In[10]:


# loading in holdout data
# setting seed
n_classes = len(np.unique(y_labels.values))

# loading all holdouts
plate_holdout_df = pd.read_csv(plate_holdout_path)
treatment_holdout_df = pd.read_csv(treatment_holdout_path)
well_holdout_df = pd.read_csv(wells_holdout_path)

# splitting the dataset into
X_plate_holdout = plate_holdout_df[feat_cols]
y_plate_holout = label_binarize(
    y=plate_holdout_df["injury_code"],
    classes=[*range(n_classes)],
)

X_treatment_holdout = treatment_holdout_df[feat_cols]
y_treatment_holout = label_binarize(
    y=treatment_holdout_df["injury_code"],
    classes=[*range(n_classes)],
)

X_well_holdout = well_holdout_df[feat_cols]
y_well_holout = label_binarize(
    y=well_holdout_df["injury_code"],
    classes=[*range(n_classes)],
)


# ### Evaluating Multi-class model trained with original split with holdout data

# In[11]:


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


# In[12]:


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

# saving all f1 scores
all_f1_scores.to_csv(
    modeling_dir / "all_f1_scores.csv.gz", index=False, compression="gzip"
)


# In[13]:


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

# saving pr scores
all_pr_scores.to_csv(
    modeling_dir / "precision_recall_scores.csv.gz", index=False, compression="gzip"
)
