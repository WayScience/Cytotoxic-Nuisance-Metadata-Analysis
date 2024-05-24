#!/usr/bin/env python
# coding: utf-8

# # Spliting Data
# Here, we utilize the feature-selected profiles generated in the preceding module notebook [here](../0.feature_selection/0.feature_selection.ipynb), focusing on dividing the data into training, testing, and holdout sets for machine learning training.

# In[1]:


import json
import pathlib
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append("../../")  # noqa
from src.utils import get_injury_treatment_info, split_meta_and_features  # noqa

# ignoring warnings
warnings.catch_warnings(action="ignore")


# ## Paramters
#
# Below are the parameters defined that are used in this notebook

# In[2]:


# setting seed constants
seed = 0
np.random.seed(seed)

# directory to get all the inputs for this notebook
data_dir = pathlib.Path("../../data").resolve(strict=True)
results_dir = pathlib.Path("../../results").resolve(strict=True)
fs_dir = (results_dir / "0.feature_selection").resolve(strict=True)

# directory to store all the output of this notebook
data_split_dir = (results_dir / "1.data_splits").resolve()
data_split_dir.mkdir(exist_ok=True)


# In[3]:


# data paths
fs_profile_path = (fs_dir / "cell_injury_profile_fs.csv.gz").resolve(strict=True)

# load data
fs_profile_df = pd.read_csv(fs_profile_path)

# splitting meta and feature column names
fs_meta, fs_feats = split_meta_and_features(fs_profile_df)

# display
print("fs profile with control: ", fs_profile_df.shape)
fs_profile_df.head()


# ## Exploring the data set
#
# Below is a  exploration of the selected features dataset. The aim is to identify treatments, extract metadata, and gain a understanding of the experiment's design.

# Below demonstrates the amount of wells does each treatment have.

# In[4]:


# displying the amount of wells per treatments
well_treatments_counts_df = (
    fs_profile_df["Compound Name"].value_counts().to_frame().reset_index()
)

well_treatments_counts_df


# Below we show the amount of wells does a specific cell celluar injury has

# In[5]:


# Displaying how many how wells does each cell injury have
cell_injury_well_counts = (
    fs_profile_df["injury_type"].value_counts().to_frame().reset_index()
)
cell_injury_well_counts


# Next we wanted to extract some metadata regarding how many compound and wells are treated with a given compounds
#
# This will be saved in the `results/0.data_splits` directory

# In[6]:


# get summary information and save it
injury_before_holdout_info_df = get_injury_treatment_info(
    profile=fs_profile_df, groupby_key="injury_type"
).reset_index(drop=True)
injury_before_holdout_info_df.to_csv(
    data_split_dir / "injury_data_summary_before_holdout.csv", index=False
)

# display
print("Shape:", injury_before_holdout_info_df.shape)
injury_before_holdout_info_df


# Next, we construct the profile metadata. This provides a structured overview of how the treatments assicoated with injuries were applied, detailing the treatments administered to each plate.
#
# This will be saved in the `results/0.data_splits` directory

# In[7]:


injury_meta_dict = {}
for injury, df in fs_profile_df.groupby("injury_type"):
    # collecting treatment metadata
    plates = df["Plate"].unique().tolist()
    treatment_meta = {}
    treatment_meta["n_plates"] = len(plates)
    treatment_meta["n_wells"] = df.shape[0]
    treatment_meta["n_treatments"] = len(df["Compound Name"].unique())
    treatment_meta["associated_plates"] = plates

    # counting treatments
    treatment_counter = {}
    for treatment, df2 in df.groupby("Compound Name"):
        if treatment is np.nan:
            continue
        n_treatments = df2.shape[0]
        treatment_counter[treatment] = n_treatments

    # storing treatment counts
    treatment_meta["treatments"] = treatment_counter
    injury_meta_dict[injury] = treatment_meta

# save dictionary into a json file
with open(data_split_dir / "injury_metadata.json", mode="w") as stream:
    json.dump(injury_meta_dict, stream)


# Here we build a plate metadata infromations where we look at the type of treatments and amount of wells with the treatment that are present in the dataset
#
# This will be saved in `results/0.data_splits`

# In[8]:


plate_meta = {}
for plate_id, df in fs_profile_df.groupby("Plate"):
    unique_compounds = list(df["Compound Name"].unique())
    n_treatments = len(unique_compounds)

    # counting treatments
    treatment_counter = {}
    for treatment, df2 in df.groupby("Compound Name"):
        n_treatments = df2.shape[0]
        treatment_counter[treatment] = n_treatments

    plate_meta[plate_id] = treatment_counter

# save dictionary into a json file
with open(data_split_dir / "cell_injury_plate_info.json", mode="w") as stream:
    json.dump(plate_meta, stream)


# ## Data Splitting
# ---

# ### Holdout Dataset
#
# Here we collected out holdout dataset. The holdout dataset is a subset of the dataset that is not used during model training or tuning. Instead, it is reserved solely for evaluating the model's performance after it has been trained.
#
# In this notebook, we will include three different types of held-out datasets before proceeding with our machine learning training and evaluation.
#  - Plate hold out
#  - treatment hold out
#  - well hold out
#
# Each of these held outdata will be stored in the `results/1.data_splits` directory
#

# ### Plate Holdout
#
# Plates are randomly selected based on their Plate ID and save them as our `plate_holdout` data.

# In[9]:


# plate
seed = 0
n_plates = 10

# setting random seed globally
np.random.seed(seed)

# selecting plates randomly from a list
selected_plates = (
    np.random.choice(fs_profile_df["Plate"].unique().tolist(), (n_plates, 1))
    .flatten()
    .tolist()
)
plate_holdout_df = fs_profile_df.loc[fs_profile_df["Plate"].isin(selected_plates)]

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
plate_idx_to_drop = plate_holdout_df.index.tolist()
fs_profile_df = fs_profile_df.drop(plate_idx_to_drop)
assert all(
    [
        True if num not in fs_profile_df.index.tolist() else False
        for num in plate_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
plate_holdout_df.to_csv(
    data_split_dir / "plate_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("plate holdout shape:", plate_holdout_df.shape)
plate_holdout_df.head()


# ### Treatment holdout
#
# To establish our treatment holdout, we first need to find the number of treatments and wells associated with a specific cell injury, considering the removal of randomly selected plates from the previous step.
#
# To determine which cell injuries should be considered for a single treatment holdout, we establish a threshold of 10 unique compounds. This means that a cell injury type must have at least 10 unique compounds to qualify for selection in the treatment holdout. Any cell injury types failing to meet this criterion will be disregarded.
#
# Once the cell injuries are identified for treatment holdout, we select our holdout treatment by grouping each injury type and choosing the treatment with the fewest wells. This becomes our treatment holdout dataset.

# In[10]:


injury_treatment_metadata = (
    fs_profile_df.groupby(["injury_type", "Compound Name"])
    .size()
    .reset_index(name="n_wells")
)
injury_treatment_metadata


# In[11]:


# setting random seed
min_treatments_per_injury = 10

# Filter out the injury types for which we can select a complete treatment.
# We are using a threshold of 10. If an injury type is associated with fewer than 10 compounds,
# we do not conduct treatment holdout on those injury types.
accepted_injuries = []
for injury_type, df in injury_treatment_metadata.groupby("injury_type"):
    n_treatments = df.shape[0]
    if n_treatments >= min_treatments_per_injury:
        accepted_injuries.append(df)

accepted_injuries = pd.concat(accepted_injuries)

# Next, we select the treatment that will be held out within each injury type.
# We group treatments based on injury type and choose the treatment with the fewest wells
# as our holdout.
selected_treatments_to_holdout = []
for injury_type, df in accepted_injuries.groupby("injury_type"):
    held_treatment = df.min().iloc[1]
    selected_treatments_to_holdout.append([injury_type, held_treatment])

# convert to dataframe
selected_treatments_to_holdout = pd.DataFrame(
    selected_treatments_to_holdout, columns="injury_type held_treatment".split()
)

print("Below are the accepted cell injuries and treatments to be held out")
selected_treatments_to_holdout


# In[12]:


# select all wells that have the treatments to be heldout
treatment_holdout_df = fs_profile_df.loc[
    fs_profile_df["Compound Name"].isin(
        selected_treatments_to_holdout["held_treatment"]
    )
]

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
treatment_idx_to_drop = treatment_holdout_df.index.tolist()
fs_profile_df = fs_profile_df.drop(treatment_idx_to_drop)
assert all(
    [
        True if num not in fs_profile_df.index.tolist() else False
        for num in treatment_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"
# saving the holdout data
treatment_holdout_df.to_csv(
    data_split_dir / "treatment_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("Treatment holdout shape:", treatment_holdout_df.shape)
treatment_holdout_df.head()


# ### Well holdout
#
# To generate the well hold out data, each plate was iterated and random wells were selected. However, an additional step was condcuting which was to seperate the control wells and the treated wells, due to the large label imbalance with the controls. Therefore, 5 wells were randomly selected and 10 wells were randomly selected from each individual plate

# In[13]:


# parameters
seed = 0
n_controls = 5
n_samples = 10

# setting random seed globally
np.random.seed(seed)

# collecting randomly select wells based on treatment
wells_heldout_df = []
for treatment, df in fs_profile_df.groupby("Plate", as_index=False):
    # separate control wells and rest of all wells since there is a huge label imbalance
    # selected 5 control wells and 10 random wells from the plate
    df_control = df.loc[df["Compound Name"] == "DMSO"].sample(
        n=n_controls, random_state=seed
    )
    df_treated = df.loc[df["Compound Name"] != "DMSO"].sample(
        n=n_samples, random_state=seed
    )

    # concatenate those together
    well_heldout = pd.concat([df_control, df_treated])

    wells_heldout_df.append(well_heldout)

# genearte treatment holdout dataframe
wells_heldout_df = pd.concat(wells_heldout_df)

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
wells_idx_to_drop = wells_heldout_df.index.tolist()
fs_profile_df = fs_profile_df.drop(wells_idx_to_drop)
assert all(
    [
        True if num not in fs_profile_df.index.tolist() else False
        for num in treatment_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
wells_heldout_df.to_csv(
    data_split_dir / "wells_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("Wells holdout shape:", wells_heldout_df.shape)
wells_heldout_df.head()


# ## Saving training dataset

# Once the data holdout has been generated, the next step is to save the training dataset that will serve as the basis for training the multi-class logistic regression model.

# In[14]:


# get summary cell injury dataset treatment and well info after holdouts
injury_after_holdout_info_df = get_injury_treatment_info(
    profile=fs_profile_df, groupby_key="injury_type"
)
injury_after_holdout_info_df.to_csv(
    data_split_dir / "injury_data_summary_after_holdout.csv", index=False
)

# display
print("shape:", injury_after_holdout_info_df.shape)
injury_after_holdout_info_df


# In[15]:


# shape of the update training and testing dataset after removing holdout
print("training shape after removing holdouts", fs_profile_df.shape)
fs_profile_df.head()


# In[16]:


# split the data into trianing and testing sets
meta_cols, feat_cols = split_meta_and_features(fs_profile_df)
X = fs_profile_df[feat_cols]
y = fs_profile_df["injury_code"]

# spliting dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.80, random_state=seed, stratify=y
)

# saving training dataset as csv file
X_train.to_csv(data_split_dir / "X_train.csv.gz", compression="gzip", index=False)
X_test.to_csv(data_split_dir / "X_test.csv.gz", compression="gzip", index=False)
y_train.to_csv(data_split_dir / "y_train.csv.gz", compression="gzip", index=False)
y_test.to_csv(data_split_dir / "y_test.csv.gz", compression="gzip", index=False)

# display data split sizes
print("X training size", X_train.shape)
print("X testing size", X_test.shape)
print("y training size", y_train.shape)
print("y testing size", y_test.shape)


# In[17]:


# save metadata after holdout
cell_injury_metadata = fs_profile_df[fs_meta]
cell_injury_metadata.to_csv(
    data_split_dir / "cell_injury_metadata_after_holdout.csv.gz",
    compression="gzip",
    index=False,
)

# display
print("Metadata shape", cell_injury_metadata.shape)
cell_injury_metadata.head()


# In[18]:


injury_train_info_df = get_injury_treatment_info(
    profile=X_train.merge(
        fs_profile_df[meta_cols], how="left", right_index=True, left_index=True
    )[meta_cols + feat_cols],
    groupby_key="injury_type",
)
injury_test_info_df = get_injury_treatment_info(
    profile=X_test.merge(
        fs_profile_df[meta_cols], how="left", right_index=True, left_index=True
    )[meta_cols + feat_cols],
    groupby_key="injury_type",
)

# save both files
injury_train_info_df.to_csv(
    data_split_dir / "injury_data_summary_train_split.csv", index=False
)
injury_test_info_df.to_csv(
    data_split_dir / "injury_data_summary_test_split.csv", index=False
)


# In[19]:


print("Showing summary data of train split")
injury_train_info_df


# In[20]:


print("Showing summary data of test split")
injury_test_info_df
