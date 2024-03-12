#!/usr/bin/env python
# coding: utf-8

# # Spliting Data
# Here, we utilize the feature-selected profiles generated in the preceding module notebook [here](../0.freature_selection/), focusing on dividing the data into training, testing, and holdout sets for machine learning training.

# In[1]:


import json
import pathlib
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ignoring warnings
warnings.catch_warnings(action="ignore")


# ## Paramters
#
# Below are the parameters defined that are used in this notebook

# ----

# In[2]:


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


# Here, we're storing the metadata and feature column names into a JSON file to simplify loading during feature engineering processes.
#
# This will be saved in the `results/0.data_splits` directory

# In[6]:


# collecting metadata and feature column names
feature_cols = fs_profile_df.columns[32:].tolist()
raw_features = {
    "compartments": list(set([name.split("_")[0] for name in feature_cols])),
    "meta_features": fs_profile_df.columns[:32].tolist(),
    "feature_cols": feature_cols,
}

# saving into JSON file
with open(data_split_dir / "raw_feature_names.json", mode="w") as stream:
    json.dump(raw_features, stream)


# Next we wanted to extract some metadata regarding how many compound and wells are treated with a given compounds
#
# This will be saved in the `results/0.data_splits` directory

# In[7]:


meta_injury = []
for injury_type, df in fs_profile_df.groupby("injury_type"):
    # extract n_wells, n_compounds and unique compounds per injury_type
    n_wells = df.shape[0]
    unique_compounds = list(df["Compound Name"].unique())
    n_compounds = len(unique_compounds)

    # store information
    meta_injury.append([injury_type, n_wells, n_compounds, unique_compounds])

injury_meta_df = pd.DataFrame(
    meta_injury, columns=["injury_type", "n_wells", "n_compounds", "compound_list"]
).sort_values("n_wells", ascending=False)
injury_meta_df.to_csv(data_split_dir / "injury_well_counts_table.csv", index=False)

# display
print("shape:", injury_meta_df.shape)
injury_meta_df


# > Barchart showing the number of wells that are labeled with a given injury

# Next, we construct the profile metadata. This provides a structured overview of how the treatments assicoated with injuries were applied, detailing the treatments administered to each plate.
#
# This will be saved in the `results/0.data_splits` directory

# In[8]:


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

# In[9]:


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
with open(data_split_dir / "plate_info.json", mode="w") as stream:
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

# ## Holdout plate
#
# Plates are randomly selected based on their Plate ID and save them as our `plate_holdout` data.

# In[10]:


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


# ### Holdout out a treatment holdout plate
#
# To create our treatment holdout dataset, we group all wells treated with the same compound, then randomly select 15 wells per treatment group.

# In[11]:


#### Plate heldout dataset
seed = 0
n_samples = 15

# collecting randomly select wells based on treatment
treatment_holdout_df = []
for treatment, df in fs_profile_df.groupby("Compound Name", as_index=False):
    heldout_treatment = df.sample(n=10, random_state=seed, replace="True")
    treatment_holdout_df.append(heldout_treatment)

# genearte treatment holdout dataframe
treatment_holdout_df = pd.concat(treatment_holdout_df)

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


# ### Generating well holdout data
#
# To generate the well hold out data, each plate was iterated and random wells were selected. However, an additional step was condcuting which was to seperate the control wells and the treated wells, due to the large label imbalance with the controls. Therefore, 5 wells were randomly selected and 10 wells were randomly selected from each individual plate

# In[12]:


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


# In[13]:


# Showing the amount of data we have after removing the holdout data
meta_injury = []
for injury_type, df in fs_profile_df.groupby("injury_type"):
    # extract n_wells, n_compounds and unique compounds per injury_type
    n_wells = df.shape[0]
    unique_compounds = list(df["Compound Name"].unique())
    n_compounds = len(unique_compounds)

    # store information
    meta_injury.append([injury_type, n_wells, n_compounds, unique_compounds])

# creating data frame
injury_meta_df = pd.DataFrame(
    meta_injury, columns=["injury_type", "n_wells", "n_compounds", "compound_list"]
).sort_values("n_wells", ascending=False)
injury_meta_df.to_csv(data_split_dir / "injury_well_counts_table.csv", index=False)

# display
injury_meta_df


# In[14]:


# shape of the update training and testing dataset after removing holdout
print("training shape after removing holdouts", fs_profile_df.shape)
fs_profile_df.head()


# ### Splitting the data
#
# Splitting the data and saving them into csv files:
# Files are split into test and training dataset.
#

# In[15]:


# spliting the meta features and the feature column names
# loading feature columns json file
with open(data_split_dir / "raw_feature_names.json") as stream:
    feature_info = json.load(stream)

# selecing columns for splitting
y_col = "injury_type"
X_cols = feature_info["feature_cols"]


# In[16]:


# spliting the dataset
seed = 0

X = fs_profile_df[X_cols]
y = fs_profile_df[y_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed, stratify=y
)


# In[17]:


X_train.to_csv(data_split_dir / "X_train.csv.gz", index=False, compression="gzip")
y_train.to_csv(data_split_dir / "y_train.csv.gz", index=False, compression="gzip")
X_test.to_csv(data_split_dir / "X_test.csv.gz", index=False, compression="gzip")
y_test.to_csv(data_split_dir / "y_test.csv.gz", index=False, compression="gzip")
