#!/usr/bin/env python
# coding: utf-8

# # Feature Processing and Selection
#
# This notebook focuses on exploration using two essential files: the annotations data extracted from the actual screening profile (available in the [IDR repository](https://github.com/IDR/idr0133-dahlin-cellpainting/tree/main/screenA)) and the metadata retrieved from the supplementary section of the [research paper](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36829-x/MediaObjects/41467_2023_36829_MOESM5_ESM.xlsx).
#
# We explore the number of unique compounds associated with each cell injury and subsequently cross-reference this information with the screening profile. The aim is to assess the feasibility of using the data for training a machine learning model to predict cell injury.
#
# We apply feature selection through [pycytominer](https://github.com/cytomining/pycytominer) to capture the most informative features representing various cellular injury types within the morphology space. Then, we utilize the selected feature profiles for machine learning applications.

# In[1]:


import json
import pathlib
import sys
from collections import defaultdict

import pandas as pd
from pycytominer import feature_select

sys.path.append("../../")
from src import utils

# Setting up paths and parameters

# In[2]:


# data directory
data_dir = pathlib.Path("../../data").resolve(strict=True)
results_dir = pathlib.Path("../../results").resolve(strict=True)
fs_dir = (results_dir / "0.feature_selection").resolve()
fs_dir.mkdir(exist_ok=True)

# jump feature space path
jump_feature_space_path = (data_dir / "JUMP_data/jump_feature_space.json").resolve(
    strict=True
)

# data paths
suppl_meta_path = (data_dir / "41467_2023_36829_MOESM5_ESM.csv.gz").resolve(strict=True)
screen_anno_path = (data_dir / "idr0133-screenA-annotation.csv.gz").resolve(strict=True)

# load data
image_profile_df = pd.read_csv(screen_anno_path)

# spit columns and only get metadata dataframe
meta, feature = utils.split_meta_and_features(image_profile_df)
meta_df = image_profile_df[meta]

compounds_df = meta_df[["Compound Name", "Compound Class"]]

suppl_meta_df = pd.read_csv(suppl_meta_path)
cell_injury_df = suppl_meta_df[["Cellular injury category", "Compound alias"]]

print("Cell injury screen shape:", image_profile_df.shape)


# ## Labeling Cell Injury data

# Here, we are collecting all the samples treated solely with DMSO. Any well treated with DMSO will be labeled as "Control."

# In[3]:


# Get all wells treated with DMSO and label them as "Control" as the injury_type
control_df = image_profile_df.loc[image_profile_df["Compound Name"] == "DMSO"]
control_df.insert(0, "injury_type", "Control")

# display
print("Shape of the control:", control_df.shape)
control_df.head()


# Next, the `injured_df` is generated, which will exclusively contain wells treated with a component that induces an injury. This was accomplished by utilizing supplemental data that detailed which treatments caused specific injuries. We then cross-referenced this data with the image-based profile to identify wells treated with those components and labeled them with the associated injury.

# In[4]:


# creating a dictionary that contains the {injury_type : [list of treatments]}
injury_and_compounds = defaultdict(list)
for injury, compound in cell_injury_df.values.tolist():
    injury_and_compounds[injury].append(compound)

# cross reference injury and associated treatments into the screen image-based profile
injury_profiles = []
for injury_type, compound_list in injury_and_compounds.items():
    # selecting data frame with the treatments associated with the injury
    sel_profile = image_profile_df[
        image_profile_df["Compound Name"].isin(compound_list)
    ]

    # add a column to the data subset indicating what type of injury it is
    # and store it
    sel_profile.insert(0, "injury_type", injury_type)
    injury_profiles.append(sel_profile)

# concat the control and all injured labeled wells into a single data frame
injured_df = pd.concat(
    [
        control_df,
        pd.concat(injury_profiles).dropna(subset="injury_type").reset_index(drop=True),
    ]
)

# creating cell injury coder and encoder dictionary
cell_injuries = injured_df["injury_type"].unique()
injury_codes = defaultdict(lambda: {})
for idx, injury in enumerate(cell_injuries):
    injury_codes["encoder"][injury] = idx
    injury_codes["decoder"][idx] = injury


# update injured_df with injury codes
injured_df.insert(
    0,
    "injury_code",
    injured_df["injury_type"].apply(lambda injury: injury_codes["encoder"][injury]),
)

# split meta and feature column names
injury_meta, injury_feats = utils.split_meta_and_features(injured_df)

# save the injury codes json file
with open(fs_dir / "injury_codes.json", mode="w") as f:
    json.dump(injury_codes, f)

# display
print("Shape of cell injury dataframe", injured_df.shape)
print("Number of meta features", len(injury_meta))
print("Number of features", len(injury_feats))
print("Number of plates", len(injured_df["Plate"].unique()))
print("Number of injuries", len(injured_df["injury_type"].unique()))
print("Number of treatments", len(injured_df["Compound Name"].unique()))
print("List of Compounds", injured_df["Compound Name"].unique())
print("List of Injuries", injured_df["injury_type"].unique())
injured_df.head()


# After generating the complete cell injury dataframe, we will check for any rows containing NaN values and remove them if found.

# In[5]:


# next is to drop rows that NaNs
df = injured_df[injury_feats]
nan_idx_to_drop = df[df.isna().any(axis=1)].index

# display
print(f"shape of dataframe before drop NaN rows {injured_df.shape}")
print(f"There are {len(nan_idx_to_drop)} rows to drop that contains NaN's")

# update
injured_df = injured_df.drop(nan_idx_to_drop)
print(injured_df.shape)
injured_df.head()


# ## Feature Selection on the Cell-Injury Data
#
# Here, we will perform a feature selection using Pycytominer on the labeled cell-injury dataset to identify morphological features that are indicative of cellular damage. By selecting these key features, we aim to enhance our understanding of the biological mechanisms underlying cellular injuries. The selected features will be utilized to train a multi-class logistic regression model, allowing us to determine which morphological characteristics are most significant in discerning various types of cellular injuries.## Feature selecting on the cell-injury data

# In[6]:


# conduct feature selection using pycytominer
fs_cell_injury_profile = feature_select(
    profiles=injured_df,
    features=injury_feats,
    operation=[
        "correlation_threshold",
        "variance_threshold",
        "drop_outliers",
        "drop_na_columns",
    ],
)

# split meta and morphology feature columns
fs_cell_injury_meta, fs_cell_injury_feats = utils.split_meta_and_features(
    fs_cell_injury_profile
)

# display
print(f"N features cell-injury profile {len(injury_feats)}")
print(f"N features fs-cell-injury profile {len(fs_cell_injury_feats)}")
print(f"N features dropped {len(injury_feats) - len(fs_cell_injury_feats)}")

# if the feature space json file does not exists, create one and use this feature space for downstream
cell_injury_selected_feature_space_path = (
    fs_dir / "fs_cell_injury_only_feature_space.json"
).resolve()
if not cell_injury_selected_feature_space_path.exists():
    # saving morphology feature space in JSON file
    print("Feature space file does not exist, creating one...")
    fs_cell_injury_feature_space = {}
    fs_cell_injury_feature_space["name"] = "fs_cell_injury"
    fs_cell_injury_feature_space["n_plates"] = len(
        fs_cell_injury_profile["Plate"].unique()
    )
    fs_cell_injury_feature_space["n_meta_features"] = len(fs_cell_injury_meta)
    fs_cell_injury_feature_space["n_features"] = len(fs_cell_injury_feats)
    fs_cell_injury_feature_space["meta_features"] = fs_cell_injury_meta
    fs_cell_injury_feature_space["features"] = fs_cell_injury_feats
    with open(fs_dir / "fs_cell_injury_only_feature_space.json", mode="w") as stream:
        json.dump(fs_cell_injury_feature_space, stream)

# saving feature selected cell-injury profile
fs_cell_injury_profile.to_csv(fs_dir / "fs_cell_injury_only.csv.gz", index=False)

print(fs_cell_injury_profile.shape)
fs_cell_injury_profile.head()


# ## Identifying Shared Features between JUMP and Cell Injury Datasets
#
# In this section, we identify the shared features present in both the normalized cell-injury and the JUMP pilot dataset. Next, we utilize these shared features to update our dataset and use it for feature selection in the next step.

# In[7]:


# load in JUMP feature space
jump_feature_space = utils.load_json_file(jump_feature_space_path)
jump_feats = set(jump_feature_space["features"])

# find shared features and create data frame
shared_features = list(jump_feats.intersection(set(injury_feats)))
shared_features_df = pd.concat(
    [injured_df[injury_meta], injured_df[shared_features].fillna(0)], axis=1
)

# split meta and feature column
shared_meta, shared_feats = utils.split_meta_and_features(shared_features_df)

# display
print("Number of features in Cell Injury", len(injury_feats))
print("Number of features in JUMP", len(jump_feats))
print("Number of shared feats", len(shared_features))
print(
    "Number of features that are not overlapping",
    len(injury_feats) - len(shared_features),
)
print("N features in shared injured profile", len(shared_feats))
print("Shape of shared cell injury profile", shared_features_df.shape)
shared_features_df.head()


# ## Applying Feature Selection with Pycytominer
#
# In this section, we utilize Pycytominer's feature selection function to obtain features that will be used in training our machine learning models.

# In[8]:


# Applying feature selection using pycytominer
aligned_cell_injury_fs_df = feature_select(
    profiles=shared_features_df,
    features=shared_feats,
)

# split meta and feature column names
fs_injury_meta, fs_injury_feats = utils.split_meta_and_features(
    aligned_cell_injury_fs_df
)

# counting number of cell injuries
cell_injuries = aligned_cell_injury_fs_df["injury_type"].unique()

# display
print("Number of meta features", len(fs_injury_meta))
print("Number of features", len(fs_injury_feats))
print("Shape of fs shared profile", aligned_cell_injury_fs_df.shape)
print("number of cell injury types", len(cell_injuries))
print(cell_injuries)
print(aligned_cell_injury_fs_df.shape)
aligned_cell_injury_fs_df.head()

# save shared feature selected profile
aligned_cell_injury_fs_df.to_csv(
    fs_dir / "aligned_cell_injury_profile_fs.csv.gz",
    index=False,
    compression="gzip",
)


# Save the aligned feature space information while maintaining feature space order

# In[9]:


# split meta and feature column names
fs_injury_meta, fs_injury_feats = utils.split_meta_and_features(
    aligned_cell_injury_fs_df
)

# saving info of feature space
jump_feature_space = {
    "name": "cell_injury",
    "n_plates": len(aligned_cell_injury_fs_df["Plate"].unique()),
    "n_meta_features": len(fs_injury_meta),
    "n_features": len(fs_injury_feats),
    "meta_features": fs_injury_meta,
    "features": fs_injury_feats,
}

# if the feature space file does not exists, create one and use this feature space for downstream
selected_feature_space_path = (
    fs_dir / "aligned_cell_injury_shared_feature_space.json"
).resolve()
if not selected_feature_space_path.exists():
    print("Feature space file does not exist, creating one...")
    with open(selected_feature_space_path, mode="w") as f:
        json.dump(jump_feature_space, f)

# if it d oes exist then we have to check the selected features in this notebook matches with the one saved
loaded_selected_feature_space = utils.load_json_file(selected_feature_space_path)[
    "features"
]

# Check if all elements of list1 are in list2 and vice versa
all_in_list2 = all(item in fs_injury_feats for item in loaded_selected_feature_space)
all_in_list1 = all(item in loaded_selected_feature_space for item in fs_injury_feats)
assert all_in_list2 and all_in_list1, "The lists do not contain the same elements."
