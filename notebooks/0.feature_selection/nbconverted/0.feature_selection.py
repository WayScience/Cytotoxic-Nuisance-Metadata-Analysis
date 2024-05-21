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

# split meta and feature column names
injury_meta, injury_feats = utils.split_meta_and_features(injured_df)

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


# ## Identifying Shared Features between JUMP and Cell Injury Datasets
#
# In this section, we identify the shared features present in both the normalized cell-injury and the JUMP pilot dataset. Next, we utilize these shared features to update our dataset and use it for feature selection in the next step.

# In[5]:


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
# In this section, we utilize Pycytominer's feature selection function to obtain informative features that will be employed in training our machine learning models.

# In[6]:


# Applying feature selection using pycytominer
fs_injury_df = feature_select(
    profiles=shared_features_df,
    features=shared_feats,
)

# split meta and feature column names
fs_injury_meta, fs_injury_feats = utils.split_meta_and_features(fs_injury_df)
cell_injuries = fs_injury_df["injury_type"].unique()

# display
print("Number of meta features", len(fs_injury_meta))
print("Number of features", len(fs_injury_feats))
print("Shape of fs shared profile", fs_injury_df.shape)
print("number of cell injury types", len(cell_injuries))
print(cell_injuries)
fs_injury_df.head()


# Generate encoders and decoders for injuries, and save the file as a JSON file. Additionally, update the feature-selected profile with the injury codes and save it.

# In[7]:


# next lets make an injury code
injury_codes = defaultdict(lambda: {})
for idx, injury in enumerate(cell_injuries):
    injury_codes["encoder"][injury] = idx
    injury_codes["decoder"][idx] = injury

# update shared fs profile with injury codes
fs_injury_df.insert(
    0,
    "injury_code",
    fs_injury_df["injury_type"].apply(lambda injury: injury_codes["encoder"][injury]),
)

# now save the injury codes json file
with open(fs_dir / "injury_codes.json", mode="w") as f:
    json.dump(injury_codes, f)

# save shared feature selected profile
fs_injury_df.to_csv(
    fs_dir / "cell_injury_profile_fs.csv.gz",
    index=False,
    compression="gzip",
)


# Save feature space information while maintaining feature space order

# In[ ]:


# split meta and feature column names
fs_injury_meta, fs_injury_feats = utils.split_meta_and_features(fs_injury_df)

# saving info of feature space
jump_feature_space = {
    "name": "cell_injury",
    "n_plates": len(fs_injury_df["Plate"].unique()),
    "n_meta_features": len(fs_injury_meta),
    "n_features": len(fs_injury_feats),
    "meta_features": fs_injury_meta,
    "features": fs_injury_feats,
}

# save json file
with open(fs_dir / "cell_injury_shared_feature_space.json", mode="w") as f:
    json.dump(jump_feature_space, f)
