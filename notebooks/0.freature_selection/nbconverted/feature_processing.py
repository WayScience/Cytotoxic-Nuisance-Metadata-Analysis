#!/usr/bin/env python
# coding: utf-8

# # Feature Processing and Selection
# This notebook focuses on exploration using two essential files: the annotations data extracted from the actual screening profile (available in the [IDR repository](https://github.com/IDR/idr0133-dahlin-cellpainting/tree/main/screenA)) and the metadata retrieved from the supplementary section of the [research paper](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36829-x/MediaObjects/41467_2023_36829_MOESM5_ESM.xlsx).
#
# We explore the number of unique compounds associated with each cell injury and subsequently cross-reference this information with the screening profile. The aim is to assess the feasibility of using the data for training a machine learning model to predict cell injury.
#
# We apply feature selection through [pycytominer](https://github.com/cytomining/pycytominer) to capture the most informative features representing various cellular injury types within the morphology space. Then, we utilize the selected feature profiles for machine learning applications.
#

# In[1]:


import pathlib
import sys
from collections import defaultdict

import pandas as pd
from pycytominer import feature_select

sys.path.append("../../")
from src import utils

# In[2]:


# data directory
data_dir = pathlib.Path("../../data").resolve(strict=True)
results_dir = pathlib.Path("../../results").resolve(strict=True)
fs_dir = (results_dir / "0.feature_selection").resolve()
fs_dir.mkdir(exist_ok=True)

# data paths
suppl_meta_path = (data_dir / "41467_2023_36829_MOESM5_ESM.csv.gz").resolve(strict=True)
screen_anno_path = (data_dir / "idr0133-screenA-annotation.csv.gz").resolve(strict=True)

# load data
image_profile_df = pd.read_csv(screen_anno_path)
meta_df = image_profile_df[image_profile_df.columns[:31]]
compounds_df = meta_df[["Compound Name", "Compound Class"]]

suppl_meta_df = pd.read_csv(suppl_meta_path)
cell_injury_df = suppl_meta_df[["Cellular injury category", "Compound alias"]]


# In[3]:


# get the control
control_df = image_profile_df.loc[image_profile_df["Compound Name"] == "DMSO"]
control_df.insert(0, "injury_type", "Control")

# display
print("Shape of the control:", control_df.shape)
control_df.head()


# In[4]:


# getting profiles based on injury and compound type
injury_and_compounds = defaultdict(list)
for injury, compound in cell_injury_df.values.tolist():
    injury_and_compounds[injury].append(compound)

# cross reference selected injury and associated components into the screen profile
injury_profiles = []
for injury_type, compound_list in injury_and_compounds.items():
    sel_profile = image_profile_df[
        image_profile_df["Compound Name"].isin(compound_list)
    ]
    sel_profile.insert(0, "injury_type", injury_type)
    injury_profiles.append(sel_profile)


# In[5]:


# creating a dataframe that contains stratified screen Data
injured_df = pd.concat(injury_profiles)

# drop wells that do not have an injury
injured_df = injured_df.dropna(subset="injury_type").reset_index(drop=True)
print("Number of wells", len(injured_df["Plate"].unique()))

# display df
print("shape:", injured_df.shape)
injured_df.head()


# In[6]:


# seperating meta and feature columns
meta = injured_df.columns.tolist()[:32]
features = injured_df.columns.tolist()[32:]


# In[7]:


# dropping samples that have at least 1 NaN
injured_df = utils.drop_na_samples(profile=injured_df, features=features, cut_off=0)

# display
print("Shape after removing samples: ", injured_df.shape)
injured_df.head()


# In[8]:


# setting feature selection operations
all_operations = [
    "variance_threshold",
    "correlation_threshold",
    "drop_na_columns",
    "blocklist",
    "drop_outliers",
]

# Applying feature selection using pycytominer
fs_injury_df = feature_select(
    profiles=injured_df,
    features=features,
    operation=all_operations,
    freq_cut=0.05,
    corr_method="pearson",
    corr_threshold=0.90,
    na_cutoff=0.0,
    outlier_cutoff=100,
)


# In[9]:


print("Feature selected profile shape:", fs_injury_df.shape)
fs_injury_df.head()


# In[10]:


# update the control with the retained features in the injury_fs_profile
control_df = control_df[fs_injury_df.columns]

# display
print(
    "Shape of control after using feature retained from injury_fs profile",
    control_df.shape,
)
control_df.head()


# In[11]:


# concat both the injury and control together and make this is that feature selected profile
fs_profile = pd.concat([control_df, fs_injury_df])

# save and display
fs_profile.to_csv(
    fs_dir / "cell_injury_profile_fs.csv.gz",
    index=False,
    compression="gzip",
)


# In[12]:


# setting which injr
cell_injuries = fs_profile["injury_type"].unique()
print("number of cell injury types", len(cell_injuries))
cell_injuries
