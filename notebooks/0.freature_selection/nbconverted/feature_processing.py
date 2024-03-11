#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pathlib
import sys
from collections import defaultdict

import pandas as pd
from pycytominer import feature_select

sys.path.append("../../")
from src import utils

# In[9]:


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


# In[10]:


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


# In[11]:


# creating a dataframe that contains stratified screen Data
injured_df = pd.concat(injury_profiles)

# drop wells that do not have an injury
injured_df = injured_df.dropna(subset="injury_type").reset_index(drop=True)
print("Number of wells", len(injured_df["Plate"].unique()))

# display df
print("shape:", injured_df.shape)
injured_df.head()


# In[12]:


# seperating meta and feature columns
meta = injured_df.columns.tolist()[:32]
features = injured_df.columns.tolist()[32:]


# In[13]:


# dropping samples that have at least 1 NaN
injured_df = utils.drop_na_samples(profile=injured_df, features=features, cut_off=0)

# display
print("Shape after removing samples: ", injured_df.shape)
injured_df.head()


# In[15]:


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

# saving dataframe
fs_injury_df.to_csv(
    fs_dir / "cell_injury_profile_fs.csv.gz",
    index=False,
    compression="gzip",
)


# In[16]:


print("Feature selected profile shape:", fs_injury_df.shape)
fs_injury_df.head()


# In[20]:


# setting which injr
cell_injuries = fs_injury_df["injury_type"].unique()
print("number of cell injury types", len(cell_injuries))
cell_injuries
