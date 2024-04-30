#!/usr/bin/env python
# coding: utf-8

# # Downloading JUMP Pilot Dataset
#
# This notebook focuses on downloading the JUMP-CellPainting dataset. The pilot dataset comprises aggregate profiles at the well level, spanning 51 plates. These profiles have been normalized using the negative controls within each plate. We downloaded all 51 negative-controlled normalized aggregate profiles and concatenating them into a single dataset file. The JUMP dataset profile will be saved in the `./data/JUMP_data` directory.

# In[1]:


import json
import sys

import pandas as pd

sys.path.append("../../")
from src.utils import split_meta_and_features

# Reading the plate map to get all the Plate ID's

# In[2]:


# loading plate map
platemap_df = pd.read_csv("./barcode_platemap.csv")
platemap_df.head()


# Next, we use the plate IDs to the URL in order to download the aggregated profiles. We use pandas to download and load each profile, and then concatenate them into a single dataframe. The merged dataframe serves as our main JUMP dataset.

# In[3]:


# download all normalized aggregated profiles
jump_df = []
for plate_id in platemap_df["Assay_Plate_Barcode"]:
    url = f"https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/{plate_id}/{plate_id}_normalized_negcon.csv.gz"
    df = pd.read_csv(url)
    jump_df.append(df)

# concat all downloaded concatenate all aggregate profiles
jump_df = pd.concat(jump_df)

# save concatenated df into ./data/JUMP_data folders
jump_df.to_csv(
    "JUMP_all_plates_normalized_negcon.csv.gz", index=False, compression="gzip"
)


# Here, we obtain information about the feature space by splitting both the meta and feature column names and storing them in a dictionary.
# This dictionary holds information about the feature space and will be utilized for downstream analysis when identifying shared features across different datasets, such as the Cell-injury dataset.

# In[4]:


# saving feature space
jump_meta, jump_feat = split_meta_and_features(jump_df, metadata_tag=True)

# saving info of feature space
jump_feature_space = {
    "name": "JUMP",
    "n_plates": len(jump_df["Metadata_Plate"].unique()),
    "n_meta_features": len(jump_meta),
    "n_features": len(jump_feat),
    "meta_features": jump_meta,
    "features": jump_feat,
}

# save json file
with open("jump_feature_space.json", mode="w") as f:
    json.dump(jump_feature_space, f)

# display
print("Shape of Merged dataset", jump_df.shape)
print("NUmber of plates", len(jump_df["Metadata_Plate"].unique()))
print("Number of meta features", len(jump_meta))
print("Number of features", len(jump_feat))
