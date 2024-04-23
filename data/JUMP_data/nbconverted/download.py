#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
import os
import pathlib
import sys

import pandas as pd
import requests

sys.path.append("../../")
from src.utils import split_meta_and_features

# In[2]:


# read
platemap_df = pd.read_csv("./barcode_platemap.csv")
platemap_df.head()


# In[3]:


# download normalized data
for plate_id in platemap_df["Assay_Plate_Barcode"]:
    url = f"https://cellpainting-gallery.s3.amazonaws.com/cpg0000-jump-pilot/source_4/workspace/profiles/2020_11_04_CPJUMP1/{plate_id}/{plate_id}_normalized_negcon.csv.gz"

    # request data
    with requests.get(url) as response:
        response.raise_for_status()
        save_path = pathlib.Path(f"./{plate_id}_normalized_negcon.csv.gz").resolve()

        # save content
        with open(save_path, mode="wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


# In[4]:


# after downloading all dataset, concat into a single dataframe
data_files = list(pathlib.Path.cwd().glob("*.csv.gz"))

# create main df by concatenating all file
main_df = pd.concat([pd.read_csv(file) for file in data_files])

# remove single_dfs
[os.remove(file) for file in data_files]

# save concatenated df into ./data/JUMP_data folders
main_df.to_csv(
    "JUMP_all_plates_normalized_negcon.csv.gz", index=False, compression="gzip"
)


# In[5]:


# saving feature space
jump_meta, jump_feat = split_meta_and_features(main_df, metadata_tag=True)

# saving info of feature space
jump_feature_space = {
    "name": "JUMP",
    "n_plates": len(main_df["Metadata_Plate"].unique()),
    "n_meta_features": len(jump_meta),
    "n_features": len(jump_feat),
    "meta_features": jump_meta,
    "features": jump_feat,
}

# save json file
with open("jump_feature_space.json", mode="w") as f:
    json.dump(jump_feature_space, f)

# display
print("NUmber of plates", len(main_df["Metadata_Plate"].unique()))
print("Number of meta features", len(jump_meta))
print("Number of features", len(jump_feat))
