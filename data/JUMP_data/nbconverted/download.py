#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import pathlib

import pandas as pd
import requests

# In[6]:


# read
platemap_df = pd.read_csv("./barcode_platemap.csv")
platemap_df.head()


# In[14]:


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


# In[15]:


# after downloading all dataset, concat into a single dataframe
data_files = list(pathlib.Path.cwd().glob("*.csv.gz"))

# create main df
main_df = pd.concat([pd.read_csv(file) for file in data_files])

# remove single_dfs
[os.remove(file) for file in data_files]

# save concatenated df
main_df.to_csv("all_plates_normalized_negcon.csv.gz", index=False, compression="gzip")
