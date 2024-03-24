#!/usr/bin/env python
# coding: utf-8

# In[9]:


import json
import pathlib
import sys

import joblib
import pandas as pd
from pycytominer.cyto_utils.features import infer_cp_features

# project module imports
sys.path.append("../../")  # noqa
from src.utils import check_feature_order  # noqa

# ## Setting up file paths and parameters

# In[10]:


# setting up paths
results_dir = pathlib.Path("../../results")
data_split_dir = (results_dir / "1.data_splits/").resolve(strict=True)
jump_data_dir = pathlib.Path("../../data/JUMP_data").resolve(strict=True)
modeling_dir = pathlib.Path("../../results/2.modeling").resolve(strict=True)


# data files
jump_data_path = (jump_data_dir / "JUMP_all_plates_normalized_negcon.csv.gz").resolve(
    strict=True
)
multi_class_model_path = (modeling_dir / "multi_class_model.joblib").resolve(
    strict=True
)
feature_col_names = (data_split_dir / "feature_cols.json").resolve(strict=True)


# output paths
jump_analysis_dir = (results_dir / "3.jump_analysis").resolve()
jump_analysis_dir.mkdir(exist_ok=True)


# ## loading files

# In[11]:


# loading in the negatlive controled normalized profiles
jump_df = pd.read_csv(jump_data_path)

# loading json file containing selected feature names
with open(feature_col_names, mode="r") as infile:
    cell_injury_cp_feature_cols = json.load(infile)

# loading json file that contains the coder and decoder injury labels
with open(data_split_dir / "injury_codes.json") as infile:
    injury_codes = json.load(infile)

# display dataframe and size
print("JUMP dataset size:", jump_df.shape)
jump_df.head()


# ## Feature alignment
# In this section, we are identifying the shared features present in both the cell injury and JUMP datasets.
# Once these features are identified, we update the JUMP dataset to include only those features that are shared between both profiles for our machine learning application.

# First we identify the CellProfiler (CP) features present in the JUMP data.
# We accomplish this by utilizing `pycytominer`'s  'infer_cp_features()', which helps us identify CP features in the JUMP dataset.

# In[12]:


# get compartments
metadata_prefix = "Metadata_"
compartments = list(
    set(
        [
            feature_name.split("_")[0]
            for feature_name in jump_df.columns.tolist()
            if not feature_name.startswith("Metadata_")
        ]
    )
)

# find CP features in JUMP dataset
jump_cp_features = infer_cp_features(jump_df, compartments=compartments)
meta_features = infer_cp_features(jump_df, compartments=compartments, metadata=True)

# display number of features of both profiles
print("Number of Metadata Features:", len(meta_features))
print(
    "Number of CP features that cell injury has",
    len(cell_injury_cp_feature_cols["feature_cols"]),
)
print("Number of CP features that JUMP has:", len(jump_cp_features))


# Now that we have identified the features present in both datasets, the next step is to align them. This involves identifying the common features between both profiles and utilizing these features to update our JUMP dataset for our machine learning model.

# In[13]:


cell_injury_cp_features = cell_injury_cp_feature_cols["feature_cols"]

# finding shared featues using intersection
aligned_features = list(set(cell_injury_cp_features) & set(jump_cp_features))

# displaying number of shared features between both profiles
print("Number of shapred features of both profiles", len(aligned_features))


# The objective of this step is to preserve the order of the feature space.
#
# Since we have identified the shared feature space across both profiles, we still need to address those that are missing.
# Therefore, to maintain the feature space order, we opted to use the cell injury feature space as our reference feature space order, as our multi-class model was trained to understand this specific order.
#
# Next, we addressed features that were not found within the JUMP dataset.
# This was done by including them in the alignment process, but defaulted their values to 0.
#
# Ultimately, we generated a new profile called `aligned_jump_df`, which contains the correctly aligned and ordered feature space from the cell injury dataset.

# In[14]:


# multiplier is the number of samples in JUMP data
# this is used to default non-aligned features to 0
multiplier = jump_df.shape[0]

# storing feature and values in order
aligned_jump = {}
for injury_feat in cell_injury_cp_features:
    if injury_feat not in aligned_features:
        aligned_jump[injury_feat] = [0.0] * multiplier
    else:
        aligned_jump[injury_feat] = jump_df[injury_feat].values.tolist()

# creating dataframe with the aligned features and retained feature order
aligned_jump_feats_df = pd.DataFrame.from_dict(aligned_jump, orient="columns")

# sanity check: see if the feature order in the `cell_injury_cp_feature_cols` is the same with
# the newly generated aligned JUMP dataset
assert (
    cell_injury_cp_features == aligned_jump_feats_df.columns.tolist()
), "feature space are not aligned"
assert check_feature_order(
    ref_feat_order=cell_injury_cp_features,
    input_feat_order=aligned_jump_feats_df.columns.tolist(),
), "feature space do not follow the same order"


# ## Applying to our Multi-Class trained model
#
# We applying the aligned JUMP dataset to our trained multi-class model and measure the probabiltiies of which cell injury each well possessed.

# In[15]:


# loading in mutliclass model
multi_class_cell_injury_model = joblib.load(multi_class_model_path)


# In[16]:


# apply
pred_proba = multi_class_cell_injury_model.predict_proba(aligned_jump_feats_df)

# convert prediction probabilities to a pandas daraframe
pred_proba_df = pd.DataFrame(pred_proba)

# update the column names with the name of the injury class
pred_proba_df.columns = [
    injury_codes["decoder"][str(colname)] for colname in pred_proba_df.columns.tolist()
]

# augment the porbability scores by meringing with the metadata associated with each well
pred_proba_df = jump_df[meta_features].merge(
    pred_proba_df, left_index=True, right_index=True
)

# # save the probability dataframe
pred_proba_df.to_csv(jump_analysis_dir / "JUMP_injury_proba.csv.gz", compression="gzip")

# # display shape and size
print("Probability shape:", pred_proba_df.shape)
pred_proba_df.head()
