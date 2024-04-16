#!/usr/bin/env python
# coding: utf-8

# # Applying JUMP dataset to pre-trained multi-class logistic regression model

# In[1]:


import json
import pathlib
import sys

import joblib
import pandas as pd

# project module imports
sys.path.append("../../")  # noqa
from src.utils import (  # noqa
    check_feature_order,
    generate_confusion_matrix_tl,
    split_meta_and_features,
)

# ## Setting up file paths and parameters

# In[2]:


# setting up paths
results_dir = pathlib.Path("../../results")
data_split_dir = (results_dir / "1.data_splits/").resolve(strict=True)
jump_data_dir = pathlib.Path("../../data/JUMP_data").resolve(strict=True)
modeling_dir = pathlib.Path("../../results/2.modeling").resolve(strict=True)

# JUMP data files
jump_data_path = (jump_data_dir / "JUMP_all_plates_normalized_negcon.csv.gz").resolve(
    strict=True
)

# After holdout metadata
cell_injury_metadata_path = (
    data_split_dir / "cell_injury_metadata_after_holdout.csv.gz"
).resolve(strict=True)

# model paths
multi_class_model_path = (modeling_dir / "multi_class_model.joblib").resolve(
    strict=True
)
shuffled_multi_class_model_path = (
    modeling_dir / "shuffled_multi_class_model.joblib"
).resolve(strict=True)

# feature columns (from feature selected profile)
feature_col_names = (data_split_dir / "feature_cols.json").resolve(strict=True)
injury_codes_path = (data_split_dir / "injury_codes.json").resolve(strict=True)

# output paths
jump_analysis_dir = (results_dir / "3.jump_analysis").resolve()
jump_analysis_dir.mkdir(exist_ok=True)


# ## Loading Files

# In[3]:


# loading in the negatlive controled normalized profiles
jump_df = pd.read_csv(jump_data_path)
cell_injury_meta_df = pd.read_csv(cell_injury_metadata_path)

# loading json file containing selected feature names
with open(feature_col_names, mode="r") as infile:
    cell_injury_cp_feature_cols = json.load(infile)

# loading json file that contains the coder and decoder injury labels
with open(injury_codes_path) as infile:
    injury_codes = json.load(infile)

injury_decoder = injury_codes["decoder"]
injury_encoder = injury_codes["encoder"]

# display dataframe and size
print("JUMP dataset size:", jump_df.shape)
jump_df.head()


# ## Finding overlapping Compounds
#
# This notebook aims to identify overlapping compounds present in both the `cell_injury` and `JUMP` datasets. These overlapping compounds will be used for subsetting the `JUMP` dataset, which we'll consider as the ground truth for subsequent analyses.
#
# ## Approach
# 1. **Identifying Overlapping Compounds**: We compare the compounds present in both datasets to identify the overlapping ones.
# 2. **Subsetting the JUMP Dataset**: Once the overlapping compounds are identified, we subset the `JUMP` dataset to include only those compounds, forming our ground truth dataset.
# 3. **Save dataset**: The dataset will be saved in the `./results/3.jump_analysis`

# ### Step 1: Identifying Overlapping Compounds
# Here, we used the International Chemical Identifier (InChI) to identify chemicals shared between the JUMP dataset and the Cell Injury dataset.

# In[4]:


# get all InChI keys
cell_injury_InChI_keys = cell_injury_meta_df["Compound InChIKey"].tolist()
jump_InChI_keys = jump_df["Metadata_InChIKey"].tolist()

# identify common InChI Keys
common_compounds_inchikey = list(
    set(cell_injury_InChI_keys).intersection(jump_InChI_keys)
)

# identify the compounds
overlapping_compounds_df = cell_injury_meta_df.loc[
    cell_injury_meta_df["Compound InChIKey"].isin(common_compounds_inchikey)
]

# inserting injury code
overlapping_compounds_df.insert(
    0,
    "injury_code",
    overlapping_compounds_df["injury_type"].apply(lambda name: injury_encoder[name]),
)
unique_compound_names = overlapping_compounds_df["Compound Name"].unique().tolist()
print("Identified overlapping compounds:", ", ".join(unique_compound_names))


# now create a dataframe where it contains
overlapping_compounds_df = (
    overlapping_compounds_df[
        ["injury_code", "injury_type", "Compound Name", "Compound InChIKey"]
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)
overlapping_compounds_df


# Once the common compounds and their associated cell injury types are identified, the next step involves selecintg it from the JUMP dataset to select only wells that possess the common InChI keys.

# In[5]:


overlapping_jump_df = jump_df.loc[
    jump_df["Metadata_InChIKey"].isin(common_compounds_inchikey)
]

# agument filtered JUMP data with labels
overlapping_jump_df = pd.merge(
    overlapping_jump_df,
    overlapping_compounds_df,
    left_on="Metadata_InChIKey",
    right_on="Compound InChIKey",
)


print("shape: ", overlapping_jump_df.shape)
overlapping_jump_df.head()


# Now that we have identified the wells treated with overlapping treatments, we want to know the number of wells that a specific treatment have.

# In[6]:


# count number of wells and agument with injury_code injury_yype and compound name
well_counts_df = (
    overlapping_jump_df.groupby("Metadata_InChIKey")
    # counting the numbver of wells
    .size()
    .to_frame()
    .reset_index()
    # merge based on InChIKey
    .merge(
        overlapping_compounds_df,
        left_on="Metadata_InChIKey",
        right_on="Compound InChIKey",
    )
    # remove duplicate InChIKey Column
    .drop(columns=["Compound InChIKey"])
)

# update columns
well_counts_df.columns = [
    "Metadata_InChIKey",
    "n_wells",
    "injury_code",
    "injury_type",
    "compund_name",
]
well_counts_df


# Next, we wanted to examine the distribution of treatments across plates.

# In[7]:


# now lets look at the amount of wells have treatments and controls per plate
n_well_treatments = {}
for plate, df in overlapping_jump_df.groupby("Metadata_Plate"):
    treatment_counts = {}
    for treatment, df2 in df.groupby("Metadata_InChIKey"):
        counts = df2.shape[0]
        treatment_counts[df2["Compound Name"].unique()[0]] = counts

    n_well_treatments[plate] = treatment_counts

# looking treatment distribution across each plate
plate_treatments = (
    pd.DataFrame.from_dict(n_well_treatments, orient="columns")
    .T[["DMSO", "Colchicine", "Menadione", "Cycloheximide"]]
    .fillna(0)
    .astype(int)
    .reset_index()
)
plate_treatments.columns = [
    "plate_id",
    "DMSO",
    "Colchicine",
    "Menadione",
    "Cycloheximide",
]

# display
print(
    "Number of Plates that contain overlapping treatments:", plate_treatments.shape[0]
)
plate_treatments


# Finally we save the overlapping_treaments_df as a csv.gz file.

# In[8]:


# save overlapping files
overlapping_jump_df.to_csv(
    modeling_dir / "overlapping_treatments_jump_data.csv.gz",
    compression="gzip",
    index=False,
)


# ## Feature alignment
#
# In this section, we are identifying the shared features present in both the cell injury and JUMP datasets.
# Once these features are identified, we update the JUMP dataset to include only those features that are shared between both profiles for our machine learning application

# First we identify the CellProfiler (CP) features present in the JUMP data.
# We accomplish this by utilizing `pycytominer`'s  `infer_cp_features()`, which helps us identify CP features in the JUMP dataset.

# In[9]:


# get compartments
metadata_prefix = "Metadata_"

# split metadata and feature column names
jump_meta_cols, jump_feat_cols = split_meta_and_features(jump_df, metadata_tag=True)

# display number of features of both profiles
print("Number of Metadata Features:", len(jump_meta_cols))
print(
    "Number of CP features that cell injury has",
    len(cell_injury_cp_feature_cols["feature_cols"]),
)
print("Number of CP features that JUMP has:", len(jump_feat_cols))


# Now that we have identified the features present in both datasets, the next step is to align them. This involves identifying the common features between both profiles and utilizing these features to update our JUMP dataset for our machine learning model.

# In[10]:


cell_injury_cp_features = cell_injury_cp_feature_cols["feature_cols"]

# finding shared features using intersection
aligned_features = list(set(cell_injury_cp_features) & set(jump_feat_cols))

# displaying the number of shared features between both profiles
print("Number of shared features of both profiles", len(aligned_features))


# The objective of this step is to preserve the order of the feature space.
#
# Since we have identified the shared feature space across both profiles, we still need to address those that are missing.
# Therefore, to maintain the feature space order, we used the the cell injury feature space as our reference feature space order, since our multi-class model was trained to understand this specific order.
#
# Next, we addressed features that were not found within the JUMP dataset.
# This was done by including them in the alignment process, but defaulted their values to 0.
#
# Ultimately, we generated a new profile called `aligned_jump_df`, which contains the correctly aligned and ordered feature space from the cell injury dataset.

# In[11]:


# multiplier is the number of samples in JUMP data in order to maintaing data shape
multiplier = jump_df.shape[0]

# storing feature and values in order
aligned_jump = {}
for injury_feat in cell_injury_cp_features:
    if injury_feat not in aligned_features:
        aligned_jump[injury_feat] = [0.0] * multiplier
    else:
        aligned_jump[injury_feat] = jump_df[injury_feat].values.tolist()

# creating dataframe with the aligned features and retained feature order
aligned_jump_df = pd.DataFrame.from_dict(aligned_jump, orient="columns")

# sanity check: see if the feature order in the `cell_injury_cp_feature_cols` is the same with
# the newly generated aligned JUMP dataset
assert (
    cell_injury_cp_features == aligned_jump_df.columns.tolist()
), "feature space are not aligned"
assert check_feature_order(
    ref_feat_order=cell_injury_cp_features,
    input_feat_order=aligned_jump_df.columns.tolist(),
), "feature space do not follow the same order"


# In[12]:


# augment aligned jump with the metadata and save it
aligned_jump_df = jump_df[jump_meta_cols].merge(
    aligned_jump_df, left_index=True, right_index=True
)

# display
print("shape of aligned dataset", aligned_jump_df.shape)
aligned_jump_df.head()


# ## Applying JUMP dataset to Multi-Class Logistics Regression Model

# ### Applying to Complete JUMP dataset

# In[13]:


# split the data
aligned_meta_cols, aligned_feature_cols = split_meta_and_features(aligned_jump_df)
X = aligned_jump_df[aligned_feature_cols]


# In[14]:


# Loading in model
model = joblib.load(modeling_dir / "multi_class_model.joblib")
shuffled_model = joblib.load(modeling_dir / "shuffled_multi_class_model.joblib")


# Here, we apply the JUMP dataset to the model to calculate the probabilities of each injury being present per well. These probabilities are then saved in a tidy long format suitable for plotting in R.

# In[15]:


# get all injury classes
injury_classes = [injury_decoder[str(code)] for code in model.classes_.tolist()]

# prediction probabilities on both non-shuffled and shuffled models
y_proba = model.predict_proba(X)
shuffled_y_proba = shuffled_model.predict_proba(X)

# convert to pandas dataframe
y_proba_df = pd.DataFrame(y_proba)
shuffled_y_proba_df = pd.DataFrame(shuffled_y_proba)

# update column names with injury type names
y_proba_df.columns = [
    injury_codes["decoder"][str(colname)] for colname in y_proba_df.columns.tolist()
]

shuffled_y_proba_df.columns = [
    injury_codes["decoder"][str(colname)]
    for colname in shuffled_y_proba_df.columns.tolist()
]

# adding column if labels indicating if the prediction was done with a shuffled model
y_proba_df.insert(0, "shuffled_model", False)
shuffled_y_proba_df.insert(0, "shuffled_model", True)

# merge InChIKey based on index, since order is retained
# jump_df[aligned_meta_cols].merge(y_proba_df)
y_proba_df = pd.merge(
    jump_df[aligned_meta_cols]["Metadata_InChIKey"].to_frame(),
    y_proba_df,
    left_index=True,
    right_index=True,
)
shuffled_y_proba_df = pd.merge(
    jump_df[aligned_meta_cols]["Metadata_InChIKey"].to_frame(),
    shuffled_y_proba_df,
    left_index=True,
    right_index=True,
)

# concat all probabilities into one dataframe
all_probas_df = pd.concat([y_proba_df, shuffled_y_proba_df]).reset_index(drop=True)

# Add a column to indicate the most probable injury
# This is achieved by selecting the injury with the highest probability
all_probas_df.insert(
    2,
    "pred_injury",
    all_probas_df[injury_classes].apply(lambda row: row.idxmax(), axis=1),
)

# next is to convert the probabilities dataframe into tidy long
all_probas_df_tl = pd.melt(
    all_probas_df,
    id_vars=["Metadata_InChIKey", "shuffled_model", "pred_injury"],
    value_vars=injury_classes,
    var_name="injury_type",
    value_name="proba",
)

# save probabilities in tidy long format
all_probas_df_tl.to_csv(jump_analysis_dir / "JUMP_injury_proba.csv.gz", index=False)

print("tidy long format probability shape", all_probas_df_tl.shape)


# Now that the Metadata_InChIKey metadata has been added to the probabilities dataframe, we can filter out the overlapping treatments based on their InChIKeys.

# In[16]:


# display overlapping compounds
overlapping_compounds_df


# In[17]:


overlapping_compounds_probas_df = all_probas_df.loc[
    all_probas_df["Metadata_InChIKey"].isin(
        overlapping_compounds_df["Compound InChIKey"]
    )
]
overlapping_compounds_probas_df = overlapping_compounds_df.merge(
    overlapping_compounds_probas_df,
    how="inner",
    left_on="Compound InChIKey",
    right_on="Metadata_InChIKey",
)
overlapping_compounds_probas_df


# In[26]:


overlapping_compounds_probas_df.loc[overlapping_compounds_probas_df["shuffled_model"]]


# ### Confusion Matrix with Overlapping Treatments

# In[19]:


overlapp_df = aligned_jump_df.loc[
    aligned_jump_df["Metadata_InChIKey"].isin(
        overlapping_compounds_df["Compound InChIKey"]
    )
]

# separate metadata and feature columns
overlapp_meta, overlapp_feats = split_meta_and_features(overlapp_df)

overlapp_df = overlapping_compounds_df.merge(
    overlapp_df, how="inner", left_on="Compound InChIKey", right_on="Metadata_InChIKey"
)
overlapp_df.head()


# In[20]:


# splitting data
X = overlapp_df[overlapp_feats]
y = overlapp_df["injury_code"]


# In[21]:


# generated a confusion matrix in tidy long format
jump_overlap_cm = generate_confusion_matrix_tl(
    model, X, y, shuffled=False, dataset_type="JUMP Overlap"
).fillna(0)
shuffled_jump_overlap_cm = generate_confusion_matrix_tl(
    shuffled_model, X, y, shuffled=True, dataset_type="JUMP Overlap"
).fillna(0)


# In[22]:


# save confusion matrix
pd.concat([jump_overlap_cm, shuffled_jump_overlap_cm]).to_csv(
    modeling_dir / "jump_overlap_confusion_matrix.csv.gz",
    compression="gzip",
    index=False,
)
