#!/usr/bin/env python
# coding: utf-8

# # Module 3: JUMP Analysis
#
# In this notebook, we utilize the Joint Undertaking in Morphological Profile [dataset](https://jump-cellpainting.broadinstitute.org/cell-painting) and integrate it into our model. Our objective is to assess the probability of specific cell injuries present within each well entry from the JUMP dataset.
#
# Additionally, we identify shared treatments between the JUMP and cell-injury datasets to construct a confusion matrix. This enables us to evaluate the performance of predicting cellular injury across different datasets.

# In[1]:


import pathlib
import sys

import joblib
import pandas as pd

# project module imports
sys.path.append("../../")  # noqa
from src.utils import (
    check_feature_order,
    generate_confusion_matrix_tl,
    load_json_file,
    split_meta_and_features,
)

# ## Setting up parameters and paths

# In[2]:


# setting up paths and output paths
results_dir = pathlib.Path("../../results")
fs_results_dir = (results_dir / "0.feature_selection").resolve(strict=True)
data_split_dir = (results_dir / "1.data_splits/").resolve(strict=True)
jump_data_dir = pathlib.Path("../../data/JUMP_data").resolve(strict=True)
modeling_dir = pathlib.Path("../../results/2.modeling").resolve(strict=True)

# JUMP data files
jump_data_path = (jump_data_dir / "JUMP_all_plates_normalized_negcon.csv.gz").resolve(
    strict=True
)

# loading only cell injury metadata (after holdout has been applied)
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

# overlapping feature space path
shared_feature_space_path = (
    fs_results_dir / "cell_injury_shared_feature_space.json"
).resolve(strict=True)

# injury codes
injury_codes_path = (fs_results_dir / "injury_codes.json").resolve(strict=True)

# output paths
jump_analysis_dir = (results_dir / "3.jump_analysis").resolve()
jump_analysis_dir.mkdir(exist_ok=True)


# ## Loading in datasets and json files
#
# Here we are loading the JUMP dataset along with the cell injury metadata, injury codes and the files representing the overlapping feature space.

# In[3]:


# loading in JUMP dataset
jump_df = pd.read_csv(jump_data_path)

# loading in cell injury metadata only (after holdout)
cell_injury_meta_df = pd.read_csv(cell_injury_metadata_path)

# split metadata and feature columns
jump_meta, jump_feats = split_meta_and_features(jump_df, metadata_tag=True)

# loading json file that contains the shared feature
injury_codes = load_json_file(injury_codes_path)
injury_encoder = injury_codes["encoder"]
injury_decoder = injury_codes["decoder"]

# loading in shared feature space
shared_feature_space = load_json_file(shared_feature_space_path)
shared_features = shared_feature_space["features"]

# Display data
print("JUMP dataset shape", jump_df.shape)
print("Number of Meta features", len(jump_meta))
print("Number of JUMP features", len(jump_feats))
print("Number of shared features between JUMP and Cell Injury", len(shared_features))
jump_df.head()


# ## Updating the JUMP Dataset by Selecting Only Shared Features
#
# During this step, we utilize the shared feature list to update our JUMP dataset, selecting only those features that overlap.
#
# Note that the shared feature space file maintains the same order as the feature space used during model training.

# In[4]:


# update the over lapping jump df
# Augment the overlapping feature space with the metadata
shared_jump_df = jump_df[shared_features]
shared_jump_df = pd.concat([jump_df[jump_meta], shared_jump_df], axis=1)

# split the features
shared_meta, shared_feats = split_meta_and_features(shared_jump_df, metadata_tag=True)

# checking if the feature space are identical (also looks for feature space order)
assert check_feature_order(
    ref_feat_order=shared_features, input_feat_order=shared_feats
), "Feature space are not identical"

# display
print(
    "Shape of overlapping jump datadrame with overlapping features",
    shared_jump_df.shape,
)
print("Number of meta features", len(shared_meta))
print("Number of features", len(shared_feats))
shared_jump_df.head()


# In[5]:


# save overlapping files
shared_jump_df.to_csv(
    jump_analysis_dir / "shared_feats_jump_data.csv.gz",
    compression="gzip",
    index=False,
)


# ## Identifying shared treatments
# Once the feature space has been narrowed down to only those features shared between both datasets, the next step is to generate a dataset containing shared treatments that are both presentin in the `cell_injury` and `JUMP` datasets. These shared compounds will then be utilized to subset the `JUMP` dataset, which will be considered as the ground truth for downstream analyses.
#
# **Approach**:
# 1. **Identifying shared Compounds**: We compare the compounds present in both datasets to identify the overlapping ones.
# 2. **Subsetting the JUMP Dataset**: Once the overlapping compounds are identified, we subset the `JUMP` dataset to include only those compounds, forming our ground truth dataset.
# 3. **Save dataset**: The dataset will be saved in the `./results/3.jump_analysis`

# ### Identifying Overlapping Compounds
# Here, we used the International Chemical Identifier (InChI) to identify chemicals shared between the JUMP dataset and the Cell Injury dataset.

# In[6]:


cell_injury_InChI_keys = cell_injury_meta_df["Compound InChIKey"].unique().tolist()
jump_InChI_keys = shared_jump_df["Metadata_InChIKey"].unique().tolist()

# identify common InChI Keys
common_compounds_inchikey = list(
    set(cell_injury_InChI_keys).intersection(jump_InChI_keys)
)

# identify the compounds and display
overlapping_compounds_df = cell_injury_meta_df.loc[
    cell_injury_meta_df["Compound InChIKey"].isin(common_compounds_inchikey)
]
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

# lower casing all the entries
overlapping_compounds_df["injury_type"] = overlapping_compounds_df[
    "injury_type"
].str.lower()
overlapping_compounds_df


# Once the common compounds and their associated cell injury types are identified, the next step involves selecting it from the JUMP dataset to select only wells that possess the common InChI keys.

# In[7]:


# selecting rows that contains the overlapping compounds
shared_treat_jump_df = shared_jump_df.loc[
    shared_jump_df["Metadata_InChIKey"].isin(common_compounds_inchikey)
]

# augment filtered JUMP data with labels
shared_treat_jump_df = pd.merge(
    overlapping_compounds_df,
    shared_treat_jump_df,
    right_on="Metadata_InChIKey",
    left_on="Compound InChIKey",
)

# shared treatment jump df
print("shape: ", shared_jump_df.shape)
shared_treat_jump_df.head()


# Now that we have identified the wells treated with overlapping treatments, we want to know the amount of wells that a specific treatment have.

# In[8]:


# count number of wells and agument with injury_code injury_yype and compound name
well_counts_df = (
    shared_treat_jump_df.groupby("Metadata_InChIKey")
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

# In[9]:


# now lets look at the amount of wells have treatments and controls per plate
n_well_treatments = {}
for plate, df in shared_treat_jump_df.groupby("Metadata_Plate"):
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


# Finally we save the shared_treaments_df as a csv.gz file.

# In[10]:


# save overlapping files
shared_treat_jump_df.to_csv(
    jump_analysis_dir / "shared_treatments_jump_data.csv.gz",
    compression="gzip",
    index=False,
)


# ## Applying JUMP dataset to Multi-Class Logistics Regression Model

# In[11]:


# split the data
aligned_meta_cols, aligned_feature_cols = split_meta_and_features(shared_jump_df)

# check if the feature space are the same
X = shared_jump_df[aligned_feature_cols]


assert check_feature_order(
    ref_feat_order=shared_features, input_feat_order=X.columns.tolist()
), "Feature space are not identical"


# In[12]:


# Loading in model
model = joblib.load(modeling_dir / "multi_class_model.joblib")
shuffled_model = joblib.load(modeling_dir / "shuffled_multi_class_model.joblib")


# Here, we apply the JUMP dataset to the model to calculate the probabilities of each injury being present per well. These probabilities are then saved in a tidy long format suitable for plotting in R.

# In[13]:


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
    shared_jump_df[aligned_meta_cols]["Metadata_InChIKey"].to_frame(),
    y_proba_df,
    left_index=True,
    right_index=True,
)
shuffled_y_proba_df = pd.merge(
    shared_jump_df[aligned_meta_cols]["Metadata_InChIKey"].to_frame(),
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


# ## Generating Confusion Matrix

# In[14]:


shared_treat_meta, shared_treat_feats = split_meta_and_features(shared_treat_jump_df)
shared_X = shared_treat_jump_df[shared_treat_feats]
shared_y = shared_treat_jump_df["injury_code"]


# In[15]:


jump_overlap_cm = generate_confusion_matrix_tl(
    model, shared_X, shared_y, shuffled=False, dataset_type="JUMP Overlap"
).fillna(0)
shuffled_jump_overlap_cm = generate_confusion_matrix_tl(
    shuffled_model, shared_X, shared_y, shuffled=True, dataset_type="JUMP Overlap"
).fillna(0)


# In[16]:


# save confusion matrix
pd.concat([jump_overlap_cm, shuffled_jump_overlap_cm]).to_csv(
    modeling_dir / "jump_overlap_confusion_matrix.csv.gz",
    compression="gzip",
    index=False,
)
