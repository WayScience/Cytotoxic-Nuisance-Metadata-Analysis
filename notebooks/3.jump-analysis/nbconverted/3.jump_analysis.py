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
barcode_path = (jump_data_dir / "barcode_platemap.csv").resolve(strict=True)
# loading only cell injury metadata (after holdout has been applied)
cell_injury_metadata_path = (
    data_split_dir / "aligned_cell_injury_metadata_after_holdout.csv.gz"
).resolve(strict=True)

# model paths
multi_class_model_path = (modeling_dir / "aligned_multi_class_model.joblib").resolve(
    strict=True
)
shuffled_multi_class_model_path = (
    modeling_dir / "aligned_shuffled_multi_class_model.joblib"
).resolve(strict=True)

# overlapping feature space path
shared_feature_space_path = (
    fs_results_dir / "aligned_cell_injury_shared_feature_space.json"
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

# loading JUMP barcode
barcode_df = pd.read_csv(barcode_path)

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

# Experimental metadata
jump_exp_meta = pd.read_csv(
    "https://raw.githubusercontent.com/WayScience/JUMP-single-cell/d00868cfdb18143db0469e3cc0700b35c03cf811/reference_plate_data/experiment-metadata.tsv",
    delimiter="\t",
)


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

# # split the features
shared_meta, shared_feats = split_meta_and_features(shared_jump_df, metadata_tag=True)

# # checking if the feature space are identical (also looks for feature space order)
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
    jump_analysis_dir / "JUMP_shared_feats_jump_data.csv.gz",
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

# In[ ]:


cell_injury_InChI_keys = cell_injury_meta_df["Compound InChIKey"].unique().tolist()
jump_InChI_keys = shared_jump_df["Metadata_InChIKey"].unique().tolist()

# # identify common InChI Keys
common_compounds_inchikey = list(
    set(cell_injury_InChI_keys).intersection(jump_InChI_keys)
)

# # identify the compounds and display in cell injury data
overlapping_compounds_df = cell_injury_meta_df.loc[
    cell_injury_meta_df["Compound InChIKey"].isin(common_compounds_inchikey)
]
unique_compound_names = overlapping_compounds_df["Compound Name"].unique().tolist()
print("Identified overlapping compounds:", ", ".join(unique_compound_names))

# now create a dataframe where it contains the injury code, name and injury type
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

# In[ ]:


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
print("shape: ", shared_treat_jump_df.shape)
shared_treat_jump_df.head()


# Now that we have identified the wells treated with overlapping treatments, we want to know the amount of wells that a specific treatment have.

# In[8]:


# count number of wells and augment with injury_code injury_type and compound name
well_counts_df = (
    shared_treat_jump_df.groupby("Metadata_InChIKey")
    # counting the number of wells
    .size()
    .to_frame()
    .reset_index()
    # merge based on InChIKey
    .merge(
        overlapping_compounds_df,
        left_on="Metadata_InChIKey",
        right_on="Compound InChIKey",
    )
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


# In[9]:


# Here we select the the compound associated with the cytoskeletal injury
# below we use the InChIKey to extract all wells that have been treated by the overlapping compound
# these wells will serve as our grouth truth
gt_jump_cyto_injury_df = shared_jump_df.loc[
    shared_jump_df["Metadata_InChIKey"] == "IAKHMKGGTNLKSZ-INIZCTEOSA-N"
]

# updating the shared_jump_df by removing the ground truth entries
shared_jump_df = shared_jump_df.drop(index=gt_jump_cyto_injury_df.index, inplace=False)


# Finally we save the shared_treaments_df as a csv.gz file.

# In[10]:


# save overlapping files
shared_treat_jump_df.to_csv(
    jump_analysis_dir / "JUMP_shared_treatments_jump_data.csv.gz",
    compression="gzip",
    index=False,
)


# ## Applying JUMP dataset to Multi-Class Logistics Regression Model

# In[11]:


# metadata to select from the jump dataset to add into the ground truth
gt_metadata_cols = [
    "Metadata_broad_sample",
    "Metadata_Plate",
    "Metadata_Well",
    "Metadata_pert_iname",
    "Metadata_InChIKey",
]

# split the data
aligned_meta_cols, aligned_feature_cols = split_meta_and_features(shared_jump_df)

# JUMP ground truth feature space (24 wells)
gt_df = gt_jump_cyto_injury_df[gt_metadata_cols + aligned_feature_cols]
gt_X = gt_jump_cyto_injury_df[aligned_feature_cols]


# other JUMP wells feature space (not labeled)
X = shared_jump_df[aligned_feature_cols]

# check if the feature space are the same
assert check_feature_order(
    ref_feat_order=shared_features, input_feat_order=X.columns.tolist()
), "Feature space are not identical"


# In[12]:


# Loading in model
model = joblib.load(modeling_dir / "aligned_multi_class_model.joblib")
shuffled_model = joblib.load(modeling_dir / "aligned_shuffled_multi_class_model.joblib")


# Here, we apply the JUMP dataset to the model to calculate the probabilities of each injury being present per well. These probabilities are then saved in a tidy long format suitable for plotting in R.

# In[13]:


# cols to selected
col_to_sel = ["pred_injury", "datatype", "shuffled"]
# get all injury classes
injury_classes = [injury_decoder[str(code)] for code in model.classes_.tolist()]

# prediction probabilities on both non-shuffled and shuffled models
y_pred = model.predict(X)
gt_y_pred = model.predict(gt_X)

y_proba = model.predict_proba(X)
gt_y_proba = model.predict_proba(gt_X)

shuffled_y_pred = shuffled_model.predict(X)
shuffled_gt_y_pred = shuffled_model.predict(gt_X)

shuffled_y_proba = shuffled_model.predict_proba(X)
shuffled_gt_y_proba = shuffled_model.predict_proba(gt_X)

# convert to pandas dataframe add prediction col
y_proba_df = pd.DataFrame(y_proba)
y_proba_df["pred_injury"] = y_pred.flatten()
y_proba_df["datatype"] = "JUMP"
y_proba_df["shuffled_model"] = False

gt_y_proba_df = pd.DataFrame(gt_y_proba)
gt_y_proba_df["pred_injury"] = gt_y_pred.flatten()
gt_y_proba_df["datatype"] = "JUMP Overlap"
gt_y_proba_df["shuffled_model"] = False
# gt_y_proba_df = pd.concat([gt_df[gt_metadata_cols], gt_y_proba_df], axis=1)

shuffled_y_proba_df = pd.DataFrame(shuffled_y_proba)
shuffled_y_proba_df["pred_injury"] = shuffled_y_pred.flatten()
shuffled_y_proba_df["datatype"] = "JUMP"
shuffled_y_proba_df["shuffled_model"] = True

shuffled_gt_y_proba_df = pd.DataFrame(shuffled_gt_y_proba)
shuffled_gt_y_proba_df["pred_injury"] = shuffled_gt_y_pred.flatten()
shuffled_gt_y_proba_df["datatype"] = "JUMP Overlap"
shuffled_gt_y_proba_df["shuffled_model"] = True
# shuffled_gt_y_proba_df = pd.concat([gt_df[gt_metadata_cols], shuffled_gt_y_proba_df], axis=1)

# concatenate all prediction
# update the predicted label columns to injury name
all_proba_scores = pd.concat(
    [y_proba_df, gt_y_proba_df, shuffled_y_proba_df, shuffled_gt_y_proba_df]
)
all_proba_scores.columns = [
    injury_decoder[str(col_name)] for col_name in all_proba_scores.columns[0:15]
] + col_to_sel
all_proba_scores["pred_injury"] = all_proba_scores["pred_injury"].apply(
    lambda injury_code: injury_decoder[str(injury_code)]
)


# We will now save the ground truth predictions, which include probability scores for each injury type and model type, as well as the predicted injury. These results will be stored in the `./results/3.jump_analysis` directory.

# In[ ]:


# update columns by replacing the column index (which are the injury codes) to the injury type
gt_y_proba_df.columns = [
    injury_decoder[str(injury_code)]
    for injury_code in gt_y_proba_df.columns.tolist()[0:15]
] + col_to_sel
shuffled_gt_y_proba_df.columns = [
    injury_decoder[str(injury_code)]
    for injury_code in shuffled_gt_y_proba_df.columns.tolist()[0:15]
] + col_to_sel

# add the metadata to both prediction dataframes
gt_y_proba_df = pd.concat(
    [
        gt_df[gt_metadata_cols].reset_index(drop=True),
        gt_y_proba_df.reset_index(drop=True),
    ],
    axis=1,
)
shuffled_gt_y_proba = pd.concat(
    [
        gt_df[gt_metadata_cols].reset_index(drop=True),
        shuffled_gt_y_proba_df.reset_index(drop=True),
    ],
    axis=1,
)

# concat both dataframes into one
gt_proba_preds_df = pd.concat([gt_y_proba_df, shuffled_gt_y_proba])

# updating the "pred_injury" injury codes to injury_type names
gt_proba_preds_df["pred_injury"] = (
    gt_proba_preds_df["pred_injury"]
    .astype(str)
    .apply(lambda injury_code: injury_decoder[injury_code])
)

# saving ground truth probabilities
gt_save_path = (jump_analysis_dir / "JUMP_ground_truth_probabilities.csv").resolve()
gt_proba_preds_df.to_csv(gt_save_path, index=False)

# display dataframe
gt_proba_preds_df.head()


# Next, we will focus only on the probability scores for JUMP wells predicted to have cytoskeletal injury (excluding ground truth data).

# In[15]:


# next only select cytoskeletal probability scores
cytoskeletal_proba_scores = all_proba_scores[col_to_sel + ["Cytoskeletal"]]
cytoskeletal_proba_scores = cytoskeletal_proba_scores.rename(
    columns={"Cytoskeletal": "Cytoskeletal_proba"}
)

# Saving only cytoskeletal probability scores
cytoskeletal_proba_scores.to_csv(
    jump_analysis_dir / "JUMP_cytoskeletal_proba_scores.csv.gz",
    compression="gzip",
    index=False,
)

# display
cytoskeletal_proba_scores.head()


# Next, we will obtain all probability scores for JUMP wells predicted to have any injury (excluding ground truth data).

# In[16]:


# making all probabilities tidy long
all_injury_proba = all_proba_scores[col_to_sel + injury_classes].melt(
    id_vars=["pred_injury", "datatype", "shuffled"],
    var_name="injury_compared_to",
    value_name="proba",
)

# save file
all_injury_proba.to_csv(
    jump_analysis_dir / "JUMP_all_injury_proba.csv.gz", index=False, compression="gzip"
)
all_injury_proba.head()


# ## Generating Confusion Matrix

# In[17]:


shared_treat_meta, shared_treat_feats = split_meta_and_features(shared_treat_jump_df)
shared_X = shared_treat_jump_df[shared_treat_feats]
shared_y = shared_treat_jump_df["injury_code"]


# In[18]:


jump_overlap_cm = generate_confusion_matrix_tl(
    model, shared_X, shared_y, shuffled=False, dataset_type="JUMP Overlap"
).fillna(0)
shuffled_jump_overlap_cm = generate_confusion_matrix_tl(
    shuffled_model, shared_X, shared_y, shuffled=True, dataset_type="JUMP Overlap"
).fillna(0)


# In[19]:


# save confusion matrix
pd.concat([jump_overlap_cm, shuffled_jump_overlap_cm]).to_csv(
    modeling_dir / "jump_overlap_confusion_matrix.csv.gz",
    compression="gzip",
    index=False,
)


# ## Creating supplemental Table

# Below we are creating a supplemental table showing the types of injury predicted associated with the compounds found in the JUMP-CP datat set

# In[20]:


# setting column arrangement
col_arrangement = [
    "Metadata_Plate",
    "Plate_Map_Name",
    "Metadata_Well",
    "Cell_type",
    "Metadata_pert_type",
    "Time",
    "Metadata_gene",
    "Metadata_pert_iname",
    "Metadata_target_sequence",
    "pred_injury",
    "probability",
]

# creating a rename dict
rename_assay_type = {
    "JUMP-Target-1_orf_platemap": "orf",
    "JUMP-Target-1_compound_platemap": "compound",
    "JUMP-Target-1_crispr_platemap": "crispr",
}

# split meta and feature columns
shared_jump_meta, shared_jump_feats = split_meta_and_features(shared_jump_df)

# selecting the columns
predicted_df = shared_jump_df[
    [
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_gene",
        "Metadata_pert_iname",
        "Metadata_target_sequence",
        "Metadata_pert_type",
    ]
]

# converting injury codes to injury names
predicted_df["pred_injury"] = [
    injury_decoder[str(injury_code)] for injury_code in y_pred.tolist()
]

# obtaining the probability score of the predicted injury
predicted_df["probability"] = y_proba.max(axis=1).tolist()

# add experimental metadata
sel_jump_exp_meta = jump_exp_meta[["Assay_Plate_Barcode", "Cell_type", "Time"]]
predicted_df = predicted_df.merge(
    sel_jump_exp_meta, left_on="Metadata_Plate", right_on="Assay_Plate_Barcode"
).drop(columns=["Assay_Plate_Barcode"])

# Merge barcode information by using the Plate ID to indicate the type of assay conducted
predicted_df = pd.merge(
    predicted_df, barcode_df, left_on="Metadata_Plate", right_on="Assay_Plate_Barcode"
)
predicted_df = predicted_df.drop("Assay_Plate_Barcode", axis=1)
predicted_df = predicted_df[col_arrangement].rename(
    columns={"Plate_Map_Name": "Assay_type"}
)


# updating column containign assay information
predicted_df["Assay_type"] = predicted_df["Assay_type"].apply(
    lambda assay_code: rename_assay_type[assay_code]
)

# if 'Metadata_pert_type' is NaN this means that no sample was added
# rational behind this is because the "Metadata_broad_sample" is empty indicating no sample was added
predicted_df["Metadata_pert_type"] = predicted_df["Metadata_pert_type"].fillna("EMPTY")

# update 'trt' value to "treatment" to improve readability
predicted_df["Metadata_pert_type"] = predicted_df["Metadata_pert_type"].apply(
    lambda name: "treatment" if name == "trt" else name
)

# dropping duplicates
predicted_df = predicted_df.drop_duplicates()

# display
print(predicted_df.shape)
predicted_df.head()


# In[21]:


predicted_df.to_csv(
    jump_analysis_dir / "JUMP_CP_Pilot_predicted_injuries.csv", index=False
)
