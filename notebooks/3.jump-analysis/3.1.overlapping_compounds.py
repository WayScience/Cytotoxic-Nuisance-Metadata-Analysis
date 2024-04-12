#!/usr/bin/env python
# coding: utf-8

# # Finding overlapping compounds
#
# This notebook aims to identify overlapping compounds present in both the `cell_injury` and `JUMP` datasets. These overlapping compounds will be used for subsetting the `JUMP` dataset, which we'll consider as the ground truth for subsequent analyses.
#
# ## Approach
# 1. **Identifying Overlapping Compounds**: We compare the compounds present in both datasets to identify the overlapping ones.
# 2. **Subsetting the JUMP Dataset**: Once the overlapping compounds are identified, we subset the `JUMP` dataset to include only those compounds, forming our ground truth dataset.
# 3. **Save dataset**: The dataset will be saved in the `./results/3.jump_analysis`
# 4. **Apply to model and evaluate**: Apply to trained shuffled and not shuffled model and evaluate
#

# In[1]:


import json
import pathlib
import sys

import joblib
import pandas as pd
from pycytominer.cyto_utils import infer_cp_features

sys.path.append("../../")  # noqa
from src.utils import generate_confusion_matrix_tl

# In[2]:


# setting seed =
seed = 0

# setting paths
data_path = pathlib.Path("../../data").resolve(strict=True)
jump_data_dir = (data_path / "JUMP_data").resolve(strict=True)
results_dir_path = pathlib.Path("../../results").resolve(strict=True)
data_split_dir = (results_dir_path / "1.data_splits").resolve(strict=True)
modeling_dir = (results_dir_path / "2.modeling").resolve(strict=True)

# datasets paths
cell_injury_metadata_path = (
    data_split_dir / "cell_injury_metadata_after_holdout.csv.gz"
).resolve(strict=True)
jump_data_path = (
    data_path / "JUMP_data/JUMP_aligned_all_plates_normalized_negcon.csv.gz"
).resolve(strict=True)
model_path = (results_dir_path / "2.modeling/multi_class_model.joblib").resolve(
    strict=True
)
shuffled_model_path = (
    results_dir_path / "2.modeling/shuffled_multi_class_model.joblib"
).resolve(strict=True)
injury_codes_path = (data_split_dir / "injury_codes.json").resolve(strict=True)


# In[3]:


# loading in the data
jump_df = pd.read_csv(jump_data_path)
cell_injury_df = pd.read_csv(cell_injury_metadata_path)

# loading json file containing selected feature names
with open(injury_codes_path, mode="r") as infile:
    injury_codes = json.load(infile)

injury_codes_decoder = injury_codes["decoder"]
injury_codes_encoder = injury_codes["encoder"]


# ## Identifying Overlapping Compounds

# Here, we used the International Chemical Identifier (InChI) to identify chemicals shared between the JUMP dataset and the Cell Injury dataset.

# In[4]:


# get all InChI keys
cell_injury_InChI_keys = cell_injury_df["Compound InChIKey"].tolist()
jump_InChI_keys = jump_df["Metadata_InChIKey"].tolist()

# identify common InChI Keys
common_compounds_inchikey = list(
    set(cell_injury_InChI_keys).intersection(jump_InChI_keys)
)

# identify the compounds
overlapping_compounds_df = cell_injury_df.loc[
    cell_injury_df["Compound InChIKey"].isin(common_compounds_inchikey)
]

# inserting injury code
overlapping_compounds_df.insert(
    0,
    "injury_code",
    overlapping_compounds_df["injury_type"].apply(
        lambda name: injury_codes_encoder[name]
    ),
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


# In[5]:


overlapping_compounds_df


# Once the common compounds and their associated cell injury types are identified, the next step involves filtering out the JUMP dataset to select only wells that possess the common InChI keys.

# In[6]:


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


# In[7]:


# count number of wells and agument with injury_code injury_yype and compound name
well_counts_df = (
    overlapping_jump_df.groupby("Metadata_InChIKey")
    .size()
    .to_frame()
    .reset_index()
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


# In[8]:


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
plate_treatments


# In[9]:


# save the dataset
overlapping_jump_df.to_csv(
    jump_data_dir / "overlapping_jump_data.csv.gz", compression="gzip", index=False
)


# ## Applying to Pre-trained model

# Before applying the pretrained model, we must create a downsampled version of the dataset. We saw that there are 3044 wells treated with DMSO, we decided to downsample the DMSO wells. However, instead of randomly selecting DMSO wells, we choose to randomly select only 2 wells per plate. Given that we are working with 24 plates, this approach yields a total of 48 wells. By doing so, we minimize the impact of plate-based variability.
#

# In[10]:


# select only DMSO wells
dmso_wells_df = overlapping_jump_df.loc[overlapping_jump_df["Compound Name"] == "DMSO"]

dmso_wells_per_plate = []
for plate, df in dmso_wells_df.groupby("Metadata_Plate"):
    dmso_df = df.sample(n=2, random_state=seed)
    dmso_wells_per_plate.append(dmso_df)

dmso_wells_df = pd.concat(dmso_wells_per_plate)
print("dmso_wells_df shape", dmso_wells_df.shape)

# now concat balanced DMSO wells with original dataset
overlapping_jump_df = overlapping_jump_df.loc[
    overlapping_jump_df["Compound Name"] != "DMSO"
]
overlapping_jump_df = pd.concat([dmso_wells_df, overlapping_jump_df])

print("DMSO downsampled data:", overlapping_jump_df.shape)
overlapping_jump_df.head()


# In[11]:


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
plate_treatments


# Now that the dataset has been downsampled, next we need to find the common features that are shared between the cell injury dataset and JUMP

# In[12]:


# spliting the data into X and y
cp_features = infer_cp_features(overlapping_jump_df)
y_var = "injury_code"

X = overlapping_jump_df[cp_features].values
y = overlapping_jump_df[y_var].values


# In[13]:


# loading in both Shuffled and Not shuffled models
model = joblib.load(model_path)
shuffled_model = joblib.load(shuffled_model_path)


# In[14]:


# generated a confusion matrix in tidy long format
jump_overlap_cm = generate_confusion_matrix_tl(
    model, X, y, shuffled=False, dataset_type="JUMP Overlap"
).fillna(0)
shuffled_jump_overlap_cm = generate_confusion_matrix_tl(
    shuffled_model, X, y, shuffled=True, dataset_type="JUMP Overlap"
).fillna(0)


# In[15]:


pd.concat([jump_overlap_cm, shuffled_jump_overlap_cm]).to_csv(
    modeling_dir / "jump_overlap_confusion_matrix.csv.gz",
    compression="gzip",
    index=False,
)
