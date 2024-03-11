#!/usr/bin/env python
# coding: utf-8

# # Spliting Data
#
# This notebook focuses on exploration using two essential files: the annotations data extracted from the actual screening profile (available in the [IDR repository](https://github.com/IDR/idr0133-dahlin-cellpainting/tree/main/screenA)) and the metadata retrieved from the supplementary section of the [research paper](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36829-x/MediaObjects/41467_2023_36829_MOESM5_ESM.xlsx).
#
# We explore the number of unique compounds associated with each cell injury and subsequently cross-reference this information with the screening profile. The aim is to assess the feasibility of using the data for training a machine learning model to predict cell injury.
#
# Next, with the information collected, we then split our data, identify the controls, and determine our training, test, and holdout sets for subsequent analysis.
#

# In[1]:


import json
import pathlib
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ignoring warnings
warnings.catch_warnings(action="ignore")


# ## Paramters
#
# Below are the parameters defined that are used in this notebook

# ----

# In[2]:


# data directory
data_dir = pathlib.Path("../../data").resolve(strict=True)
results_dir = pathlib.Path("../../results").resolve(strict=True)
data_split_dir = (results_dir / "0.data_splits").resolve()
data_split_dir.mkdir(exist_ok=True)

# data paths
suppl_meta_path = data_dir / "41467_2023_36829_MOESM5_ESM.csv.gz"
screen_anno_path = data_dir / "idr0133-screenA-annotation.csv.gz"

# load data
image_profile_df = pd.read_csv(screen_anno_path)
meta_df = image_profile_df[image_profile_df.columns[:31]]
compounds_df = meta_df[["Compound Name", "Compound Class"]]

suppl_meta_df = pd.read_csv(suppl_meta_path)
cell_injury_df = suppl_meta_df[["Cellular injury category", "Compound alias"]]


# ## Data splitting and exploration

# In this notebook, we explore the structure of the experiments, focusing on how treatments were applied, conducting quality data checks, and identifying other relationships within the dataset.
#
# Additionally, we extract datasets that will be utilized in our machine learning processes. These datasets encompass our training data, control sets, and holdout sets.

# -----

# The code below, we extract information regarding various injury types and the corresponding number of compounds known to induce each type of injury.
# then, we cross-reference with the selected compounds and identify wells that have a match in the profile data.
#
# This will be our training dataset, where it contains all the cellular injury labeles and it will be used for machine learning modeling

# In[3]:


# getting profilies based on injury and compound type
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


# In[4]:


# creating a dataframe that contains stratified screen Data
injured_df = pd.concat(injury_profiles)

# display df
print(injured_df.shape)
injured_df.head()


# >This DataFrame categorizes wells based on their injury types and with its corresponding compounds linked to each specific injury type.
# > Note the new column `injury_type` indicating the assigned injury type for each well.
# > This assignment is determined by the component with which the well has been treated.

# In[5]:


neg_control_df = image_profile_df.loc[image_profile_df["Compound Name"] == "DMSO"]
neg_control_df.insert(0, "injury_type", "Control")
neg_control_df.to_csv("negative_control.csv.gz", compression="gzip", index=False)

print("Shape:", neg_control_df.shape)
neg_control_df.head()


# > This dataframe contains all the wells that are considered to be controls.
# > All of these wells have been treated with DMSO

# Now that our dataset is labeled, the next step is to concatenate them and begin splitting the dataset into training, testing, and holdout sets. Additionally, we'll extract metadata from the labeled data to gain a better understanding of how the treatments and plates were generated in this experiment.

# In[6]:


# creating labeled_df
labeled_df = pd.concat([neg_control_df, injured_df]).reset_index(drop=True)

# only select entries where there are not NaN in the 'injury_type'
labeled_df = labeled_df.loc[~labeled_df["injury_type"].isna()]

# save and display
print("shape:", labeled_df.shape)
labeled_df.head()


# Here, we're storing the metadata and feature column names into a JSON file to simplify loading during feature engineering processes.
#
# This will be saved in the `results/0.data_splits` directory

# In[7]:


# collecting metadata and feature column names
feature_cols = labeled_df.columns[32:].tolist()
raw_features = {
    "compartments": list(set([name.split("_")[0] for name in feature_cols])),
    "meta_features": injured_df.columns[:32].tolist(),
    "feature_cols": feature_cols,
}

# saving into JSON file
with open(data_split_dir / "raw_feature_names.json", mode="w") as stream:
    json.dump(raw_features, stream)


# Next we wanted to extract some metadata regarding how many compound and wells are treated with a given compounds
#
# This will be saved in the `results/0.data_splits` directory

# In[8]:


meta_injury = []
for injury_type, df in labeled_df.groupby("injury_type"):
    # extract n_wells, n_compounds and unique compounds per injury_type
    n_wells = df.shape[0]
    unique_compounds = list(df["Compound Name"].unique())
    n_compounds = len(unique_compounds)

    # store information
    meta_injury.append([injury_type, n_wells, n_compounds, unique_compounds])

injury_meta_df = pd.DataFrame(
    meta_injury, columns=["injury_type", "n_wells", "n_compounds", "compound_list"]
).sort_values("n_wells", ascending=False)
injury_meta_df.to_csv(data_split_dir / "injury_well_counts_table.csv", index=False)

# display
print("shape:", injury_meta_df.shape)
injury_meta_df


# > This DataFrame contains information about wells associated with a specific injury type.
# > It includes details such as the number of components used along with the list of the components responsible for the identified injury type.

#

# In[9]:


# make a figure of this table
plt.figure(figsize=(12, 8))

# plotting
sns.barplot(
    data=injury_meta_df,
    x="n_wells",
    y="injury_type",
    palette="viridis",
)
plt.xlabel("Numer of Wells")
plt.ylabel("Injury Type")
plt.title("Number of Wells per Injury Type")

plt.show()


# > Barchart showing the number of wells that are labeled with a given injury

# Next, we construct the profile metadata. This provides a structured overview of how the treatments assicoated with injuries were applied, detailing the treatments administered to each plate.
#
# This will be saved in the `results/0.data_splits` directory

# In[10]:


injury_meta_dict = {}
for injury, df in labeled_df.groupby("injury_type"):
    # collecting treatment metadata
    plates = df["Plate"].unique().tolist()
    treatment_meta = {}
    treatment_meta["n_plates"] = len(plates)
    treatment_meta["n_wells"] = df.shape[0]
    treatment_meta["n_treatments"] = len(df["Compound Name"].unique())
    treatment_meta["associated_plates"] = plates

    # counting treatments
    treatment_counter = {}
    for treatment, df2 in df.groupby("Compound Name"):
        if treatment is np.nan:
            continue
        n_treatments = df2.shape[0]
        treatment_counter[treatment] = n_treatments

    # storing treatment counts
    treatment_meta["treatments"] = treatment_counter
    injury_meta_dict[injury] = treatment_meta

# save dictionary into a json file
with open(data_split_dir / "injury_metadata.json", mode="w") as stream:
    json.dump(injury_meta_dict, stream)


# Here we build a plate metadata infromations where we look at the type of treatments and amount of wells with the treatment that are present in the dataset
#
# This will be saved in `results/0.data_splits`

# In[11]:


plate_meta = {}
for plate_id, df in labeled_df.groupby("Plate"):
    unique_compounds = list(df["Compound Name"].unique())
    n_treatments = len(unique_compounds)

    # counting treatments
    treatment_counter = {}
    for treatment, df2 in df.groupby("Compound Name"):
        n_treatments = df2.shape[0]
        treatment_counter[treatment] = n_treatments

    plate_meta[plate_id] = treatment_counter

# save dictionary into a json file
with open(data_split_dir / "plate_info.json", mode="w") as stream:
    json.dump(plate_meta, stream)


# ## Data Splitting
# ---

# ### Holdout Dataset
#
# Here we collected out holdout dataset. The holdout dataset is a subset of the dataset that is not used during model training or tuning. Instead, it is reserved solely for evaluating the model's performance after it has been trained.
#
# In this notebook, we will include three different types of held-out datasets before proceeding with our machine learning training and evaluation.
#  - Plate hold out
#  - treatment hold out
#  - well hold out
#
# Each of these held outdata will be stored in the `results/0.data_splits` directory
#

# ### Holdout out a treatment holdout plate
#
# Here we are randomly select 15 wells from each treatment from the main dataframe `labeled_df`. Then we remove those wells from the main dataframe

# In[12]:


#### Plate heldout dataset
seed = 0
n_samples = 15

# collecting randomly select wells based on treatment
treatment_holdout_df = []
for treatment, df in labeled_df.groupby("Compound Name", as_index=False):
    heldout_treatment = df.sample(n=15, random_state=seed)
    treatment_holdout_df.append(heldout_treatment)

# genearte treatment holdout dataframe
treatment_holdout_df = pd.concat(treatment_holdout_df)

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
treatment_idx_to_drop = treatment_holdout_df.index.tolist()
labeled_df = labeled_df.drop(treatment_idx_to_drop)
assert all(
    [
        True if num not in labeled_df.index.tolist() else False
        for num in treatment_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
treatment_holdout_df.to_csv(
    data_split_dir / "treatment_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("Treatment holdout shape:", treatment_holdout_df.shape)
treatment_holdout_df.head()


# > Table containig all the held out treatment samples. Saved as `treatment_holdout.csv.gz` in the `results/0.data_splits`

# ### Generating well holdout data
#
# To generate the well hold out data, each plate was iterated and random wells were selected. However, an additional step was condcuting which was to seperate the control wells and the treated wells, due to the large label imbalance with the controls. Therefore, 5 wells were randomly selected and 10 wells were randomly selected from each individual plate

# In[13]:


# parameters
seed = 0
n_controls = 5
n_samples = 10

# setting random seed globally
np.random.seed(seed)


# collecting randomly select wells based on treatment
wells_heldout_df = []
for treatment, df in labeled_df.groupby("Plate", as_index=False):
    # seperate control wells and rest of all wells since there is a huge label imbalance
    # selected 5 control wells and 10 random wells from the plate
    df_control = df.loc[df["Compound Name"] == "DMSO"].sample(
        n=n_controls, random_state=seed
    )
    df_treated = df.loc[df["Compound Name"] != "DMSO"].sample(
        n=n_samples, random_state=seed
    )

    # concatenate those together
    well_heldout = pd.concat([df_control, df_treated])

    wells_heldout_df.append(well_heldout)

# genearte treatment holdout dataframe
wells_heldout_df = pd.concat(wells_heldout_df)

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
wells_idx_to_drop = wells_heldout_df.index.tolist()
labeled_df = labeled_df.drop(wells_idx_to_drop)
assert all(
    [
        True if num not in labeled_df.index.tolist() else False
        for num in treatment_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
wells_heldout_df.to_csv(
    data_split_dir / "wells_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("Wells holdout shape:", wells_heldout_df.shape)
wells_heldout_df.head(10)


# > Table containig all the held out treatment samples. Saved as `wells_holdout.csv.gz` in the `results/0.data_splits`

#

# In[14]:


# plate
seed = 0
n_plates = 10

# setting random seed globally
np.random.seed(seed)

# selecting 10 plates randomly from a list
selected_plates = (
    np.random.choice(labeled_df["Plate"].unique().tolist(), (n_plates, 1))
    .flatten()
    .tolist()
)
plate_holdout_df = labeled_df.loc[labeled_df["Plate"].isin(selected_plates)]

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
plate_idx_to_drop = plate_holdout_df.index.tolist()
labeled_df = labeled_df.drop(plate_idx_to_drop)
assert all(
    [
        True if num not in labeled_df.index.tolist() else False
        for num in treatment_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
plate_holdout_df.to_csv(
    data_split_dir / "plate_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("plate holdout shape:", plate_holdout_df.shape)
plate_holdout_df.head()


# ### Training and Test sets
#
# Below we are going to use the remaining dataset and split them into test and training sets

# In[15]:


# Showing the amount of data we have after removing the holdout data
meta_injury = []
for injury_type, df in labeled_df.groupby("injury_type"):
    # extract n_wells, n_compounds and unique compounds per injury_type
    n_wells = df.shape[0]
    unique_compounds = list(df["Compound Name"].unique())
    n_compounds = len(unique_compounds)

    # store information
    meta_injury.append([injury_type, n_wells, n_compounds, unique_compounds])

injury_meta_df = pd.DataFrame(
    meta_injury, columns=["injury_type", "n_wells", "n_compounds", "compound_list"]
).sort_values("n_wells", ascending=False)
injury_meta_df.to_csv(data_split_dir / "injury_well_counts_table.csv", index=False)

# display
print("shape:", injury_meta_df.shape)
injury_meta_df


# In[16]:


# spliting the meta features and the feature column names
# loading feature columns json file
with open(data_split_dir / "raw_feature_names.json") as stream:
    feature_info = json.load(stream)

# selecing columns for splitting
y_col = "injury_type"
X_cols = feature_info["feature_cols"]


# In[17]:


# spliting the dataset
seed = 0

X = labeled_df[X_cols]
y = labeled_df[y_col]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=seed
)


# In[18]:


X_train.to_csv(data_split_dir / "X_train.csv.gz", index=False, compression="gzip")
y_train.to_csv(data_split_dir / "y_train.csv.gz", index=False, compression="gzip")
X_test.to_csv(data_split_dir / "X_test.csv.gz", index=False, compression="gzip")
y_test.to_csv(data_split_dir / "y_test.csv.gz", index=False, compression="gzip")
