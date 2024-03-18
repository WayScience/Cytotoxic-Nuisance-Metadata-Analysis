#!/usr/bin/env python
# coding: utf-8

# # Spliting Data
# Here, we utilize the feature-selected profiles generated in the preceding module notebook [here](../0.freature_selection/), focusing on dividing the data into training, testing, and holdout sets for machine learning training.

# In[1]:


import json
import pathlib
import warnings

import numpy as np
import pandas as pd

# ignoring warnings
warnings.catch_warnings(action="ignore")


# ## Paramters
#
# Below are the parameters defined that are used in this notebook

# In[2]:


# directory to get all the inputs for this notebook
data_dir = pathlib.Path("../../data").resolve(strict=True)
results_dir = pathlib.Path("../../results").resolve(strict=True)
fs_dir = (results_dir / "0.feature_selection").resolve(strict=True)

# directory to store all the output of this notebook

data_split_dir = (results_dir / "1.data_splits").resolve()
data_split_dir.mkdir(exist_ok=True)


# In[3]:


# data paths
fs_profile_path = (fs_dir / "cell_injury_profile_fs.csv.gz").resolve(strict=True)

# load data
fs_profile_df = pd.read_csv(fs_profile_path)

# display
print("fs profile with control: ", fs_profile_df.shape)
fs_profile_df.head()


# ## Exploring the data set
#
# Below is a  exploration of the selected features dataset. The aim is to identify treatments, extract metadata, and gain a understanding of the experiment's design.

# Below demonstrates the amount of wells does each treatment have.

# In[4]:


# displying the amount of wells per treatments
well_treatments_counts_df = (
    fs_profile_df["Compound Name"].value_counts().to_frame().reset_index()
)

well_treatments_counts_df


# Below we show the amount of wells does a specific cell celluar injury has

# In[5]:


# Displaying how many how wells does each cell injury have
cell_injury_well_counts = (
    fs_profile_df["injury_type"].value_counts().to_frame().reset_index()
)
cell_injury_well_counts


# Next we wanted to extract some metadata regarding how many compound and wells are treated with a given compounds
#
# This will be saved in the `results/0.data_splits` directory

# In[6]:


meta_injury = []
for injury_type, df in fs_profile_df.groupby("injury_type"):
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


# > Barchart showing the number of wells that are labeled with a given injury

# Next, we construct the profile metadata. This provides a structured overview of how the treatments assicoated with injuries were applied, detailing the treatments administered to each plate.
#
# This will be saved in the `results/0.data_splits` directory

# In[7]:


injury_meta_dict = {}
for injury, df in fs_profile_df.groupby("injury_type"):
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

# In[8]:


plate_meta = {}
for plate_id, df in fs_profile_df.groupby("Plate"):
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


# Set numerical labels for the treatment

# In[9]:


# creating a dictionary that contains the numeric-encoded labels and write out as json file
main_labeler = {}
injury_labels_encoder = {
    name: idx for idx, name in enumerate(fs_profile_df["injury_type"].unique().tolist())
}
injury_labels_decoder = {
    idx: name for idx, name in enumerate(fs_profile_df["injury_type"].unique().tolist())
}
main_labeler["encoder"] = injury_labels_encoder
main_labeler["decoder"] = injury_labels_decoder

# write out as json file
with open(data_split_dir / "injury_codes.json", mode="w") as file_buffer:
    json.dump(main_labeler, file_buffer)

# display main_labeler
main_labeler


# Now that we have assigned numerical labels to each type of cell injury, we can replace the corresponding injury names with these numerical values to meet the requirements of machine learning algorithms.

# In[10]:


# updating main dataframe with numerical labels that represents cell injury
# this will be saved as an "injury_code"
injury_code = fs_profile_df["injury_type"].apply(
    lambda injury: injury_labels_encoder[injury]
)

# add the injury code into the main data set
fs_profile_df.insert(0, "injury_code", injury_code)

# # display new injury column
print(fs_profile_df["injury_type"].unique())
print(fs_profile_df["injury_code"].unique())


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
# Each of these held outdata will be stored in the `results/1.data_splits` directory
#

# ### Plate Holdout
#
# Plates are randomly selected based on their Plate ID and save them as our `plate_holdout` data.

# In[11]:


# plate
seed = 0
n_plates = 10

# setting random seed globally
np.random.seed(seed)

# selecting plates randomly from a list
selected_plates = (
    np.random.choice(fs_profile_df["Plate"].unique().tolist(), (n_plates, 1))
    .flatten()
    .tolist()
)
plate_holdout_df = fs_profile_df.loc[fs_profile_df["Plate"].isin(selected_plates)]

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
plate_idx_to_drop = plate_holdout_df.index.tolist()
fs_profile_df = fs_profile_df.drop(plate_idx_to_drop)
assert all(
    [
        True if num not in fs_profile_df.index.tolist() else False
        for num in plate_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
plate_holdout_df.to_csv(
    data_split_dir / "plate_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("plate holdout shape:", plate_holdout_df.shape)
plate_holdout_df.head()


# ### Treatment holdout
#
# To establish our treatment holdout, we first need to find the number of treatments and wells associated with a specific cell injury, considering the removal of randomly selected plates from the previous step.
#
# To determine which cell injuries should be considered for a single treatment holdout, we establish a threshold of 10 unique compounds. This means that a cell injury type must have at least 10 unique compounds to qualify for selection in the treatment holdout. Any cell injury types failing to meet this criterion will be disregarded.
#
# Once the cell injuries are identified for treatment holdout, we select our holdout treatment by grouping each injury type and choosing the treatment with the fewest wells. This becomes our treatment holdout dataset.

# In[12]:


# first we need to find what the treatment and well metadata after removing plates
injury_treatment_metadata = []
for injury_type, df in fs_profile_df.groupby("injury_type"):
    for treatment, df2 in df.groupby("Compound Name"):
        n_wells = df2.shape[0]
        injury_treatment_metadata.append([injury_type, treatment, n_wells])

# convert to df
injury_treatment_metadata = pd.DataFrame(
    injury_treatment_metadata, columns="injury_type treatment n_wells".split()
)
injury_treatment_metadata


# In[13]:


# setting random seed
seed = 0
min_treatments_per_injury = 10

# setting random seed globally
np.random.seed(seed)

# Filter out the injury types for which we can select a complete treatment.
# We are using a threshold of 10. If an injury type is associated with fewer than 10 compounds,
# we do not conduct treatment holdout on those injury types.
accepted_injuries = []
for injury_type, df in injury_treatment_metadata.groupby("injury_type"):
    n_treatments = df.shape[0]
    if n_treatments >= min_treatments_per_injury:
        accepted_injuries.append(df)

accepted_injuries = pd.concat(accepted_injuries)

# Next, we select the treatment that will be held out within each injury type.
# We group treatments based on injury type and choose the treatment with the fewest wells
# as our holdout.
selected_treatments_to_holdout = []
for injury_type, df in accepted_injuries.groupby("injury_type"):
    held_treatment = df.min().iloc[1]
    selected_treatments_to_holdout.append([injury_type, held_treatment])

# convert to dataframe
selected_treatments_to_holdout = pd.DataFrame(
    selected_treatments_to_holdout, columns="injury_type held_treatment".split()
)

print("Below are the accepted cell injuries and treatments to be held out")
selected_treatments_to_holdout


# In[14]:


seed = 0
n_samples = 15

# setting random seed globally
np.random.seed(seed)

# select all wells that have the treatments to be heldout
treatment_holdout_df = fs_profile_df.loc[
    fs_profile_df["Compound Name"].isin(
        selected_treatments_to_holdout["held_treatment"]
    )
]

# take the indices of the held out data frame and use it to drop those samples from
# the main dataset. And then check if those indices are dropped
treatment_idx_to_drop = treatment_holdout_df.index.tolist()
fs_profile_df = fs_profile_df.drop(treatment_idx_to_drop)
assert all(
    [
        True if num not in fs_profile_df.index.tolist() else False
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


# ### Well holdout
#
# To generate the well hold out data, each plate was iterated and random wells were selected. However, an additional step was condcuting which was to seperate the control wells and the treated wells, due to the large label imbalance with the controls. Therefore, 5 wells were randomly selected and 10 wells were randomly selected from each individual plate

# In[15]:


# parameters
seed = 0
n_controls = 5
n_samples = 10

# setting random seed globally
np.random.seed(seed)

# collecting randomly select wells based on treatment
wells_heldout_df = []
for treatment, df in fs_profile_df.groupby("Plate", as_index=False):
    # separate control wells and rest of all wells since there is a huge label imbalance
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
fs_profile_df = fs_profile_df.drop(wells_idx_to_drop)
assert all(
    [
        True if num not in fs_profile_df.index.tolist() else False
        for num in treatment_idx_to_drop
    ]
), "index to be dropped found in the main dataframe"

# saving the holdout data
wells_heldout_df.to_csv(
    data_split_dir / "wells_holdout.csv.gz", index=False, compression="gzip"
)

# display
print("Wells holdout shape:", wells_heldout_df.shape)
wells_heldout_df.head()


# ## Saving training dataset

# Once the data holdout has been generated, the next step is to save the training dataset that will serve as the basis for training the multi-class logistic regression model.

# In[16]:


# Showing the amount of data we have after removing the holdout data
meta_injury = []
for injury_type, df in fs_profile_df.groupby("injury_type"):
    # extract n_wells, n_compounds and unique compounds per injury_type
    n_wells = df.shape[0]
    injury_code = df["injury_code"].unique()[0]
    unique_compounds = list(df["Compound Name"].unique())
    n_compounds = len(unique_compounds)

    # store information
    meta_injury.append(
        [injury_type, injury_code, n_wells, n_compounds, unique_compounds]
    )

# creating data frame
injury_meta_df = pd.DataFrame(
    meta_injury,
    columns=["injury_type", "injury_code", "n_wells", "n_compounds", "compound_list"],
).sort_values("n_wells", ascending=False)
injury_meta_df.to_csv(data_split_dir / "injury_well_counts_table.csv", index=False)

# display
injury_meta_df


# In[17]:


# shape of the update training and testing dataset after removing holdout
print("training shape after removing holdouts", fs_profile_df.shape)
fs_profile_df.head()


# In[18]:


fs_profile_df.to_csv(
    data_split_dir / "training_data.csv.gz", index=False, compression="gzip"
)
