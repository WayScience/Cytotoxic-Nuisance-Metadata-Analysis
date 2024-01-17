#!/usr/bin/env python
# coding: utf-8

# # Metadata Search
#
# This notebook focuses on metadata search using two essential files: the annotations data extracted from the actual screening profile (available in the [IDR repository](https://github.com/IDR/idr0133-dahlin-cellpainting/tree/main/screenA)) and the metadata retrieved from the supplementary section of the [research paper](https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-36829-x/MediaObjects/41467_2023_36829_MOESM5_ESM.xlsx).
#
# The objective is to identify the number of unique compounds associated with each cell injury and subsequently cross-reference this information with the screening profile. The aim is to assess the feasibility of using the data for training a machine learning model to predict cell injury.
#

# In[1]:


import json
import pathlib
from collections import defaultdict

import pandas as pd

# Setting up parameters below:
#

# In[2]:


# data directory
data_dir = pathlib.Path("./data").resolve(strict=True)
results_dir = pathlib.Path("./results")
results_dir.mkdir(exist_ok=True)

# data paths
suppl_meta_path = data_dir / "41467_2023_36829_MOESM5_ESM.csv.gz"
screen_anno_path = data_dir / "idr0133-screenA-annotation.csv.gz"

# load data
image_profile_df = pd.read_csv(screen_anno_path)
meta_df = image_profile_df[image_profile_df.columns[:31]]
compounds_df = meta_df[["Compound Name", "Compound Class"]]

suppl_meta_df = pd.read_csv(suppl_meta_path)
cell_injury_df = suppl_meta_df[["Cellular injury category", "Compound alias"]]


# In this process, we extract information regarding various injury types and the corresponding number of compounds known to induce each type of injury.
# Subsequently, we perform a cross-reference with the selected compounds and identify wells that exhibit a match.
#

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
strat_screen_df = pd.concat(injury_profiles)
strat_screen_df.to_csv(results_dir / "stratified_plate_screen_profile.csv", index=False)

# display df
strat_screen_df.head()


# > **Table 1:** This DataFrame categorizes wells based on their injury types and with its corresponding compounds linked to each specific injury type.
# > Note the new column `injury_type` indicating the assigned injury type for each well.
# > This assignment is determined by the component with which the well has been treated.
#

# Next we wanted to extract some metadata regarding how many compound and wells are treated with a given compounds
#

# In[5]:


# getting meta information of the collected data
meta_injury = []
for df in injury_profiles:
    injury_type = df["injury_type"].unique()[0]
    n_wells = df.shape[0]
    n_compounds = len(df["Compound Name"].unique().tolist())
    compound_list = df["Compound Name"].unique().tolist()

    meta_injury.append([injury_type, n_wells, n_compounds, compound_list])


injury_meta_df = pd.DataFrame(
    meta_injury, columns=["injury_type", "n_wells", "n_compounds", "compound_list"]
)
injury_meta_df.to_csv(results_dir / "injury_metadata.csv", index=False)
injury_meta_df


# > **Table 2** This DataFrame contains information about wells associated with a specific injury type.
# > It includes details such as the number of components used along with the list of the components responsible for the identified injury type.
#

# Next we take table 2 and format it into json format for improved readability

# In[6]:


# collect metadata from table
injury_meta_dict = {}
for row in injury_meta_df.iterrows():
    # selecting row based on injury type
    selected_row = row[1].values.tolist()
    injury_name = selected_row[0]
    print(injury_name)

    # creating a sub dictionary gathers all meta data
    meta_dict = dict(
        n_wells=selected_row[1],
        n_compounds=selected_row[2],
        compound_list=selected_row[3],
    )

    # adding to main dictionary
    injury_meta_dict[selected_row[0]] = meta_dict

# save dictionary into a json file
with open(results_dir / "injury_metadata.json", mode="w") as stream:
    json.dump(injury_meta_dict, stream)


# Lastly, we extract of all control wells, which are treated with DMSO.
#

# In[7]:


# getting only control wells
control_df = image_profile_df.loc[image_profile_df["Compound Name"] == "DMSO"]
control_df.to_csv(results_dir / "control_wells.csv", index=False)
control_df


# > **Table 3**: This dataframe only contains the control wells
#
