#!/usr/bin/env python
# coding: utf-8

# ### Supplemental Tables and Figures
#
# In this notebook, we create supplemental tables that describe the type of data withheld in each split or under other conditions.

# In[1]:


import pathlib

import dataframe_image as dfi
import pandas as pd

# In[2]:


# columns to select
cols = ["injury_type", "n_wells", "n_compounds"]

# setting up input paths
data_splits_dir = pathlib.Path("../../results/1.data_splits").resolve(strict=True)
fig_dir = pathlib.Path("./figures/supplemental").resolve(strict=True)

# summary info data paths
injury_summary_before_holdout_path = (
    data_splits_dir / "injury_data_summary_before_holdout.csv"
).resolve(strict=True)
injury_summary_after_holdout_path = (
    data_splits_dir / "injury_data_summary_after_holdout.csv"
).resolve(strict=True)
injury_summary_train_split_path = (
    data_splits_dir / "injury_data_summary_train_split.csv"
).resolve(strict=True)
injury_summary_test_split_path = (
    data_splits_dir / "injury_data_summary_test_split.csv"
).resolve(strict=True)


# In[3]:


# loading all the data
injury_summary_before_holdout_df = pd.read_csv(injury_summary_before_holdout_path)[cols]
injury_summary_after_holdout_df = pd.read_csv(injury_summary_after_holdout_path)[cols]
injury_summary_train_split_df = pd.read_csv(injury_summary_train_split_path)[cols]
injury_summary_test_split_df = pd.read_csv(injury_summary_test_split_path)[cols]


# ## showing training datasets before and after holdout

# In[4]:


dfi.export(
    injury_summary_before_holdout_df,
    str(fig_dir / "stable_A_injury_summary_before_holdout.png"),
)
injury_summary_before_holdout_df


# In[5]:


dfi.export(
    injury_summary_after_holdout_df,
    str(fig_dir / "stable_B_injury_summary_after_holdout.png"),
)
injury_summary_after_holdout_df


# ## Display data summary from training splits

# In[6]:


dfi.export(
    injury_summary_train_split_df,
    str(fig_dir / "stable_C_injury_summary_train_split.png"),
)
injury_summary_train_split_df


# In[7]:


dfi.export(
    injury_summary_test_split_df,
    str(fig_dir / "stable_D_injury_summary_test_split.png"),
)
injury_summary_test_split_df


# In[ ]:
