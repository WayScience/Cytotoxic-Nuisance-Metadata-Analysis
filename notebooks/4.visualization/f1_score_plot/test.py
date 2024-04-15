#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# In[2]:


df = pd.read_csv("../../../results/2.modeling/all_f1_scores.csv.gz")


# In[3]:


df


# In[4]:


df.dataset_type.unique()


# In[5]:


df.loc[df.dataset_type == "Treatment Holdout"]


# In[ ]:
