#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
from PipeLine import *
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


# In[27]:


d = pd.read_csv('InternshipProject.csv')


# In[28]:


d.info()


# In[29]:


df = d


# In[30]:


df = df.drop('Conversion', axis = 1)


# In[31]:


df


# In[11]:


# Age of the buyer: V1
# Frequency of call: V2
# Frequency of website visit: V3
# Floor desired: V4
# Parking space: V5
# Rooms: V6
# Lift: V7
#variable selector, missing value imputation 

# Loan period: V8
#varaiable selector, string cleaning, convert to numeric & missing value imputation

# Budget: V9
#variable selector, budget splitter

# Area: V10
#variable selector, area splitter

# Age of the property: V11
#variable selector,age splitter

# Public Convenience: V12
#variable selector,public splitter

# Profession: V13
# Furnishing: V14
# Agency: V15
# Type desired: V16
# Location desired: V17
#variable selector, miising value imputation and get dummies 


# In[12]:


df.head(10)


# In[13]:


col = "Age of the property"


# In[14]:


updated = (df[col] == 'New') 
        
updated
        


# In[15]:


df.loc[updated, col]='0-0'


# In[16]:


df


# In[17]:


target = d['Conversion']


# In[18]:


target


# In[34]:


p1=pdPipeline([
    ('var_select',VarSelector(['Age of the prospective buyer',
                               'Frequency of call', 
                               'Frequency of website visit',
                               'Floor desired ',
                               ' Parking space',
                               'Rooms',
                               'Lift'
                                  ])),
    ('missing_trt',DataFrameImputer())
])


# In[25]:


p2=pdPipeline([
    ('var_select',VarSelector(['Loan Tenure'])),
    ('string_clean',string_clean(replace_it='yrs',replace_with='')),
    ('convert_to_numeric',convert_to_numeric()),
    ('missing_trt',DataFrameImputer())
])


# In[ ]:


p3=pdPipeline([
    ('var_select',VarSelector(['Furnishing',' Profession', 'Agency', 'Type Desired', 'Location Desired'])),
    ('missing_trt',DataFrameImputer()),
    ('create_dummies',get_dummies_Pipe(5))
])


# In[ ]:


p4=pdPipeline([
    ('var_select',VarSelector(['Budget in Lakhs'
                              ])),
    ('custom_budget',custom_budget()),
    ('missing_trt',DataFrameImputer())
])


# In[35]:


p5=pdPipeline([
    ('var_select',VarSelector(['Carpet Area(sq. ft)'
                              ])),
    ('custom_budget',custom_area()),
    ('missing_trt',DataFrameImputer())
])


# In[ ]:


p6=pdPipeline([
    ('var_select',VarSelector(['Age of the property'
                              ])),
    ('custom_budget',custom_age()),
    ('missing_trt',DataFrameImputer())
])


# In[ ]:


p7=pdPipeline([
    ('var_select',VarSelector(['Public Convenience (Kms)'
                              ])),
    ('custom_budget',custom_dist()),
    ('missing_trt',DataFrameImputer())
])


# In[ ]:


data_pipe = FeatureUnion([
    ('p1', p1),
    ('p2', p2),
    ('p3', p3),
    ('p4', p4),
    ('p5', p5),
    ('p6', p6),
    ('p7', p7)
])


# In[ ]:


data_pipe.fit(df)


# In[ ]:


data_pipe.transform(df)


# In[ ]:


x_train = pd.DataFrame(data = data_pipe.transform(df), columns = data_pipe.get_feature_names())


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




