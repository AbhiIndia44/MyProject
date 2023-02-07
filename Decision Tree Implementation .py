#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import tree


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV 


# In[ ]:


params = { 'class_weight': [None, 'balanced'],
          'criterion': ['entropy', 'gini'],
          'max_depth': [None, 5 ,10, 20, 30, 50, 70],
          'min_samples_leaf': [1,2,5,10,15,20],
          'min_samples_split': [2,5,10,15,20]
}


# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf = DecisionTreeClassifier()


# In[ ]:


random_search = RandomizedSearchCV(df, 
                                   cv = 10,
                                   param_distributions = params,
                                   scoring = 'roc_auc',
                                   n_iter = 10,
                                   n_jobs = -1,
                                   verbose = 20
)


# In[ ]:


random_search.fit(x_train, y_train)


# In[ ]:


report(random_search.cv_results, 5)


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




