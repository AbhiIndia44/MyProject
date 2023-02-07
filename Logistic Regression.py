#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


# In[3]:


params = { 'class weight': ['balanced', None],
          'penalty': ['l1', 'l2'],
          'C': np.linspace(0.001, 1, 10)
}


# In[4]:


model = LogisticRegression(fit_intercept = True)


# In[9]:


grid_search = GridSearchCV(model,
                           param_grid = params, 
                           cv = 10,
                          scoring = 'roc_auc', 
                           n_jobs = -1, 
                           verbose = 20)


# In[8]:


grid_search.fit(x_train, y_train)


# In[10]:


grid_search.fit(x_train, y_train)


# In[ ]:


grid_search.bestestimator_

logr = grid_search.best_estimator_
# In[ ]:


logr.fit(x_train, y_train)


# In[ ]:


(logr.coef_[0]==0).sum()


# In[ ]:


list(zip(x_train.columns, logr.coef_[0]))


# In[ ]:


logr.predict_proba(x_test)


# In[ ]:


logr.classes_


# In[ ]:


cutoffs = np.linspace(0.01, 0.99, 99)


# In[ ]:


logr.predict_proba(x_train)


# In[ ]:


logr.classes_


# In[ ]:


train_score = logr.predict_proba(x_train)[:,1]
real = y_train


# In[13]:


KS_all = []

for cutoff in cutoffs:
    predicted = (train_score>cutoff).astype(int)
    
    TP = ((preicted == 1) & (real == 1)).sum()
    TN = ((predicted == 0) & (real==0)).sum()
    FP = ((predicted == 1) & (real == 0)).sum()
    FN = ((predicted == 0) & (real == 1)).sum()
    
    P = TP + FN
    N = TN + FP
    
    KS = (TP/P) - (FP/N)
    KS_all.append(KS)
    
    


# In[ ]:


list(zip(cutoffs, KS_all))


# In[12]:


mycutoff = cutoffs[KS_all == max(KS_all)][0]
mycutoff


# In[14]:


test_score = logr.predict_proba(x_test)[:,1]
test_score


# In[ ]:


(test_score > mycutoff).astype(interesting)


# In[ ]:




