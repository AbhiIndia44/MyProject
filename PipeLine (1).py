#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.pipeline import Pipeline, FeatureUnion 
from sklearn.base import BaseEstimator, TransformerMixin


# In[6]:


class Varselector( BaseEstimator, TransformerMixin):
    
    def __init__(self, var_names):
        
        self.feature_names = var_names
        
    def fit(self, x, y = None):
        return self
    
    def transform(self, X):
        return X[self.feature_names]
    
    def get_feature_names(self):
        return self.feature_names

class custom_budget( BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.feature_names = ["AvgBudget"]
        
    def fit(self, x, y = None):
        return self
    
    def transform(self,x, y = None):
        k = x['Budget in Lakhs'].str.split('-', expand = True).astype(float)
        AvgBudget = 0.5 *(k[0] + k[1])
        return pd.DataFrame({'AvgBudget': AvgBudget})
    
    def get_feature_names(self):
        return self.feature_names
    
class custom_area( BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.feature_names = ["AvgArea"]
        
    def fit(self, x, y = None):
        return self
    
    def transform(self,x, y = None):
        k = x["Carpet Area(sq.ft)"].str.split('-', expand = True).astype(float)
        AvgArea = 0.5 *(k[0] + k[1])
        return pd.DataFrame({'AvgArea': AvgArea})
    
    def get_feature_names(self):
        return self.feature_names
    
class custom_age( BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.feature_names = ["AvgAge"]
        
    def fit(self, x, y = None):
        return self
    
    def transform(self,x, y = None):
        k = x['Age of the property'].str.split('-', expand = True).astype(float)
        AvgAge = 0.5 *(k[0] + k[1])
        return pd.DataFrame({'AvgAge': AvgAge})
    
    def get_feature_names(self):
        return self.feature_names
    
    
class custom_dist( BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.feature_names = ["AvgDist"]
        
    def fit(self, x, y = None):
        return self
    
    def transform(self,x, y = None):
        k = x['Public Convenience (Kms)'].str.split('-', expand = True).astype(float)
        AvgAge = 0.5 *(k[0] + k[1])
        return pd.DataFrame({'AvgDist': AvgDist})
    
    def get_feature_names(self):
        return self.feature_names
    
    
    
class string_clean(BaseEstimator, TransformerMixin):
    
    def __init__(self, replace_it = '', self_with = ''):
        self.replace_it = replace_it
        self.replace_with = replace_with
        self.feature_names = []
    
    def fit(self, x , y=None):
        self.feature_names = x.columns
        return self
    
    def transform(self, x):
        for col in x.columns:
            x[col]=x[col].str.replace(self.replace_it, self.replace_with)
        return x
    def get_feature_names(self):
        return self.feature_names
    
    
class convert_to_numeric(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, x , y = True):
        self.feature_names = x.columns
        return self
         
    def transform(self,X):
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors = 'coerce')
        return X 
    
    def get_feature_names(self):
        return self.feature_names 
    
class get_dummies_Pipe(BaseEstimator, TransformerMixin):
    
    def __init__(self,freq_cutoff = 0):
        
        self.freq_cutoff = freq_cutoff 
        self.var_cat_dict = {}
        self.feature_names = []
        
    def fit(self, x, y = None):
        data_cols = x.columns 
        
        for col in data_cols:
            k = x[col].value_counts()
            if (k<= self.frequency_cutoff).sum()== 0:
                cats = k.index[:-1]
            else:
                cats = k.index[k>self.freq_cutoff]
                
            self.var_cat_dict[col] = cats 
            
        for col in self.var_cat_dict[col] :
            for cat in self.var_cat_dict[col]:
                self.feature_names.append(col + '_'+ cat)
                
        return self
    
    def transform(self,x, y=None):
        dummy_data = x.copy()
        for col in self.var_cat_dict.keys():
            for cat in self.var_cat_dict[col]:
                name = col+'_'+cat
                dummy_data[name] = (dummy_data[col]==cat).astype(int)
                
            del dummy_data[col]
        return dummy_data
    
    def get_featurenames(self):
        
        return self.feature_names
    
class DataFrameImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.impute_dict = {}
        self.feature_names= []
    
    def fit(self, x, y = None):
        for col in x.columns:
            if x[col].dtype  == "o" :
                self.impute_dict[col] = "missing"
            else:
                self.impute_dict[col] = x[col].meadian()
        self.feature_names = x.columns
        return self
    
    def transfom(self, x):
        x = x.fillna(self.impute_dict)
        return x
   
    def get_feature_names(self):
        return self.feature_names
        
         
class pdPipeline(Pipeline):

    def get_feature_names(self):

        last_step = self.steps[-1][-1]

        return last_step.get_feature_names()

            
                
  
        
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




