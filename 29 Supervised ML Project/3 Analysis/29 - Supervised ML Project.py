#!/usr/bin/env python
# coding: utf-8

# In[55]:


import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, chi2
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, recall_score, roc_auc_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from xgboost import XGBClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
import shap
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import lightgbm as lgb
from lightgbm import LGBMClassifier


# In[3]:


df=pd.read_csv("Customer-Churn-Records.csv")
df1=df.copy()
df1


# In[4]:


df1.info()


# In[5]:


df1_droped = df1.drop(columns=['RowNumber'])


# In[9]:


# new df with customer id and surname
custid_srn_df = df1_droped[['CustomerId', 'Surname']]


# In[71]:


df_cln= df1_droped.drop(columns=['Surname', 'CustomerId', 'Complain'])
df_cln


# In[72]:


df_final = pd.get_dummies(df_cln, columns=['Geography', 'Gender', 'Card Type']).astype(int)
df_final=df_final.astype(int)


# In[73]:


# X, y values, Create and print correlation matrix
X=df_final.drop('Exited',axis=1)
y=df_final['Exited']
corr_matrix = X.corr()
corr_matrix


# In[74]:


# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[75]:


std= StandardScaler()


# In[92]:


pipe= Pipeline([
    ('scaler',std),
    ('classifier',RandomForestClassifier())
])

params=[
{'classifier': [LogisticRegression()],
     'classifier__penalty': ['l2']
     },



    {'classifier': [RandomForestClassifier()],
     'classifier__n_estimators': [100],
     'classifier__n_jobs': [3]
    },


    {'classifier': [AdaBoostClassifier()],
     'classifier__n_estimators': [100],
     'classifier__learning_rate': [0.01]
    },


    {'classifier': [GradientBoostingClassifier()],
     'classifier__learning_rate': [0.01],
     'classifier__n_estimators': [100]
    },


    {'classifier': [SVC()],
     'classifier__kernel': ['linear'],
    },


    {'classifier': [XGBClassifier()],
    'classifier__learning_rate': [0.1],
     'classifier__n_estimators': [100],
    'classifier__max_depth': [10]
    },

    {'classifier': [KNeighborsClassifier()],
     'classifier__n_neighbors': [10],
     'classifier__n_jobs': [3]
    },

    {'classifier': [LGBMClassifier()],
     'classifier__learning_rate': [0.01],
    }
]


# In[93]:


CV = GridSearchCV(pipe, params, scoring = 'recall_micro', cv = 5, error_score = 'raise', verbose = 1, n_jobs=-1)


# In[94]:


CV.fit(X_train, y_train)


# In[95]:


CV.best_score_


# In[96]:


CV.best_params_


# In[97]:


# results
pd.DataFrame(CV.cv_results_)


# In[99]:


y_pred = CV.predict(X_test)
y_pred


# In[100]:


# classification report
classification_report(y_test, y_pred)



# In[102]:


# dictionary...
report_dict = classification_report(y_test, y_pred, output_dict=True)


# In[103]:


# pd.df
report_df = pd.DataFrame(report_dict).transpose()


# In[104]:


report_df


# In[105]:


explainer = shap.Explainer(CV.predict, X_test)
shap_values = explainer(X_test)
shap.plots.bar(shap_values)
shap.summary_plot(shap_values)


# In[106]:


shap.plots.bar(shap_values)


# In[107]:


shap.summary_plot(shap_values)


# In[109]:


df_final.to_csv('df_cleaned', index=False)


# In[ ]:




