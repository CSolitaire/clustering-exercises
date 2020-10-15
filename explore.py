import pandas as pd
import numpy as np
import os
from env import host, user, password
import scipy as sp 
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

################## Explore ####################

def la_county(train):
    # Create LA County df
    la_train_df = train[train.county=='Los Angeles']
    la_validate_df = validate[validate.county=='Los Angeles']
    la_test_df = test[test.county=='Los Angeles'] 
    
    # Remove Outliers (Train) Using IQR
    
    # Make new dataframe from categorical variables
    cat_df = la_train_df[["regionidzip","county","propertylandusedesc","heatingorsystemdesc"]].copy()
    # remove categorical variavbles for outlier examination
    la_df = la_train_df.drop(columns=["regionidzip","county","propertylandusedesc","heatingorsystemdesc"])
    # Drop Outliers in Dataframe (Set = 6)
    Q1 = la_train_df.quantile(0.25)
    Q3 = la_train_df.quantile(0.75)
    IQR = Q3 - Q1
    la_df_out = la_train_df[~((la_df < (Q1 - 6 * IQR)) |(la_df > (Q3 + 6 * IQR))).any(axis=1)]
    la_train_df = pd.concat([la_df_out, cat_df], axis=1).reindex(la_df_out.index)
    return la_train_df