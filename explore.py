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
def counties_no_outliers(train):
    def la_county(train):
        # Create LA County df
        la_train_df = train[train.county=='Los Angeles']
        #la_validate_df = validate[validate.county=='Los Angeles']
        #la_test_df = test[test.county=='Los Angeles'] 
        
        # Remove Outliers (Train) Using IQR
        
        # Make new dataframe from categorical variables
        cat_df_la = la_train_df[["regionidzip","county","propertylandusedesc","heatingorsystemdesc"]].copy()
        # remove categorical variavbles for outlier examination
        la_df = la_train_df.drop(columns=["regionidzip","county","propertylandusedesc","heatingorsystemdesc"])
        # Drop Outliers in Dataframe (Set = 6)
        Q1 = la_train_df.quantile(0.25)
        Q3 = la_train_df.quantile(0.75)
        IQR = Q3 - Q1
        la_df_out = la_df[~((la_df < (Q1 - 6 * IQR)) |(la_df > (Q3 + 6 * IQR))).any(axis=1)]
        la_train_df = pd.concat([la_df_out, cat_df_la], axis=1).reindex(la_df_out.index)
        return la_train_df

    def vc_county(train):
        # Create Venture County df
        vc_train_df = train[train.county=='Ventura']
        #vc_validate_df = validate[validate.county=='Ventura']
        #vc_test_df = test[test.county== 'Ventura'] 
        
        # Remove Outliers (Train) Using IQR
        
        # Make new dataframe from categorical variables
        cat_df_vc = vc_train_df[["regionidzip","county","propertylandusedesc","heatingorsystemdesc"]].copy()
        # remove categorical variavbles for outlier examination
        vc_df = vc_train_df.drop(columns=["regionidzip","county","propertylandusedesc","heatingorsystemdesc"])
        # Drop Outliers in Dataframe (Set = 6)
        Q1 = vc_train_df.quantile(0.25)
        Q3 = vc_train_df.quantile(0.75)
        IQR = Q3 - Q1
        vc_df_out = vc_df[~((vc_df < (Q1 - 6 * IQR)) |(vc_df > (Q3 + 6 * IQR))).any(axis=1)]
        vc_train_df = pd.concat([vc_df_out, cat_df_vc], axis=1).reindex(vc_df_out.index)
        return vc_train_df

    def oc_county(train):
        # Create Orange County df
        oc_train_df = train[train.county=='Orange']
        #oc_validate_df = validate[validate.county=='Orange']
        #oc_test_df = test[test.county== 'Orange'] 
        
        # Remove Outliers (Train) Using IQR
        
        # Make new dataframe from categorical variables
        cat_df_oc = oc_train_df[["regionidzip","county","propertylandusedesc","heatingorsystemdesc"]].copy()
        # remove categorical variavbles for outlier examination
        oc_df = oc_train_df.drop(columns=["regionidzip","county","propertylandusedesc","heatingorsystemdesc"])
        # Drop Outliers in Dataframe (Set = 6)
        Q1 = oc_train_df.quantile(0.25)
        Q3 = oc_train_df.quantile(0.75)
        IQR = Q3 - Q1
        oc_df_out = oc_df[~((oc_df < (Q1 - 6 * IQR)) |(oc_df > (Q3 + 6 * IQR))).any(axis=1)]
        oc_train_df = pd.concat([oc_df_out, cat_df_oc], axis=1).reindex(oc_df_out.index)
        return oc_train_df

    return la_train_df, vc_train_df, oc_train_df