import pandas as pd
import numpy as np
import scipy as sp 
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler

#################### Prepare ##################

def wrangle_zillow(path):
    '''This function makes all necessary changes to the dataframe for exploration and modeling'''
    df = pd.read_csv(path)
    # Rename columns for clarity
    df.rename(columns={"hashottuborspa":"hottub_spa","fireplacecnt":"fireplace","garagecarcnt":"garage"}, inplace = True)
    df.rename(columns = {'Unnamed: 0':'delete', 'id.1':'delete1'}, inplace = True)

    # Replaces NaN values with 0
    df['garage'] = df['garage'].replace(np.nan, 0)
    df['hottub_spa'] = df['hottub_spa'].replace(np.nan, 0)
    df['lotsizesquarefeet'] = df['lotsizesquarefeet'].replace(np.nan, 0)
    df['poolcnt'] = df['poolcnt'].replace(np.nan, 0)
    df['fireplace'] = df['fireplace'].replace(np.nan, 0)
        
    ## Convert to Category
    df["zip"] = df["regionidzip"].astype('category')
    df["useid"]= df["propertylandusetypeid"].astype('category')
    df["year"]= df["yearbuilt"].astype('category')

    # Add Category Codes
    df["zip_cc"] = df["zip"].cat.codes
    df["useid_cc"] = df["useid"].cat.codes
    df["year_cc"] = df["year"].cat.codes

    # Columns to drop
    df.drop(columns= ['parcelid','id','airconditioningtypeid','architecturalstyletypeid','basementsqft','buildingclasstypeid','buildingqualitytypeid'], inplace = True)
    df.drop(columns= ['calculatedbathnbr','decktypeid','finishedfloor1squarefeet','finishedsquarefeet12','finishedsquarefeet13','finishedsquarefeet15'], inplace = True)
    df.drop(columns= ['finishedsquarefeet50','finishedsquarefeet6', 'fullbathcnt','heatingorsystemtypeid','poolsizesum','pooltypeid10','pooltypeid2'], inplace = True)
    df.drop(columns= ['pooltypeid7','propertycountylandusecode','propertyzoningdesc','rawcensustractandblock','regionidcity','regionidcounty','regionidneighborhood'], inplace = True)
    df.drop(columns= ['storytypeid','threequarterbathnbr','typeconstructiontypeid','unitcnt','yardbuildingsqft17','yardbuildingsqft26', 'numberofstories'], inplace = True)
    df.drop(columns= ['fireplaceflag','structuretaxvaluedollarcnt','assessmentyear','landtaxvaluedollarcnt', 'taxdelinquencyflag','taxdelinquencyyear'], inplace = True)
    df.drop(columns= ['censustractandblock','logerror','transactiondate','garagetotalsqft', "yearbuilt", "regionidzip", "propertylandusetypeid",'delete','delete1'], inplace = True)

    # Rows to drop
    rows_to_remove = [1600, 1628, 5099, 5969, 8109, 8407, 8521, 8849, 11562, 12430, 14313, 20313, 21502]
    df = df[~df.index.isin(rows_to_remove)]
    
    # Problem 'bedbathratio' - New Feature (Ratio of bedroomcnt and bathroomcnt)
    #df['bedbathratio'] = df.bedroomcnt.div(df.bathroomcnt, axis=0)

    # drop any nulls
    df = df.dropna()

    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    train.shape, validate.shape, test.shape

    # Assign variables
    # x df's are all numeric cols 
    X_train = train.drop(columns=['taxvaluedollarcnt','zip','useid',"year", 'taxamount', 'fips', 'latitude', 'longitude'])
    X_validate = validate.drop(columns=['taxvaluedollarcnt','zip','useid',"year", 'taxamount', 'fips', 'latitude', 'longitude'])
    X_test = test.drop(columns=['taxvaluedollarcnt','zip','useid',"year", 'taxamount', 'fips', 'latitude', 'longitude'])
    X_train_explore = train

    # I need X_train_explore set to train so I have access to the target variable.

    # y df's are just fertility
    y_train = train[['taxvaluedollarcnt']]
    y_validate = validate[['taxvaluedollarcnt']]
    y_test = test[['taxvaluedollarcnt']]

    scaler = MinMaxScaler(copy=True).fit(X_train)

    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, 
                                columns=X_train.columns.values).\
                                set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled, 
                                    columns=X_validate.columns.values).\
                                set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled, 
                                    columns=X_test.columns.values).\
                                set_index([X_test.index.values])

    return df, X_train_explore, X_train_scaled, y_train, X_validate_scaled, y_validate, X_test_scaled, y_test


def run():
    print("Prepare: Cleaning acquired data...")
    # Write code here
    print("Prepare: Completed!")
