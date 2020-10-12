import pandas as pd
import numpy as np
import os
from env import host, user, password
import scipy as sp 
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, QuantileTransformer, PowerTransformer, RobustScaler, MinMaxScaler


#################### Acquire ##################


def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the Codeup db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_zillow_data():
    '''
    This function reads the mall customer data from the Codeup db into a df,
    write it to a csv file, and returns the df. 
    '''
    sql_query = '''
                select * from properties_2017
                join (select id, logerror, pid, tdate from predictions_2017 pred_2017
                join (SELECT parcelid as pid, Max(transactiondate) as tdate FROM predictions_2017 GROUP BY parcelid) as sq1
                on (pred_2017.parcelid = sq1.pid and pred_2017.transactiondate = sq1.tdate)) as sq2
                on (properties_2017.parcelid = sq2.pid)
                left join airconditioningtype using (airconditioningtypeid)
                left join architecturalstyletype using (architecturalstyletypeid)
                left join buildingclasstype using (buildingclasstypeid)
                left join heatingorsystemtype using (heatingorsystemtypeid)
                left join propertylandusetype using (propertylandusetypeid)
                left join storytype using (storytypeid)
                left join typeconstructiontype using (typeconstructiontypeid)
                left join unique_properties using (parcelid)
                where latitude is not null and longitude is not null;
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))
    df.to_csv('zillow_df.csv')
    return df

def get_zillow_data(cached=False):
    '''
    This function reads in zillow customer data from Codeup database if cached == False 
    or if cached == True reads in zillow df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow_df.csv') == False:
        df = new_zillow_data()
    else:
        df = pd.read_csv('zillow_df.csv', index_col=0)
    return df

#################### Prepare ##################

# Function to Prep Data (Delete Columns and Rows)

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    
    def remove_columns(df, cols_to_remove):  
        df = df.drop(columns=cols_to_remove)
        return df

    def handle_missing_values(df, prop_required_column = .5, prop_required_row = .75):
        threshold = int(round(prop_required_column*len(df.index),0))
        df.dropna(axis=1, thresh=threshold, inplace=True)
        threshold = int(round(prop_required_row*len(df.columns),0))
        df.dropna(axis=0, thresh=threshold, inplace=True)
        return df
    
    df = remove_columns(df, cols_to_remove)  # Removes Specified Columns
    df = handle_missing_values(df, prop_required_column, prop_required_row) # Removes Specified Rows
    df.dropna(inplace=True) # Drops all Null Values From Dataframe
    return df

    # Rename duplicate column id/id with id_delete/id
def rename_columns(df):
    df.columns = ['parcelid', 'typeconstructiontypeid', 'storytypeid',
            'propertylandusetypeid', 'heatingorsystemtypeid', 'buildingclasstypeid',
            'architecturalstyletypeid', 'airconditioningtypeid', 'id_delete',
            'basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid',
            'calculatedbathnbr', 'decktypeid', 'finishedfloor1squarefeet',
            'calculatedfinishedsquarefeet', 'finishedsquarefeet12',
            'finishedsquarefeet13', 'finishedsquarefeet15', 'finishedsquarefeet50',
            'finishedsquarefeet6', 'fips', 'fireplacecnt', 'fullbathcnt',
            'garagecarcnt', 'garagetotalsqft', 'hashottuborspa', 'latitude',
            'longitude', 'lotsizesquarefeet', 'poolcnt', 'poolsizesum',
            'pooltypeid10', 'pooltypeid2', 'pooltypeid7',
            'propertycountylandusecode', 'propertyzoningdesc',
            'rawcensustractandblock', 'regionidcity', 'regionidcounty',
            'regionidneighborhood', 'regionidzip', 'roomcnt', 'threequarterbathnbr',
            'unitcnt', 'yardbuildingsqft17', 'yardbuildingsqft26', 'yearbuilt',
            'numberofstories', 'fireplaceflag', 'structuretaxvaluedollarcnt',
            'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
            'taxamount', 'taxdelinquencyflag', 'taxdelinquencyyear',
            'censustractandblock', 'id', 'logerror', 'pid', 'tdate',
            'airconditioningdesc', 'architecturalstyledesc', 'buildingclassdesc',
            'heatingorsystemdesc', 'propertylandusedesc', 'storydesc',
            'typeconstructiondesc']
    df.drop(columns = ['id_delete','pid'], inplace = True)
    return df

def split_df(df):
    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    return train, validate, test

def scale_df(train, validate, test):
     # Assign variables
    X_train = train
    X_validate = validate
    X_test = test
    X_train_explore = train

    # Scale data
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
    
    return X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled

def wrangle_zillow():
    df = get_zillow_data(cached=True)
    
    # Rename columns with duplicate id's
    df = rename_columns(df)

    # Prep Data Function
    df = data_prep(df)

    # split dataset
    train, validate, test = split_df(df)

    # scale dataset
    X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled = scale_df(train, validate, test)

    return df, X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled

    ################## Explore ####################

def count_and_percent_missing_row(df):
    percent_missing = df.isnull().sum() * 100 / len(df)
    total_missing = df.isnull().sum()
    missing_value_df = pd.DataFrame({'num_rows_missing': total_missing,
                                     'pct_rows_missing': percent_missing})
    return missing_value_df

def count_and_percent_missing_column(df):
    num_rows = df.loc[:].isnull().sum()
    num_cols_missing = df.loc[:, df.isna().any()].count()
    pct_cols_missing = round(df.loc[:, df.isna().any()].count() / len(df.index) * 100, 3)
    missing_cols_and_rows_df = pd.DataFrame({'num_cols_missing': num_cols_missing,
                                             'pct_cols_missing': pct_cols_missing,
                                             'num_rows': num_rows})
    missing_cols_and_rows_df = missing_cols_and_rows_df.fillna(0)
    missing_cols_and_rows_df['num_cols_missing'] = missing_cols_and_rows_df['num_cols_missing'].astype(int)
    return missing_cols_and_rows_df

def df_summary(df):
    print('The shape of the df:') 
    print(df.shape)  # df shape (row/column)
    print('\n')
    print('Columns, Non-Null Count, Data Type:')
    print(df.info())      # Column, Non Null Count, Data Type
    print('\n')
    print('Summary statistics for the df:') 
    print(df.describe())             # Summary Statistics on Numeric Data Types
    print('\n')
    print('Number of NaN values per column:') 
    print(df.isna().sum())           # NaN by column
    print('\n')
    print('Number of NaN values per row:')  
    print(df.isnull().sum(axis=1))   # NaN by row
    print('\n')
    print('Count and percent missing per row')
    print(count_and_percent_missing_row(df))
    print('\n')
    print('Count and percent missing per column')
    print(count_and_percent_missing_column(df))
    print('\n')
    print('Value Counts per Column:')
    for col in df.columns:
        print('-' * 40 + col + '-' * 40 , end=' - ')
        display(df[col].value_counts(dropna=False).head(10))
        #display(df_resp[col].value_counts())  # Displays all Values, not just First 10

