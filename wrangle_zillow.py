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

def data_prep(df, cols_to_remove=[], prop_required_column=.5, prop_required_row=.75):
    '''
    This function removes columns and rows below a specified 'complete' threshold
    '''
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

def rename_columns(df):
    '''
    This function renames duplicate columns (id) and deletes duplicate values (pid)
    '''
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
    '''
    This function splits our dataframe in to train, validate, and test
    '''
    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    return train, validate, test

def cat_variables(df):
    '''
    This function turns all categorical variables in to cat code columns
    '''
    for col_name in df.columns:
        if(df[col_name].dtype == 'object'):
            df[col_name]= df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes
        else:
            df[col_name]= df[col_name].astype('category')
            df[col_name] = df[col_name].cat.codes
    return df

def scale_df(train, validate, test):
    '''
    This function scales data using the MinMaxScaler
    '''
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

    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns.values).set_index([X_train.index.values])

    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns= X_validate.columns.values).set_index([X_validate.index.values])

    X_test_scaled = pd.DataFrame(X_test_scaled, columns= X_test.columns.values).set_index([X_test.index.values])
    
    return X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled


def wrangle_zillow(df):
    '''
    This function takes a SQL querry and returns several split and scaled dataframes
    '''
    # Rename columns with duplicate id's
    df = rename_columns(df)

    # Rescale lat and long
    #df.latitude = df.latitude / 1000000
    #df.longitude = df.longitude / 1000000

    # Prep Data Function
    df = data_prep(df)
    
    # Convert categorical colums to cat codes
    df = cat_variables(df)
    
    # split dataset
    train, validate, test = split_df(df)
    
    # scale dataset
    X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled = scale_df(train, validate, test)

    return df, X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled

# df, X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled = wrangle_zillow.wrangle_zillow(acquire.get_zillow_data(cached=False)) | To call function

    ################## Explore ####################

def count_and_percent_missing_row(df):
    '''
    This function determines the count and percentage of rows are missing
    '''
    percent_missing = df.isnull().sum() * 100 / len(df)
    total_missing = df.isnull().sum()
    missing_value_df = pd.DataFrame({'num_rows_missing': total_missing,
                                     'pct_rows_missing': percent_missing})
    return missing_value_df

def count_and_percent_missing_column(df):
    '''
    This function returns the count and percentage of the missing rows
    '''
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
    '''
    This function returns all the summary information of the dataframe
    '''
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

# df_summary(df) | To call function

######################################

def quartiles_and_outliers(df):
    def get_upper_outliers(s, k):
        '''
        Given a series and a cutoff value, k, returns the upper outliers for the
        series.

        The values returned will be either 0 (if the point is not an outlier), or a
        number that indicates how far away from the upper bound the observation is.
        '''
        q1, q3 = s.quantile([.25, .75])
        iqr = q3 - q1
        upper_bound = q3 + k * iqr
        return s.apply(lambda x: max([x - upper_bound, 0]))

    def add_upper_outlier_columns(df, k):
        '''
        Add a column with the suffix _outliers for all the numeric columns
        in the given dataframe.
        '''
        # outlier_cols = {col + '_outliers': get_upper_outliers(df[col], k)
        #                 for col in df.select_dtypes('number')}
        # return df.assign(**outlier_cols)
        for col in df.select_dtypes('number'):
            df[col + '_outliers'] = get_upper_outliers(df[col], k)
        return df

    add_upper_outlier_columns(df, k=1.5)    
    outlier_cols = [col for col in df if col.endswith('_outliers')]
    for col in outlier_cols:
        print('~~~\n' + col)
        data = df[col][df[col] > 0]
        print(data.describe())

# quartiles_and_outliers(df) | To call function