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
                where latitude is not null and longitude is not null
                and tdate between '2017-01-01' and '2017-12-31';
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

def label_county(row):
    if row['fips'] == 6037:
        return 'Los Angeles'
    elif row['fips'] == 6059:
        return 'Orange'
    elif row['fips'] == 6111:
        return 'Ventura'

def modify_columns(df):
    '''
    This function drops colums that are duplicated or unneessary
    '''
    df['county'] = df.apply(lambda row: label_county(row), axis=1)
    df.drop(columns = ['id','pid','id.1',"propertylandusetypeid", "heatingorsystemtypeid", 'fips',"propertyzoningdesc","calculatedbathnbr"], inplace = True)
    df.heatingorsystemdesc = df.heatingorsystemdesc.fillna("None")
    

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
    
    remove_columns(df, cols_to_remove)  # Removes Specified Columns
    handle_missing_values(df, prop_required_column, prop_required_row) # Removes Specified Rows

def cat_variables(train, validate, test):
    '''
    This function take categorical variables and splits them in to cat.codes for modeling
    '''
    train["propertycountylandusecode"] = train["propertycountylandusecode"].astype('category')
    train["propertycountylandusecode"] = train["propertycountylandusecode"].cat.codes
    validate["propertycountylandusecode"] = validate["propertycountylandusecode"].astype('category')
    validate["propertycountylandusecode"] = validate["propertycountylandusecode"].cat.codes
    test["propertycountylandusecode"] = test["propertycountylandusecode"].astype('category')
    test["propertycountylandusecode"] = test["propertycountylandusecode"].cat.codes
    ############################################################################################
    train["tdate"] = train["tdate"].astype('category')
    train["tdate"] = train["tdate"].cat.codes
    validate["tdate"] = validate["tdate"].astype('category')
    validate["tdate"] = validate["tdate"].cat.codes
    test["tdate"] = test["tdate"].astype('category')
    test["tdate"] = test["tdate"].cat.codes
    ############################################################################################
    train["county"] = train["county"].astype('category')
    train["county"] = train["county"].cat.codes
    validate["county"] = validate["county"].astype('category')
    validate["county"] = validate["county"].cat.codes
    test["county"] = test["county"].astype('category')
    test["county"] = test["county"].cat.codes
    ############################################################################################
    train["heatingorsystemdesc"] = train["heatingorsystemdesc"].astype('category')
    train["heatingorsystemdesc"] = train["heatingorsystemdesc"].cat.codes
    validate["heatingorsystemdesc"] = validate["heatingorsystemdesc"].astype('category')
    validate["heatingorsystemdesc"] = validate["heatingorsystemdesc"].cat.codes
    test["heatingorsystemdesc"] = test["heatingorsystemdesc"].astype('category')
    test["heatingorsystemdesc"] = test["heatingorsystemdesc"].cat.codes
    ############################################################################################
    train["propertylandusedesc"] = train["propertylandusedesc"].astype('category')
    train["propertylandusedesc"] = train["propertylandusedesc"].cat.codes
    validate["propertylandusedesc"] = validate["propertylandusedesc"].astype('category')
    validate["propertylandusedesc"] = validate["propertylandusedesc"].cat.codes
    test["propertylandusedesc"] = test["propertylandusedesc"].astype('category')
    test["propertylandusedesc"] = test["propertylandusedesc"].cat.codes
    ############################################################################################
    train["regionidcity"] = train["regionidcity"].astype('category')
    train["regionidcity"] = train["regionidcity"].cat.codes # Not a Number
    validate["regionidcity"] = validate["regionidcity"].astype('category')
    validate["regionidcity"] = validate["regionidcity"].cat.codes # Not a Number
    test["regionidcity"] = test["regionidcity"].astype('category')
    test["regionidcity"] = test["regionidcity"].cat.codes # Not a Number
    ############################################################################################
    train["regionidcounty"] = train["regionidcounty"].astype('category')
    train["regionidcounty"] = train["regionidcounty"].cat.codes # Not a Number
    validate["regionidcounty"] = validate["regionidcounty"].astype('category')
    validate["regionidcounty"] = validate["regionidcounty"].cat.codes # Not a Number
    test["regionidcounty"] = test["regionidcounty"].astype('category')
    test["regionidcounty"] = test["regionidcounty"].cat.codes # Not a Number
    ############################################################################################
    train["regionidzip"] = train["regionidzip"].astype('category')
    train["regionidzip"] = train["regionidzip"].cat.codes # Not a Number
    validate["regionidzip"] = validate["regionidzip"].astype('category')
    validate["regionidzip"] = validate["regionidzip"].cat.codes # Not a Number
    test["regionidzip"] = test["regionidzip"].astype('category')
    test["regionidzip"] = test["regionidzip"].cat.codes # Not a Number
    ############################################################################################
    train["yearbuilt"] = train["yearbuilt"].astype('category')
    train["yearbuilt"] = train["yearbuilt"].cat.codes # Not a Number   
    validate["yearbuilt"] = validate["yearbuilt"].astype('category')
    validate["yearbuilt"] = validate["yearbuilt"].cat.codes # Not a Number 
    test["yearbuilt"] = test["yearbuilt"].astype('category')
    test["yearbuilt"] = test["yearbuilt"].cat.codes # Not a Number  

def split_df(df):
    '''
    This function splits our dataframe in to train, validate, and test
    '''
    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    return train, validate, test    

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
    This function takes in a dataframe and prep, splits, and scales the data
    '''
    modify_columns(df)
    data_prep(df)
    train, validate, test = split_df(df)
    # Categorical/Discrete columns to use mode to replace nulls

    cols = [
        "buildingqualitytypeid",
        "regionidcity",
        "regionidzip",
        "yearbuilt",
        "regionidcity",
        "censustractandblock"
    ]

    for col in cols:
        mode = int(train[col].mode()) # I had some friction when this returned a float (and there were no decimals anyways)
        train[col].fillna(value=mode, inplace=True)
        validate[col].fillna(value=mode, inplace=True)
        test[col].fillna(value=mode, inplace=True)

    # Continuous valued columns to use median to replace nulls

    cols = [
        "structuretaxvaluedollarcnt",
        "taxamount",
        "taxvaluedollarcnt",
        "landtaxvaluedollarcnt",
        "structuretaxvaluedollarcnt",
        "finishedsquarefeet12",
        "calculatedfinishedsquarefeet",
        "fullbathcnt",
        "lotsizesquarefeet"
    ]


    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        validate[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)

    cat_variables(train, validate, test)
    X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled = scale_df(train, validate, test)
    return df, X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled

# df, X_train_explore, X_train_scaled, X_validate_scaled, X_test_scaled = wrangle_zillow.wrangle_zillow(acquire.get_zillow_data(cached=False)) | To call function

################## Explore ####################

def nulls_by_col(df):
    num_missing = df.isnull().sum()
    rows = df.shape[0]
    pct_missing = num_missing / rows
    cols_missing = pd.DataFrame({'number_missing_rows': num_missing, 'percent_rows_missing': pct_missing})
    return cols_missing

def nulls_by_row(df):
    num_cols_missing = df.isnull().sum(axis=1)
    pct_cols_missing = df.isnull().sum(axis=1)/df.shape[1]*100
    rows_missing = pd.DataFrame({'num_cols_missing': num_cols_missing, 'pct_cols_missing': pct_cols_missing}).reset_index().groupby(['num_cols_missing','pct_cols_missing']).count().rename(index=str, columns={'index': 'num_rows'}).reset_index()
    return rows_missing 

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
    for col in df.columns:
        print('-' * 40 + col + '-' * 40 , end=' - ')
        display(df[col].value_counts(dropna=False).head(10))
        #display(df_resp[col].value_counts())  # Displays all Values, not just First 10

# df_summary(df) | To call function

################## Outliers and IQR #################### STEP #1

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

def add_upper_outlier_columns(df, k): # Call This Function First
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

# add_upper_outlier_columns(X_train_explore, k=1.5) ** This is how to call the function

###################################### STEP #2 
# In the next cell type the following
'''
'''
#This text prints information regrding the outlier columns created
'''
add_upper_outlier_columns(df, k=1.5)    
outlier_cols = [col for col in df if col.endswith('_outliers')]
for col in outlier_cols:
    print('~~~\n' + col)
    data = df[col][df[col] > 0]
    print(data.describe())
'''
###################################### STEP #3

# Print this code to remove colums in dataframe
'''
X_train_explore.drop([x for x in df if x.endswith('_outliers')], 1, inplace = True)
'''

