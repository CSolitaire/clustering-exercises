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
    df.latitude = df.latitude / 1000000
    df.longitude = df.longitude / 1000000
    return df

def split_df(df):
    '''
    This function splits our dataframe in to train, validate, and test
    '''
    # split dataset
    train_validate, test = train_test_split(df, test_size = .2, random_state = 123)
    train, validate = train_test_split(train_validate, test_size = .3, random_state = 123)
    return train, validate, test    


def remove_columns(train, validate, test, cols_to_remove):  
    train = train.drop(columns=cols_to_remove)
    validate = validate.drop(columns=cols_to_remove)
    test = test.drop(columns=cols_to_remove)
    return train, validate, test

def handle_missing_values(train, validate, test, prop_required_column = .5, prop_required_row = .75):
    threshold = int(round(prop_required_column*len(train.index),0))
    train.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_column*len(validate.index),0))
    validate.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_column*len(test.index),0))
    test.dropna(axis=1, thresh=threshold, inplace=True)

    threshold = int(round(prop_required_row*len(train.columns),0))
    train.dropna(axis=0, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(validate.columns),0))
    validate.dropna(axis=0, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(test.columns),0))
    test.dropna(axis=0, thresh=threshold, inplace=True)
    
    return train, validate, test
    
def clean_data(train, validate, test):
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
        "lotsizesquarefeet",
        "unitcnt",
        "regionidcity",
        "buildingqualitytypeid",
        "regionidcity",
        "regionidzip",
        "yearbuilt",
        "censustractandblock"
    ]
    for col in cols:
        median = train[col].median()
        train[col].fillna(median, inplace=True)
        validate[col].fillna(median, inplace=True)
        test[col].fillna(median, inplace=True)
    return train, validate, test

def post_selection_processing(train, validate, test):
    
    cols = ["yearbuilt","calculatedfinishedsquarefeet","regionidzip",
            "bathroomcnt","bedroomcnt","lotsizesquarefeet","rawcensustractandblock",
            "regionidcity","roomcnt","unitcnt","assessmentyear"]
    train[cols] = train[cols].astype('int')
    validate[cols] = validate[cols].astype('int')
    test[cols] = test[cols].astype('int')
    return train, validate, test 

def clean_zillow(df):
    modify_columns(df)
    train, validate, test = split_df(df)
    train, validate, test = clean_data(train, validate, test)
    train, validate, test = remove_columns(train, validate, test, cols_to_remove=['buildingqualitytypeid','finishedsquarefeet12','fullbathcnt', 'regionidcounty'])
    train, validate, test = handle_missing_values(train, validate, test)
    train, validate, test = post_selection_processing(train, validate, test)
    return train, validate, test  


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

################## Outliers and IQR #################### 

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


###################################### STEP #1 
# In the next cell type the following
'''
'''
#This text prints information regrding the outlier columns created 
'''
wrangle.add_upper_outlier_columns(df, k=1.5)    
outlier_cols = [col for col in df if col.endswith('_outliers')]
for col in outlier_cols:
    print('~~~\n' + col)
    data = df[col][df[col] > 0]
    print(data.describe())
'''
###################################### STEP #2

# Print this code to remove colums in dataframe
'''
X_train_explore.drop([x for x in df if x.endswith('_outliers')], 1, inplace = True)
'''