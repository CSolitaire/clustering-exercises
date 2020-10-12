import pandas as pd
import numpy as np
import os
from env import host, user, password

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
    or if cached == True reads in mall customer df from a csv file, returns df
    '''
    if cached or os.path.isfile('zillow_df.csv') == False:
        df = new_zillow_data()
    else:
        df = pd.read_csv('zillow_df.csv', index_col=0)
    return df

#################### Prepare ##################

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols
    

def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(object_cols, dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df

    
def train_validate_test(df, target):
    '''
    this function takes in a dataframe and splits it into 3 samples, 
    a test, which is 20% of the entire dataframe, 
    a validate, which is 24% of the entire dataframe,
    and a train, which is 56% of the entire dataframe. 
    It then splits each of the 3 samples into a dataframe with independent variables
    and a series with the dependent, or target variable. 
    The function returns 3 dataframes and 3 series:
    X_train (df) & y_train (series), X_validate & y_validate, X_test & y_test. 
    '''
    # split df into test (20%) and train_validate (80%)
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)

    # split train_validate off into train (70% of 80% = 56%) and validate (30% of 80% = 24%)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # split train into X (dataframe, drop target) & y (series, keep target only)
    #X_train = train.drop(columns=[target]) # No Target in Clustering
    #y_train = train[target] # No Target in Clustering
    
    # split validate into X (dataframe, drop target) & y (series, keep target only)
    #X_validate = validate.drop(columns=[target]) # No Target in Clustering
    #y_validate = validate[target] # No Target in Clustering
    
    # split test into X (dataframe, drop target) & y (series, keep target only)
    #X_test = test.drop(columns=[target]) # No Target in Clustering
    #y_test = test[target]  # No Target in Clustering
    
    #return X_train, y_train, X_validate, y_validate, X_test, y_test
    return train, validate, test

def get_numeric_X_cols(train, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in train.columns.values if col not in object_cols]
        
    return numeric_cols


def min_max_scale(train, validate, test, numeric_cols):
    '''
    this function takes in 3 dataframes with the same columns, 
    a list of numeric column names (because the scaler can only work with numeric columns),
    and fits a min-max scaler to the first dataframe and transforms all
    3 dataframes using that scaler. 
    it returns 3 dataframes with the same column names and scaled values. 
    '''
    # create the scaler object and fit it to X_train (i.e. identify min and max)
    # if copy = false, inplace row normalization happens and avoids a copy (if the input is already a numpy array).

    scaler = MinMaxScaler(copy=True).fit(X_train[numeric_cols])

    #scale X_train, X_validate, X_test using the mins and maxes stored in the scaler derived from X_train. 
    # 
    train_scaled_array = scaler.transform(train[numeric_cols])
    validate_scaled_array = scaler.transform(validate[numeric_cols])
    test_scaled_array = scaler.transform(test[numeric_cols])

    # convert arrays to dataframes
    train_explore = train

    train_scaled = pd.DataFrame(train_scaled_array, 
                                  columns=numeric_cols).\
                                  set_index([train.index.values])

    validate_scaled = pd.DataFrame(validate_scaled_array, 
                                     columns=numeric_cols).\
                                     set_index([validate.index.values])

    test_scaled = pd.DataFrame(test_scaled_array, 
                                 columns=numeric_cols).\
                                 set_index([test.index.values])

    
    return train_explore, train_scaled, validate_scaled, test_scaled

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

def wrangle_zillow(path):
    df = pd.read_csv(path)
    # Rename columns for clarity
    df.rename(columns={"hashottuborspa":"hottub_spa","fireplacecnt":"fireplace","garagecarcnt":"garage"}, inplace = True) 
    # Replaces NaN values with 0
    df['garage'] = df['garage'].replace(np.nan, 0)
    df['hottub_spa'] = df['hottub_spa'].replace(np.nan, 0)
    df['lotsizesquarefeet'] = df['lotsizesquarefeet'].replace(np.nan, 0)
    df['poolcnt'] = df['poolcnt'].replace(np.nan, 0)
    df['fireplace'] = df['fireplace'].replace(np.nan, 0)
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    df["zip"] = df["regionidzip"].astype('category')
    df["zip"] = df["zip"].cat.codes
    ## Add dummy variables as new columns in dataframe and rename them, delete origional
    df["propertylandusetypeid"]=df["propertylandusetypeid"].astype('category')
    df["zip"] = df["zip"].cat.codes
    
    ''''''
    # Rename duplicate column id/id with id_delete/id
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
    # Delete duplicate columns id_delete and pid (dupilcate of parcelid)
    df.drop(columns = ['id_delete','pid'], inplace = True)
    # Sorting for single unit properties
    unit_df = df[df['unitcnt']==1]
    room_df = df[df['roomcnt']>0]
    garage_df = df[df['garagecarcnt']>0]
    bed_df = df[df['bedroomcnt']>0]
    bath_df = df[df['bathroomcnt']>0]
    seconday_featutes_df = pd.concat([bed_df, bath_df, garage_df, room_df, unit_df]).drop_duplicates('id').reset_index(drop=True)
    p261_df = df[df['propertylandusetypeid'] == 261]
    p263_df = df[df['propertylandusetypeid'] == 263]
    p264_df = df[df['propertylandusetypeid'] == 264]    
    p266_df = df[df['propertylandusetypeid'] == 266] 
    p273_df = df[df['propertylandusetypeid'] == 273]
    p275_df = df[df['propertylandusetypeid'] == 275]
    p276_df = df[df['propertylandusetypeid'] == 276]    
    p279_df = df[df['propertylandusetypeid'] == 279]
    property_df = pd.concat([p261_df, p263_df, p264_df, p266_df, p273_df, p275_df, p276_df, p279_df]).drop_duplicates('id').reset_index(drop=True)
    df = pd.concat([seconday_featutes_df, property_df]).drop_duplicates('id').reset_index(drop=True)
    ''''''

    # Function to Prep Data (Delete Columns and Rows)
    df = data_prep(df)

    # get object column names
    object_cols = get_object_cols(df)
        
    # create dummy vars
    df = create_dummies(df, object_cols)
        
    # split data (taxvaluedollarcnt is target)
    X_train, y_train, X_validate, y_validate, X_test, y_test = train_validate_test(df, 'taxvaluedollarcnt')
        
    # get numeric column names
    numeric_cols = get_numeric_X_cols(X_train, object_cols)

    # scale data 
    X_train_scaled, X_validate_scaled, X_test_scaled = min_max_scale(X_train, X_validate, X_test, numeric_cols)

    return df, train_explore, train_scaled, validate_scaled, test_scaled

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

