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
    df.drop(columns = ['id','pid','id.1',"propertylandusetypeid", "heatingorsystemtypeid",'fips',"propertyzoningdesc","calculatedbathnbr"], inplace = True)
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
            "roomcnt","unitcnt","assessmentyear"]
    train[cols] = train[cols].astype('int')
    validate[cols] = validate[cols].astype('int')
    test[cols] = test[cols].astype('int')
    return train, validate, test 

def cat_columns(train, validate, test):
    cols = ["regionidzip","heatingorsystemdesc","propertylandusedesc","county","yearbuilt"]
    train[cols] = train[cols].astype("category")
    validate[cols] = validate[cols].astype("category")
    test[cols] = test[cols].astype("category")
    return train, validate, test 

def create_features(train, validate, test):
    train['age'] = 2017 - train.yearbuilt
    validate['age'] = 2017 - validate.yearbuilt
    test['age'] = 2017 - test.yearbuilt
    # create taxrate variable
    train['taxrate'] = train.taxamount/train.taxvaluedollarcnt
    validate['taxrate'] = validate.taxamount/validate.taxvaluedollarcnt
    test['taxrate'] = test.taxamount/test.taxvaluedollarcnt
    # create acres variable
    train['acres'] = train.lotsizesquarefeet/43560
    validate['acres'] = validate.lotsizesquarefeet/43560
    test['acres'] = test.lotsizesquarefeet/43560
    # dollar per square foot-structure
    train['structure_dollar_per_sqft'] = train.structuretaxvaluedollarcnt/train.calculatedfinishedsquarefeet
    validate['structure_dollar_per_sqft'] = validate.structuretaxvaluedollarcnt/validate.calculatedfinishedsquarefeet
    test['structure_dollar_per_sqft'] = test.structuretaxvaluedollarcnt/test.calculatedfinishedsquarefeet
    # dollar per square foot-land
    train['land_dollar_per_sqft'] = train.landtaxvaluedollarcnt/train.lotsizesquarefeet
    validate['land_dollar_per_sqft'] = validate.landtaxvaluedollarcnt/validate.lotsizesquarefeet
    test['land_dollar_per_sqft'] = test.landtaxvaluedollarcnt/test.lotsizesquarefeet
    # ratio of beds to baths
    train['bed_bath_ratio'] = train.bedroomcnt/train.bathroomcnt
    validate['bed_bath_ratio'] = validate.bedroomcnt/validate.bathroomcnt
    test['bed_bath_ratio'] = test.bedroomcnt/test.bathroomcnt
    return train, validate, test

def remove_outliers(train, validate, test ):
    '''
    remove outliers in bed, bath, zip, square feet, acres & tax rate
    '''
    train[((train.bathroomcnt <= 7) & (train.bedroomcnt <= 7) & 
               (train.regionidzip < 100000) & 
               (train.bathroomcnt > 0) & 
               (train.bedroomcnt > 0) & 
               (train.acres < 10) &
               (train.calculatedfinishedsquarefeet < 7000) & 
               (train.taxrate < .05)
              )]
     
    validate[((validate.bathroomcnt <= 7) & (validate.bedroomcnt <= 7) & 
               (validate.regionidzip < 100000) & 
               (validate.bathroomcnt > 0) & 
               (validate.bedroomcnt > 0) & 
               (validate.acres < 10) &
               (validate.calculatedfinishedsquarefeet < 7000) & 
               (validate.taxrate < .05)
              )]
    
    test[((test.bathroomcnt <= 7) & (test.bedroomcnt <= 7) & 
               (test.regionidzip < 100000) & 
               (test.bathroomcnt > 0) & 
               (test.bedroomcnt > 0) & 
               (test.acres < 10) &
               (test.calculatedfinishedsquarefeet < 7000) & 
               (test.taxrate < .05)
              )]
    
    return train, validate, test

def col_to_drop_post_processing(train, validate, test):
    cols_to_drop = ['bedroomcnt', 'taxamount', 
               'taxvaluedollarcnt', 'structuretaxvaluedollarcnt',
               'landtaxvaluedollarcnt', 'yearbuilt', 
               'lotsizesquarefeet','regionidzip']

    train = train.drop(columns = cols_to_drop)
    validate = validate.drop(columns = cols_to_drop)
    test = test.drop(columns = cols_to_drop)
    
    return train, validate, test

def clean_zillow(df):
    modify_columns(df)
    train, validate, test = split_df(df)
    train, validate, test = clean_data(train, validate, test)
    train, validate, test = remove_columns(train, validate, test, cols_to_remove=['buildingqualitytypeid','finishedsquarefeet12','fullbathcnt', 'regionidcounty',"regionidcity",'tdate', 'parcelid', 'propertycountylandusecode'])
    train, validate, test = handle_missing_values(train, validate, test)
    train, validate, test = post_selection_processing(train, validate, test)
    train, validate, test = create_features(train, validate, test)
    train, validate, test = remove_outliers(train, validate, test)
    train, validate, test = col_to_drop_post_processing(train, validate, test)
    #train, validate, test = cat_columns(train, validate, test)
    return train, validate, test  

def catcode_zillow(train, validate, test):
    '''
    This function take train dataset and  categorical variables and splits them in to cat.codes for modeling
    '''
    ############################################################################################
    ############################################################################################
    train["county"] = train["county"].cat.codes
    validate["county"] = validate["county"].cat.codes
    test["county"] = test["county"].cat.codes
    ############################################################################################
    train["heatingorsystemdesc"] = train["heatingorsystemdesc"].cat.codes
    validate["heatingorsystemdesc"] = validate["heatingorsystemdesc"].cat.codes
    test["heatingorsystemdesc"] = test["heatingorsystemdesc"].cat.codes
    ############################################################################################
    train["propertylandusedesc"] = train["propertylandusedesc"].cat.codes
    validate["propertylandusedesc"] = validate["propertylandusedesc"].cat.codes
    test["propertylandusedesc"] = test["propertylandusedesc"].cat.codes
    ############################################################################################
    train["regionidzip"] = train["regionidzip"].cat.codes 
    validate["regionidzip"] = validate["regionidzip"].cat.codes 
    test["regionidzip"] = test["regionidzip"].cat.codes 
    ############################################################################################
    train["yearbuilt"] = train["yearbuilt"].cat.codes 
    validate["yearbuilt"] = validate["yearbuilt"].cat.codes 
    test["yearbuilt"] = test["yearbuilt"].cat.codes  
    
    return train, validate, test

def scale_df(train, validate, test):
    '''
    This function scales data using the MinMaxScaler
    '''
    # Assign variables
    X_train = train
    X_validate = validate
    X_test = test

    # # Might be usefull later
    # scaler = StandardScaler()
    # cols = ['age', 'bmi', 'charges']
    # train_scaled = train.copy()
    # train_scaled[cols] = scaler.fit_transform(train[cols])
    
    # Scale data
    scaler = MinMaxScaler(copy=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_validate_scaled = scaler.transform(X_validate)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns.values).set_index([X_train.index.values])
    X_validate_scaled = pd.DataFrame(X_validate_scaled, columns= X_validate.columns.values).set_index([X_validate.index.values])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns= X_test.columns.values).set_index([X_test.index.values])
    return X_train_scaled, X_validate_scaled, X_test_scaled

def county_scaler(train):
    '''
    Small scaler for county data
    '''
    X_train = train
    # Scale data
    scaler = MinMaxScaler(copy=True).fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns.values).set_index([X_train.index.values])
    return X_train_scaled
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
wrangle.add_upper_outlier_columns(df, k=3)    
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