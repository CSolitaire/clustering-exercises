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
                select *
                from properties_2017
                join predictions_2017 using (parcelid)
                WHERE transactiondate between '2017-05-01' AND '2017-06-30'
                AND propertylandusetypeid IN ('246','247','248','260','261','262','263','264','265','266','268','269','273','274','275','276','279');
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

def run():
    print("Acquire: downloading raw data files...")
    # Write code here
    print("Acquire: Completed!")
