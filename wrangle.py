import os
import env


import numpy as np
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sympy import *
# import pydataset

import seaborn as sns
from sympy.matrices import Matrix
from IPython.display import display

from functools import reduce
from itertools import combinations , product
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
import sklearn.preprocessing
from sklearn.model_selection import train_test_split





def remove_outliers_v2(df, k, col_list):
    ''' remove outliers from a list of columns in a dataframe 
        and return that dataframe
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
  
    print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def get_db_url(db, env_file=env):
    '''
    returns a formatted string ready to utilize as a sql url connection

    args: db: a string literal representing a schema
    env_file: bool: checks to see if there is an env.py present in the cwd

    make sure that if you have an env file that you import it outside of the scope 
    of this function call, otherwise env.user wont mean anything ;)
    '''
    if env_file:
        username, password, host = (env.username, env.password, env.host)
        return f'mysql+pymysql://{username}:{password}@{host}/{db}'
    else:
        return 'yo you need some credentials to access a database usually and I dont want you to type them here.'

def new_zillow_2017():
    schema='zillow'

    query='''
    select
    bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
    from
    properties_2017
    where
    (propertylandusetypeid=261	
    or
    propertylandusetypeid=279	)	

    

    
    
    
    
    '''
    return pd.read_sql(query, get_db_url(schema))










def get_zillow_2017():
    '''
    This function reads in zillow data from Codeup database, writes data to
    a csv file if a local file does not exist, and returns a df.
    '''
    if os.path.isfile('zillow_2017.csv'):
        
        # If csv file exists read in data from csv file.
        df = pd.read_csv('zillow_2017.csv', index_col=0)
        
    else:
        
        # Read fresh data from db into a DataFrame
        df = new_zillow_2017()
        
        # Cache data
        df.to_csv('zillow_2017.csv')
        
    return df


def prep_zillow_2017(k):
    df=get_zillow_2017()
    x1=len(df)
    
    cols=df.columns.to_list()
    
    df=remove_outliers_v2(df=df, k=k, col_list=cols)
    df.drop(columns=['outlier'],inplace=True)

    ## Assuming worst case that each NaN is indepedent we drop one percent of our data, so we will drop all is null

    df.dropna(inplace=True)
   

    ## Actual Percent Change
    
    x=df.yearbuilt.apply(decademap)
    df['decade']=x
    
   
    x2=len(df)
    percentchangeafterdrop=round(((x2-x1)/x1)*100,2)
    meankurt=df.kurt().mean()
    display(print(f'This is our percent change after removing all the outliers and then the nulls:\n {percentchangeafterdrop}%\nmean kurt:\n{meankurt}'))
    return df
    





def decademap(x):
    'makes a decade col when combined with an apply function '
    yearsgrouped=np.arange(1800,2020,10)
    if x >= yearsgrouped[21]:
        decade=21
        return decade
    elif x >= yearsgrouped[20]:
        decade=20
        return decade
    elif x >= yearsgrouped[19]:
        decade=19
        return decade
    elif x >= yearsgrouped[18]:
        decade=18
        return decade
    elif x >= yearsgrouped[17]:
        decade=17
        return decade
    elif x >= yearsgrouped[16]:
        decade=16
        return decade
    elif x >= yearsgrouped[15]:
        decade=15
        return decade
    elif x >= yearsgrouped[14]:
        decade=14
        return decade
    elif x >= yearsgrouped[13]:
        decade=13
        return decade
    elif x >= yearsgrouped[12]:
        decade=12
        return decade
    elif x >= yearsgrouped[11]:
        decade=11
        return decade
    elif x >= yearsgrouped[10]:
        decade=10
        return decade
    elif x >= yearsgrouped[9]:
        decade=9
        return decade
    elif x >= yearsgrouped[8]:
        decade=8
        return decade
    elif x >= yearsgrouped[7]:
        decade=7
        return decade
    elif x >= yearsgrouped[6]:
        decade=6
        return decade
    elif x >= yearsgrouped[5]:
        decade=5
        return decade
    elif x >= yearsgrouped[4]:
        decade=4
        return decade
    elif x >= yearsgrouped[3]:
        decade=3
        return decade
    elif x >= yearsgrouped[2]:
        decade=2
        return decade
    elif x >= yearsgrouped[1]:
        decade=1
        return decade
    elif x >= yearsgrouped[0]:
        decade=0
        return decade






