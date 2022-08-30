from rcm_library import *
from scipy.stats import zscore
from sklearn.impute import SimpleImputer





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


def prep_zillow_2017():
    df=get_zillow_2017()

    ## Assuming worst case that each NaN is indepedent we drop one percent of our data, so we will drop all is null
    x1=len(df)
    df.dropna(inplace=True)
    x2=len(df)
    percentchangeafterdrop=round(((x2-x1)/x1)*100,2)
    print(f'Actual percent chage after droping null rows:\n{percentchangeafterdrop:.2f}%')
    ## Actual Percent Change
    x=df.yearbuilt.apply(decademap)
    df['decade']=x
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
