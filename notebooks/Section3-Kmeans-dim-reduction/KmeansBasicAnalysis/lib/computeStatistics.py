
from numpy import linalg as LA
import numpy as np

from lib.numpy_pack import packArray,unpackArray, unpackAndScale
from lib.spark_PCA import computeCov
from time import time

_measurements=['TMAX', 'SNOW', 'SNWD', 'TMIN', 'PRCP', 'TOBS']
_measurements=_measurements+[x+'_S10' for x in _measurements] + [x+'_S20' for x in _measurements]
_measurements

def computeStatistics(sqlContext,df,measurements=_measurements):
    """Compute all of the statistics for a given dataframe
    Input: sqlContext: to perform SQL queries
            df: dataframe with the fields 
            Station(string), Measurement(string), Year(integer), Values (byteArray with 366 float16 numbers)
    returns: STAT, a dictionary of dictionaries. First key is measurement, 
             second keys described in computeStats.STAT_Descriptions
    """

    sqlContext.registerDataFrameAsTable(df,'compute_weather')
    STAT={}  # dictionary storing the statistics for each measurement
    
    for meas in measurements:
        t=time()
        Query="SELECT * FROM compute_weather\n\tWHERE measurement = '%s'"%(meas)
        mdf = sqlContext.sql(Query)
        mdf_count=mdf.count()
        print(meas,': shape of mdf is ',mdf_count)
        if mdf_count==0:
              continue

        data=mdf.rdd.map(lambda row: unpackAndScale(row))

        #Compute basic statistics
        STAT[meas]=computeOverAllDist(data)   # Compute the statistics 

        # compute covariance matrix
        OUT=computeCov(data)

        #find PCA decomposition
        eigval,eigvec=LA.eig(OUT['Cov'])

        # collect all of the statistics in STAT[meas]
        STAT[meas]['eigval']=eigval
        STAT[meas]['eigvec']=eigvec
        STAT[meas].update(OUT)

        print('time for',meas,'is',time()-t)
    
    return STAT


def find_percentiles(SortedVals,percentile):
    L=int(len(SortedVals)/percentile)
    return SortedVals[L],SortedVals[-L]
  
def computeOverAllDist(rdd0):
    # Compute a sample of the number of nan per year
    UnDef=np.array(rdd0.map(lambda row:sum(np.isnan(row))).sample(False,0.1).collect())
    flat=rdd0.flatMap(lambda v:list(v)).filter(lambda x: not np.isnan(x)).cache()
    # compute first and second order statistics
    count,S1,S2=flat.map(lambda x: np.float64([1,x,x**2]))\
                  .reduce(lambda x,y: x+y)
    mean=S1/count
    std=np.sqrt(S2/count-mean**2)
    
    #Sample 0.01 percent of the values and generate a sorted array that can be used to plot an approximate CDF
    Vals=flat.sample(False,0.0001).collect()
    SortedVals=np.array(sorted(Vals))
    low100,high100=find_percentiles(SortedVals,100)
    low1000,high1000=find_percentiles(SortedVals,1000)
    return {'UnDef':UnDef,\
          'mean':mean,\
          'std':std,\
          'SortedVals':SortedVals,\
          'low100':low100,\
          'high100':high100,\
          'low1000':low1000,\
          'high1000':high1000
          }

# description of data returned by computeOverAllDist
STAT_Descriptions=[
 # distribution of undefined/defined entries
 ('UnDef', 'sample of number of undefs per row', 'vector whose length varies between measurements'),
 ('NE', 'count of defined values per day', (366,)),

 # statistics of scalars, taken over all entries in all vectors
 ('SortedVals', 'Sample of values', 'vector whose length varies between measurements'),
 ('mean', 'mean value', ()),
 ('std', 'std', ()),
 ('low100', 'bottom 1%', ()),
 ('high100', 'top 1%', ()),
 ('low1000', 'bottom 0.1%', ()),
 ('high1000', 'top 0.1%', ()),
    
 # statistics of yearly measurement vectors   
 ('E', 'Sum of values per day', (366,)),
 ('Mean', 'E/NE', (366,)),
 ('O', 'Sum of outer products', (366, 366)),
 ('NO', 'counts for outer products', (366, 366)),
 ('Cov', 'O/NO', (366, 366)),
 ('Var', 'The variance per day = diagonal of Cov', (366,)),
 ('eigval', 'PCA eigen-values', (366,)),
 ('eigvec', 'PCA eigen-vectors', (366, 366))
]
