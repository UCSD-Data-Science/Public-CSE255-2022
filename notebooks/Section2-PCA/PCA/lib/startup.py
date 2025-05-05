# Prepare python libraries for distribution to executors
# !tar -czvf lib.tgz lib/*.py

import sys

if len(sys.argv)<=1:
    sc_type='S'
else:
    sc_type=sys.argv[1]
print('sc_type=',sc_type)

# start sparkContext
import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math

import pyspark
from pyspark import SparkContext
from lib import sparkConfig

from start_spark_context import start_spark_context, get_current_namespace

if sc_type!='S':
    sc = SparkContext('local[10]')
else:
    sc=start_spark_context()

print('sparkContext=',sc)
print()

# start sqlContext
from pyspark.sql import *
import pyspark.sql
sqlContext = SQLContext(sc)
import numpy as np

#load libraries to workers
sc.addPyFile("lib/numpy_pack.py")
sc.addPyFile("lib/spark_PCA.py")
sc.addPyFile("lib/computeStatistics.py")
sc.addPyFile("lib/decomposer.py")

import warnings  # Suppress Warnings
warnings.filterwarnings('ignore')
sc.setLogLevel("ERROR")

_figsize=(10,7)

### Load lib archive
#sc.addArchive("lib.tgz")  # extract directory on all workers

### Load the required libraries

from lib.YearPlotter import YearPlotter
#from lib.decomposer import *
#from lib.Reconstruction_plots import *

#from lib.import_modules import import_modules,modules
#import_modules(modules)

# import widgets library
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
import ipywidgets as widgets
print('version of ipwidgets=',widgets.__version__)

import warnings  # Suppress Warnings
warnings.filterwarnings('ignore')

## Change the paths here to account for current location of parquest files
## load measurement and stations dataframe
ns=get_current_namespace()
parquet_root=f'/home/{ns}/public/Data/weather'
print('parquet_root=',parquet_root)

measurements_path=parquet_root+'/weather-parquet'
measurements=sqlContext.read.parquet(measurements_path)
sqlContext.registerDataFrameAsTable(measurements,'measurements')

print('measurements is a Dataframe (and table) with %d records'%(measurements.count()))

stations_path=parquet_root+'/stations-parquet'
stations=sqlContext.read.parquet(stations_path)
sqlContext.registerDataFrameAsTable(stations,'stations')
print('stations is a Dataframe (and table) with %d records'%(stations.count()))

weather=measurements.join(stations,on='station')
print('weather is a Dataframe (and table) which is a join of measurements and stations with %d records'%(weather.count()))
sqlContext.registerDataFrameAsTable(weather,'weather')
