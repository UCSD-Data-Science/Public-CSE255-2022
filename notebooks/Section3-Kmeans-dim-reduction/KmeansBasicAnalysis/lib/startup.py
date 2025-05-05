# Prepare python libraries for distribution to executors
# !tar -czvf lib.tgz lib/*.py


# start sparkContext
import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math

import pyspark
from pyspark import SparkContext
from lib import sparkConfig

sc = SparkContext(pyFiles=['lib.tgz'])
#master="local[10]"
print('sparkContext=',sc)
print()

# start sqlContext
from pyspark.sql import *
import pyspark.sql
sqlContext = SQLContext(sc)

#load libraries
import numpy as np
from lib.numpy_pack import packArray,unpackArray,unpackAndScale
from lib.spark_PCA import computeCov
from lib.computeStatistics import *

_figsize=(10,7)

### Load the required libraries

from lib.YearPlotter import YearPlotter
from lib.decomposer import *
from lib.Reconstruction_plots import *

from lib.import_modules import import_modules,modules
import_modules(modules)

# import widgets library
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
import ipywidgets as widgets
print('version of ipwidgets=',widgets.__version__)

import warnings  # Suppress Warnings
warnings.filterwarnings('ignore')

## Change the paths here to account for current location of parquest files
## load measurement and stations dataframe
parquet_root='/datasets/weather/datasets'

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