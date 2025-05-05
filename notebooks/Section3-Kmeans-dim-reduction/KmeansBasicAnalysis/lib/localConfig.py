import pyspark
conf = pyspark.SparkConf()
conf.setMaster("local[4]")
conf.setAppName("local Spark")
