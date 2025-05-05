import os
import pyspark
from pyspark import SparkContext

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip
    
os.environ['SPARK_LOCAL_IP']="" #driver_host
driver_host = get_local_ip()
print(driver_host)

import pyspark
conf = pyspark.SparkConf()
conf.setAppName("spark on MLBP")
conf.setMaster('spark://spark-master:7077')
conf.set("spark.blockmanager.port", "50002")
conf.set("spark.driver.bindAddress", driver_host)
conf.set("spark.driver.host", driver_host)
conf.set("spark.driver.port", "50500")
conf.set("spark.cores.max", "2")
conf.set("spark.executor.memory", "512m")
conf.set('spark.authenticate', False)
