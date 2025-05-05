from pyspark import SparkConf, SparkContext

def get_local_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    ip = s.getsockname()[0]
    s.close()
    return ip


def get_current_namespace():
    with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r') as f:
        return f.read().strip()

def start_spark_context():
    namespace=get_current_namespace()
    print('namespace=',namespace)
    driver_host=get_local_ip()
    print('driver_host=',driver_host)

    conf = SparkConf()
    conf.setAppName("myapp")
    conf.setMaster(f"spark://spark-master-0.spark-headless.{namespace}.svc.cluster.local:7077") 
    conf.set("spark.driver.host", driver_host);

    sc = SparkContext(conf=conf)
    return sc
