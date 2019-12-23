import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
app_name='app'
spark=SparkSession.builder.appName(app_name).getOrCreate()
