---
title:"Clustering"
draft:"false"
---
# Clustering Project 

The meta data of each session that the hackers used to connect to the servers of a large technology firm. These are the features of the data:

* 'Session_Connection_Time': How long the session lasted in minutes
* 'Bytes Transferred': Number of MB transferred during session
* 'Kali_Trace_Used': Indicates if the hacker was using Kali Linux
* 'Servers_Corrupted': Number of server corrupted during the attack
* 'Pages_Corrupted': Number of pages illegally accessed
* 'Location': Location attack came from (Probably useless because the hackers used VPNs)
* 'WPM_Typing_Speed': Their estimated typing speed based on session logs.


The technology firm has 3 potential hackers that perpetrated the attack. Their certain of the first two hackers but they aren't very sure if the third hacker was involved or not? Morever it's also known that the hackers trade off attacks. Meaning they should each have roughly the same amount of attacks. Let's figure out how many hackers were involved.


```python
# %load /home/ubuntu/projects/initialize.py
import findspark
findspark.init('/home/ubuntu/spark-2.1.1-bin-hadoop2.7')
import pyspark
from pyspark.sql import SparkSession
app_name='app'
spark=SparkSession.builder.appName(app_name).getOrCreate()

```


```python
from pyspark.ml.clustering import KMeans

# Loads data.
dataset = spark.read.csv("data/hack_data.csv",header=True,inferSchema=True)
```


```python
dataset.head()
```




    Row(Session_Connection_Time=8.0, Bytes Transferred=391.09, Kali_Trace_Used=1, Servers_Corrupted=2.96, Pages_Corrupted=7.0, Location='Slovenia', WPM_Typing_Speed=72.37)




```python
dataset.describe().show()
```

    +-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
    |summary|Session_Connection_Time| Bytes Transferred|   Kali_Trace_Used|Servers_Corrupted|   Pages_Corrupted|   Location|  WPM_Typing_Speed|
    +-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
    |  count|                    334|               334|               334|              334|               334|        334|               334|
    |   mean|     30.008982035928145| 607.2452694610777|0.5119760479041916|5.258502994011977|10.838323353293413|       null|57.342395209580864|
    | stddev|     14.088200614636158|286.33593163576757|0.5006065264451406| 2.30190693339697|  3.06352633036022|       null| 13.41106336843464|
    |    min|                    1.0|              10.0|                 0|              1.0|               6.0|Afghanistan|              40.0|
    |    max|                   60.0|            1330.5|                 1|             10.0|              15.0|   Zimbabwe|              75.0|
    +-------+-----------------------+------------------+------------------+-----------------+------------------+-----------+------------------+
    



```python
dataset.columns
```




    ['Session_Connection_Time',
     'Bytes Transferred',
     'Kali_Trace_Used',
     'Servers_Corrupted',
     'Pages_Corrupted',
     'Location',
     'WPM_Typing_Speed']




```python
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
```


```python
feat_cols = ['Session_Connection_Time', 'Bytes Transferred', 'Kali_Trace_Used',
             'Servers_Corrupted', 'Pages_Corrupted','WPM_Typing_Speed']
```


```python
vec_assembler = VectorAssembler(inputCols = feat_cols, outputCol='features')
```


```python
final_data = vec_assembler.transform(dataset)
```


```python
from pyspark.ml.feature import StandardScaler
```


```python
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)
```


```python
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(final_data)
```


```python
# Normalize each feature to have unit standard deviation.
cluster_final_data = scalerModel.transform(final_data)
```

** Time to find out whether its 2 or 3 hackers! **


```python
kmeans3 = KMeans(featuresCol='scaledFeatures',k=3)
kmeans2 = KMeans(featuresCol='scaledFeatures',k=2)
```


```python
model_k3 = kmeans3.fit(cluster_final_data)
model_k2 = kmeans2.fit(cluster_final_data)
```


```python
wssse_k3 = model_k3.computeCost(cluster_final_data)
wssse_k2 = model_k2.computeCost(cluster_final_data)
```


```python
print("With K=3")
print("Within Set Sum of Squared Errors = " + str(wssse_k3))
print('--'*30)
print("With K=2")
print("Within Set Sum of Squared Errors = " + str(wssse_k2))
```

    With K=3
    Within Set Sum of Squared Errors = 434.1492898715845
    ------------------------------------------------------------
    With K=2
    Within Set Sum of Squared Errors = 601.7707512676716


Not much to be gained from the WSSSE, after all, we would expect that as K increases, the WSSSE decreases. 


```python
for k in range(2,9):
    kmeans = KMeans(featuresCol='scaledFeatures',k=k)
    model = kmeans.fit(cluster_final_data)
    wssse = model.computeCost(cluster_final_data)
    print("With K={}".format(k))
    print("Within Set Sum of Squared Errors = " + str(wssse))
    print('--'*30)
```

    With K=2
    Within Set Sum of Squared Errors = 601.7707512676716
    ------------------------------------------------------------
    With K=3
    Within Set Sum of Squared Errors = 434.1492898715845
    ------------------------------------------------------------
    With K=4
    Within Set Sum of Squared Errors = 267.1336116887891
    ------------------------------------------------------------
    With K=5
    Within Set Sum of Squared Errors = 253.02085829439568
    ------------------------------------------------------------
    With K=6
    Within Set Sum of Squared Errors = 229.43601355424215
    ------------------------------------------------------------
    With K=7
    Within Set Sum of Squared Errors = 213.299036334385
    ------------------------------------------------------------
    With K=8
    Within Set Sum of Squared Errors = 193.97042679220849
    ------------------------------------------------------------


** Nothing definitive can be said with the above, but one key fact that the firm mentioned was that the attacks should be evenly numbered between the hackers. Let's check with the transform and prediction columns that result form this.**


```python
model_k3.transform(cluster_final_data).groupBy('prediction').count().show()
```

    +----------+-----+
    |prediction|count|
    +----------+-----+
    |         1|   84|
    |         2|  167|
    |         0|   83|
    +----------+-----+
    



```python
model_k2.transform(cluster_final_data).groupBy('prediction').count().show()
```

    +----------+-----+
    |prediction|count|
    +----------+-----+
    |         1|  167|
    |         0|  167|
    +----------+-----+
    


________

### So it was 2 hackers, as the count is evenly distributed as required by problem statement!

