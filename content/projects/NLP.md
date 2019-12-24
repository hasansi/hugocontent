---
title: "NLP"
draft: false
---
# Natural Language Processing 

In this project, I'll build a spam filter using various NLP tools as well as the Naive Bayes classifier.

I'll use a classic dataset for this - UCI Repository SMS Spam Detection: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection


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
data = spark.read.csv("data/SMSSpamCollection",inferSchema=True,sep='\t')#seperated by tabs, not comma
```


```python
data = data.withColumnRenamed('_c0','class').withColumnRenamed('_c1','text')#re-label the _c0 and _c1 columns
```


```python
data.show()
```

    +-----+--------------------+
    |class|                text|
    +-----+--------------------+
    |  ham|Go until jurong p...|
    |  ham|Ok lar... Joking ...|
    | spam|Free entry in 2 a...|
    |  ham|U dun say so earl...|
    |  ham|Nah I don't think...|
    | spam|FreeMsg Hey there...|
    |  ham|Even my brother i...|
    |  ham|As per your reque...|
    | spam|WINNER!! As a val...|
    | spam|Had your mobile 1...|
    |  ham|I'm gonna be home...|
    | spam|SIX chances to wi...|
    | spam|URGENT! You have ...|
    |  ham|I've been searchi...|
    |  ham|I HAVE A DATE ON ...|
    | spam|XXXMobileMovieClu...|
    |  ham|Oh k...i'm watchi...|
    |  ham|Eh u remember how...|
    |  ham|Fine if thats th...|
    | spam|England v Macedon...|
    +-----+--------------------+
    only showing top 20 rows
    


## Clean and Prepare the Data

** Create a new length feature: **


```python
from pyspark.sql.functions import length
```


```python
data = data.withColumn('length',length(data['text']))
```


```python
data.show()
```

    +-----+--------------------+------+
    |class|                text|length|
    +-----+--------------------+------+
    |  ham|Go until jurong p...|   111|
    |  ham|Ok lar... Joking ...|    29|
    | spam|Free entry in 2 a...|   155|
    |  ham|U dun say so earl...|    49|
    |  ham|Nah I don't think...|    61|
    | spam|FreeMsg Hey there...|   147|
    |  ham|Even my brother i...|    77|
    |  ham|As per your reque...|   160|
    | spam|WINNER!! As a val...|   157|
    | spam|Had your mobile 1...|   154|
    |  ham|I'm gonna be home...|   109|
    | spam|SIX chances to wi...|   136|
    | spam|URGENT! You have ...|   155|
    |  ham|I've been searchi...|   196|
    |  ham|I HAVE A DATE ON ...|    35|
    | spam|XXXMobileMovieClu...|   149|
    |  ham|Oh k...i'm watchi...|    26|
    |  ham|Eh u remember how...|    81|
    |  ham|Fine if thats th...|    56|
    | spam|England v Macedon...|   155|
    +-----+--------------------+------+
    only showing top 20 rows
    



```python
# Pretty Clear Difference
data.groupby('class').mean().show()
```

    +-----+-----------------+
    |class|      avg(length)|
    +-----+-----------------+
    |  ham|71.45431945307645|
    | spam|138.6706827309237|
    +-----+-----------------+
    


## Feature Transformations


```python
from pyspark.ml.feature import Tokenizer,StopWordsRemover, CountVectorizer,IDF,StringIndexer

tokenizer = Tokenizer(inputCol="text", outputCol="token_text")
stopremove = StopWordsRemover(inputCol='token_text',outputCol='stop_tokens')
count_vec = CountVectorizer(inputCol='stop_tokens',outputCol='c_vec')
idf = IDF(inputCol="c_vec", outputCol="tf_idf")
ham_spam_to_num = StringIndexer(inputCol='class',outputCol='label')
```


```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.linalg import Vector
```


```python
clean_up = VectorAssembler(inputCols=['tf_idf','length'],outputCol='features')
```

### The Model

I'll use Naive Bayes.


```python
from pyspark.ml.classification import NaiveBayes
```


```python
# Use defaults
nb = NaiveBayes()
```

### Pipeline


```python
from pyspark.ml import Pipeline
```


```python
data_prep_pipe = Pipeline(stages=[ham_spam_to_num,tokenizer,stopremove,count_vec,idf,clean_up])
```


```python
cleaner = data_prep_pipe.fit(data)
```


```python
clean_data = cleaner.transform(data)
```

### Training and Evaluation!


```python
clean_data = clean_data.select(['label','features'])
```


```python
clean_data.show()
```

    +-----+--------------------+
    |label|            features|
    +-----+--------------------+
    |  0.0|(13459,[8,12,33,6...|
    |  0.0|(13459,[0,26,308,...|
    |  1.0|(13459,[2,14,20,3...|
    |  0.0|(13459,[0,73,84,1...|
    |  0.0|(13459,[36,39,140...|
    |  1.0|(13459,[11,57,62,...|
    |  0.0|(13459,[11,55,108...|
    |  0.0|(13459,[133,195,4...|
    |  1.0|(13459,[1,50,124,...|
    |  1.0|(13459,[0,1,14,29...|
    |  0.0|(13459,[5,19,36,4...|
    |  1.0|(13459,[9,18,40,9...|
    |  1.0|(13459,[14,32,50,...|
    |  0.0|(13459,[42,99,101...|
    |  0.0|(13459,[567,1745,...|
    |  1.0|(13459,[32,113,11...|
    |  0.0|(13459,[86,224,47...|
    |  0.0|(13459,[0,2,52,13...|
    |  0.0|(13459,[0,77,107,...|
    |  1.0|(13459,[4,32,35,6...|
    +-----+--------------------+
    only showing top 20 rows
    



```python
(training,testing) = clean_data.randomSplit([0.7,0.3])
```


```python
spam_predictor = nb.fit(training)
```


```python
data.printSchema()
```

    root
     |-- class: string (nullable = true)
     |-- text: string (nullable = true)
     |-- length: integer (nullable = true)
    



```python
test_results = spam_predictor.transform(testing)
```


```python
test_results.show()
```

    +-----+--------------------+--------------------+--------------------+----------+
    |label|            features|       rawPrediction|         probability|prediction|
    +-----+--------------------+--------------------+--------------------+----------+
    |  0.0|(13459,[0,1,2,8,9...|[-796.45570905799...|[1.0,2.0631345277...|       0.0|
    |  0.0|(13459,[0,1,6,21,...|[-804.54113783103...|[1.0,1.3341950705...|       0.0|
    |  0.0|(13459,[0,1,10,15...|[-563.39905288050...|[1.0,2.4952008385...|       0.0|
    |  0.0|(13459,[0,1,10,15...|[-563.39905288050...|[1.0,2.4952008385...|       0.0|
    |  0.0|(13459,[0,1,15,19...|[-1357.2559414407...|[1.0,1.0068429225...|       0.0|
    |  0.0|(13459,[0,1,18,20...|[-823.31614992105...|[1.0,1.9285150162...|       0.0|
    |  0.0|(13459,[0,1,24,29...|[-1016.4133071261...|[1.0,7.9763655949...|       0.0|
    |  0.0|(13459,[0,1,32,12...|[-620.25440638973...|[1.0,1.9325801586...|       0.0|
    |  0.0|(13459,[0,1,175,4...|[-166.46140413074...|[0.99999999999989...|       0.0|
    |  0.0|(13459,[0,1,896,1...|[-94.412416594565...|[0.99999999549428...|       0.0|
    |  0.0|(13459,[0,2,3,4,7...|[-1232.1013660210...|[1.0,5.2353504400...|       0.0|
    |  0.0|(13459,[0,2,4,6,1...|[-2489.9094686514...|[1.0,4.2272217941...|       0.0|
    |  0.0|(13459,[0,2,4,9,3...|[-554.32529276773...|[1.0,3.2152137527...|       0.0|
    |  0.0|(13459,[0,2,4,11,...|[-1232.8706060715...|[1.0,4.4591980406...|       0.0|
    |  0.0|(13459,[0,2,4,135...|[-637.33782358223...|[1.0,8.1455549609...|       0.0|
    |  0.0|(13459,[0,2,5,15,...|[-1088.0906843342...|[1.0,1.3326413850...|       0.0|
    |  0.0|(13459,[0,2,5,25,...|[-847.71644930059...|[0.99999998773768...|       0.0|
    |  0.0|(13459,[0,2,5,28,...|[-773.10395512111...|[1.0,9.3010213408...|       0.0|
    |  0.0|(13459,[0,2,5,74,...|[-788.42935451362...|[1.0,6.2568262707...|       0.0|
    |  0.0|(13459,[0,2,8,9,1...|[-701.37591094076...|[1.0,2.6017003391...|       0.0|
    +-----+--------------------+--------------------+--------------------+----------+
    only showing top 20 rows
    



```python
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
```


```python
acc_eval = MulticlassClassificationEvaluator()
acc = acc_eval.evaluate(test_results)
print("Accuracy of model at predicting spam was: {}".format(acc))
```

    Accuracy of model at predicting spam was: 0.920827783177929

