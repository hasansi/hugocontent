---
title: "Linear Regression"
draft: false
---
# Logistic Regression 

## Binary Customer Churn

A marketing agency has many customers that use their service to produce ads for the client/customer websites. They've noticed that they have quite a bit of churn in clients(i.e customer stopped buying products). They basically randomly assign account managers right now, but want you to create a machine learning model that will help predict which customers will churn (stop buying their service) so that they can correctly assign the customers most at risk to churn an account manager. They have some historical data that can be of help. In this project, I will create a classification algorithm that will help classify whether or not a customer churned. Then the company can test this against incoming data for future customers to predict which customers will churn and assign them an account manager.

The data is saved as customer_churn.csv. Here are the fields and their definitions:

    Name : Name of the latest contact at Company
    Age: Customer Age
    Total_Purchase: Total Ads Purchased
    Account_Manager: Binary 0=No manager, 1= Account manager assigned
    Years: Totaly Years as a customer
    Num_sites: Number of websites that use the service.
    Onboard_date: Date that the name of the latest contact was onboarded
    Location: Client HQ Address
    Company: Name of Client Company
    
Once the model is created and evaluated, I will test it out on some new data in 'new_data.csv'. The client wants to know which customers are most likely to churn given this data (they don't have the label yet).


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
data = spark.read.csv('data/customer_churn.csv',inferSchema=True,
                     header=True)
```


```python
data.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- Churn: integer (nullable = true)
    


### Check out the data


```python
data.describe().show()
```

    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    |summary|        Names|              Age|   Total_Purchase|   Account_Manager|            Years|         Num_Sites|            Location|             Company|              Churn|
    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    |  count|          900|              900|              900|               900|              900|               900|                 900|                 900|                900|
    |   mean|         null|41.81666666666667|10062.82403333334|0.4811111111111111| 5.27315555555555| 8.587777777777777|                null|                null|0.16666666666666666|
    | stddev|         null|6.127560416916251|2408.644531858096|0.4999208935073339|1.274449013194616|1.7648355920350969|                null|                null| 0.3728852122772358|
    |    min|   Aaron King|             22.0|            100.0|                 0|              1.0|               3.0|00103 Jeffrey Cre...|     Abbott-Thompson|                  0|
    |    max|Zachary Walsh|             65.0|         18026.01|                 1|             9.15|              14.0|Unit 9800 Box 287...|Zuniga, Clark and...|                  1|
    +-------+-------------+-----------------+-----------------+------------------+-----------------+------------------+--------------------+--------------------+-------------------+
    



```python
data.columns
```




    ['Names',
     'Age',
     'Total_Purchase',
     'Account_Manager',
     'Years',
     'Num_Sites',
     'Onboard_date',
     'Location',
     'Company',
     'Churn']



### Format for MLlib

We'll ues the numerical columns. We'll include Account Manager because its easy enough, but keep in mind it probably won't be any sort of a signal because the agency mentioned its randomly assigned!


```python
from pyspark.ml.feature import VectorAssembler
```


```python
assembler = VectorAssembler(inputCols=['Age',
 'Total_Purchase',
 'Account_Manager',
 'Years',
 'Num_Sites'],outputCol='features')
```


```python
output = assembler.transform(data)
```


```python
final_data = output.select('features','churn')
```

### Test Train Split


```python
train_churn,test_churn = final_data.randomSplit([0.7,0.3])
```

### Fit the model


```python
from pyspark.ml.classification import LogisticRegression
```


```python
lr_churn = LogisticRegression(labelCol='churn')
```


```python
fitted_churn_model = lr_churn.fit(train_churn)
```


```python
training_sum = fitted_churn_model.summary
```


```python
training_sum.predictions.describe().show()
```

    +-------+-------------------+-------------------+
    |summary|              churn|         prediction|
    +-------+-------------------+-------------------+
    |  count|                626|                626|
    |   mean|0.16134185303514376|0.11501597444089456|
    | stddev|0.36814013167477516| 0.3192963509725862|
    |    min|                0.0|                0.0|
    |    max|                1.0|                1.0|
    +-------+-------------------+-------------------+
    


### Evaluate results

Let's evaluate the results on the data set we were given (using the test data)


```python
from pyspark.ml.evaluation import BinaryClassificationEvaluator
```


```python
pred_and_labels = fitted_churn_model.evaluate(test_churn)
```


```python
pred_and_labels.predictions.show()
```

    +--------------------+-----+--------------------+--------------------+----------+
    |            features|churn|       rawPrediction|         probability|prediction|
    +--------------------+-----+--------------------+--------------------+----------+
    |[22.0,11254.38,1....|    0|[4.37274911189311...|[0.98754068440807...|       0.0|
    |[26.0,8939.61,0.0...|    0|[5.87476599741785...|[0.99719841936848...|       0.0|
    |[27.0,8628.8,1.0,...|    0|[5.10941120065408...|[0.99399662022887...|       0.0|
    |[28.0,9090.43,1.0...|    0|[1.36337386804470...|[0.79630749253360...|       0.0|
    |[28.0,11245.38,0....|    0|[3.31944459560114...|[0.96508988394945...|       0.0|
    |[30.0,6744.87,0.0...|    0|[3.06883440837757...|[0.95558873223614...|       0.0|
    |[30.0,8403.78,1.0...|    0|[5.65542120933540...|[0.99651369764999...|       0.0|
    |[31.0,5304.6,0.0,...|    0|[2.95142032551799...|[0.95033057407558...|       0.0|
    |[31.0,8688.21,0.0...|    0|[6.20044628785487...|[0.99797558145576...|       0.0|
    |[31.0,8829.83,1.0...|    0|[4.22749539474864...|[0.98562089093317...|       0.0|
    |[31.0,9574.89,0.0...|    0|[2.83142395269194...|[0.94435048192403...|       0.0|
    |[31.0,11743.24,0....|    0|[6.29105384971980...|[0.99815062020441...|       0.0|
    |[32.0,7896.65,0.0...|    0|[2.93498302116579...|[0.94954892595428...|       0.0|
    |[32.0,8011.38,0.0...|    0|[1.68697560091684...|[0.84382600777498...|       0.0|
    |[32.0,9885.12,1.0...|    1|[1.60809147265916...|[0.83314624386906...|       0.0|
    |[33.0,4711.89,0.0...|    0|[5.33886279392834...|[0.99522161488696...|       0.0|
    |[33.0,5738.82,0.0...|    0|[3.74180645302845...|[0.97683796907568...|       0.0|
    |[33.0,10709.39,1....|    0|[5.98128726879954...|[0.99748078934333...|       0.0|
    |[33.0,12249.96,0....|    0|[5.27615072734326...|[0.99491392962866...|       0.0|
    |[33.0,13157.08,1....|    0|[1.47782624981273...|[0.81424402404651...|       0.0|
    +--------------------+-----+--------------------+--------------------+----------+
    only showing top 20 rows
    


### Using AUC


```python
churn_eval = BinaryClassificationEvaluator(rawPredictionCol='prediction',
                                           labelCol='churn')
```


```python
auc = churn_eval.evaluate(pred_and_labels.predictions)#area under curve
```


```python
auc
```




    0.7816780045351474



what is a good AUC value? (https://stats.stackexchange.com/questions/113326/what-is-a-good-auc-for-a-precision-recall-curve)

### Predict on brand new unlabeled data

We still need to evaluate the new_customers.csv file!


```python
final_lr_model = lr_churn.fit(final_data)#fit on all final data, not just train or test
```


```python
new_customers = spark.read.csv('data/new_customers.csv',inferSchema=True,
                              header=True)
```


```python
new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
    



```python
test_new_customers = assembler.transform(new_customers)#using old assembler object to transform and see we get
# features column
```


```python
test_new_customers.printSchema()
```

    root
     |-- Names: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- Total_Purchase: double (nullable = true)
     |-- Account_Manager: integer (nullable = true)
     |-- Years: double (nullable = true)
     |-- Num_Sites: double (nullable = true)
     |-- Onboard_date: timestamp (nullable = true)
     |-- Location: string (nullable = true)
     |-- Company: string (nullable = true)
     |-- features: vector (nullable = true)
    



```python
final_results = final_lr_model.transform(test_new_customers)
```


```python
final_results.select('Company','prediction').show()
```

    +----------------+----------+
    |         Company|prediction|
    +----------------+----------+
    |        King Ltd|       0.0|
    |   Cannon-Benson|       1.0|
    |Barron-Robertson|       1.0|
    |   Sexton-Golden|       1.0|
    |        Wood LLC|       0.0|
    |   Parks-Robbins|       1.0|
    +----------------+----------+
    


Now we know that we should assign Acocunt Managers to Cannon-Benson,Barron-Robertson,Sexton-GOlden, and Parks-Robbins!
