# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# MAGIC %md # What are the risk factors for heart disease?

# COMMAND ----------

# MAGIC %md ###Import mlflow

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()
import mlflow


# COMMAND ----------

# MAGIC %md ### Importing Libraries & Dataset

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/HeartDisease.csv"
file_type = "csv"

# CSV options
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)

display(df)

# COMMAND ----------

# Create a view or table

temp_table_name = "HeartDisease_csv"

df.createOrReplaceTempView(temp_table_name)

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC /* Query the created temp table in a SQL cell */
# MAGIC 
# MAGIC select * from `HeartDisease_csv`

# COMMAND ----------

# MAGIC %md ### Columns

# COMMAND ----------

### age: The person's age
### sex : 1 = male, 0 = female
### cp : the chest pain experience 
###  trestbps: The person's resting blood pressure
### chol : the person's cholesterol measurement in mg/dl
### fbs: the person's fasting blood sugar (>120mg/dl, 1 = true, 0 =false)
### restecg : resting resting electrocardiographic measurement (0 = normal, 1 = having ST-T wave abnormality, 2 = showing probale or define left ventricular hypertrophy by Estes' criteria)
### thalach : the person's maximum heart rate achieved
### exag: Exercise induced angina ( 1= yes, 0 = no)
### oldpeak: ST depression induced by exercise relative to rest ('ST' related to positions on the ECG plot)
### slope: the slope of the peak exercise ST segment (Value 1: upsloping, Value 2: flat, Value 3:downsloping)
### ca: The number of major vessels (0-3)
### thal: A blood disorder called thalassemia (3= normal, 6= fixed, 7 =reversable defeat)
### target: Heart disease (0=no, 1= yes)

# COMMAND ----------

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "heartdisease_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

print("Our dataset has %d rows." % df.count())

# COMMAND ----------

# MAGIC %md ### Since our file is in CSV, we use panda's read_csv to read CSV data file. 

# COMMAND ----------

import pandas as pd
dt = df.toPandas()

# COMMAND ----------

# MAGIC %md ### Check if this is an imbalanced dataset

# COMMAND ----------

target_balance = dt['target'].value_counts()

# COMMAND ----------

# MAGIC %sql
# MAGIC 
# MAGIC select target from `HeartDisease_csv` count

# COMMAND ----------

# MAGIC %md ###Extract DataFrame correlations

# COMMAND ----------

dt.corr()

# COMMAND ----------

# MAGIC %md ### Visualize correlation matrix using seaborn

# COMMAND ----------

# MAGIC %md ### High correlation between thal(a blood disorder called thalassemia), oldpeak(ST depression), thalach(the person's maximum heart rate archieved) and target(disease or not diseased)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10,10))
corr = dt.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(50, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

# COMMAND ----------

# MAGIC %md ### The average, minimum and maximum of factors

# COMMAND ----------


display(df.describe())


# COMMAND ----------

# MAGIC %md ### Split the dataset randomly into 70% for training and 30% for testing. 

# COMMAND ----------

train, test = df.randomSplit([0.7, 0.3], seed = 0)
(train.count(), test.count())
print("We have %d training examples and %d test examples." % (train.count(), test.count()))



# COMMAND ----------

display(train)

# COMMAND ----------

display(test)

# COMMAND ----------

# MAGIC %md ### Data visualization

# COMMAND ----------

# MAGIC %md ###  Older people are more likely than younger people to suffer from Heart disease.

# COMMAND ----------

display(train.select("target", "age"))

# COMMAND ----------

display(train.select("sex", "target"))
#sex: The person's sex (1 = male, 0 = female)

# COMMAND ----------

# MAGIC %md ### High blood pressure is a risk factor for heart condition

# COMMAND ----------

display(train.select("target", "trestbps"))

# COMMAND ----------

# MAGIC %md ### Going higher than your maximum heart rate for long periods of time could be a risk factor for heart condition

# COMMAND ----------

display(train.select("target", "thalach"))

# COMMAND ----------

# MAGIC %md ### When there is high cholesterol in your blood, it builds up in the walls of your arteries, causing a form of heart disease. 

# COMMAND ----------

display(train.select("target", "chol"))

# COMMAND ----------

# MAGIC %md ###Deeper and more widespread ST depression generally indicates more severe or extensive disease.

# COMMAND ----------

display(train.select("target","oldpeak"))

# COMMAND ----------

# MAGIC %md ### Thalassemia(thal) is a blood disorder that causes heart disease

# COMMAND ----------

display(train.select("target","thal"))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = df.columns
featuresCols.remove('target')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)

# COMMAND ----------

from pyspark.ml.classification import GBTClassifier, DecisionTreeClassifier
# ml.classification import decisiontree
# Takes the "features" column and learns to predict "target"
gbt = GBTClassifier(labelCol="target", maxDepth=3)
#dt = DecisionTreeClassifier(labelCol="target", featuresCol="features", maxDepth=3)setLabelCol()
#dt = DecisionTreeClassifier()

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble
# In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2])\
  .addGrid(gbt.maxIter, [10])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
#evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC", labelCol=gbt.getLabelCol())
#evaluator = BinaryClassificationEvaluator()
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid, numFolds=5)

# COMMAND ----------

from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[vectorAssembler, vectorIndexer, cv])


# COMMAND ----------

pipelineModel = pipeline.fit(train)

# COMMAND ----------

predictions = pipelineModel.transform(test)

# COMMAND ----------

display(predictions.select("target", "prediction", *featuresCols))

# COMMAND ----------

# MAGIC %md ###Training and Making Predictions from sklearn

# COMMAND ----------

# MAGIC %md ### The x contains all the columsn from the dataset except the 'target' Columns. The y varaible contains the value from the 'target' Columns

# COMMAND ----------

# Separating features(X) and target(y)
X = dt.drop('target', axis=1)

# COMMAND ----------

print(X)

# COMMAND ----------

y = dt['target']
print(y)

# COMMAND ----------

# MAGIC %md ### We use split up 35% of the data in to the test set and 65% for training

# COMMAND ----------

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

# COMMAND ----------

print(f"X.shape: {X.shape}, y.shape: {y.shape}")
#original dataset

# COMMAND ----------

print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")
#splited datasets

# COMMAND ----------

# MAGIC %md ### Use the DecisionTreeClassifier to train the algorithm 

# COMMAND ----------

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# COMMAND ----------

y_pred = classifier.predict(X_test)
print(y_pred)

# COMMAND ----------

# MAGIC %md ### Evaluating the Algorithm from sklearn

# COMMAND ----------

# MAGIC %md ### From the confusion matrix, our alogrithm misclassified only 24 out of 107. This is 77% accuracy

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# COMMAND ----------

# MAGIC %md ### Conclusion: It shows that people with heart disease tend to be older, and have higher blood pressure, higher cholesterol levels, deeper and more widespread ST etc., than people without the disease.