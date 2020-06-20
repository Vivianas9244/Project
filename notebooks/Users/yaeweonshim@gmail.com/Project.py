# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

dbutils.library.installPyPI("mlflow")
dbutils.library.restartPython()
import mlflow

# COMMAND ----------

# MAGIC %sh sudo apt-get install -y graphviz

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

# MAGIC %scala 
# MAGIC val df2 = spark.read.format("csv")
# MAGIC .option("header", "true")
# MAGIC .option("inferSchema", "true")
# MAGIC .load("/FileStore/tables/HeartDisease.csv")
# MAGIC 
# MAGIC display(df2)

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

# With this registered as a temp view, it will only be available to this particular notebook. If you'd like other users to be able to query this table, you can also create a table from the DataFrame.
# Once saved, this table will persist across cluster restarts as well as allow various users across different notebooks to query this data.
# To do so, choose your table name and uncomment the bottom line.

permanent_table_name = "heartdisease_csv"

# df.write.format("parquet").saveAsTable(permanent_table_name)

# COMMAND ----------

print("Our dataset has %d rows." % df.count())

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC df2.count()
# MAGIC df2.dtypes

# COMMAND ----------

import pandas as pd
dt = df.toPandas()

# COMMAND ----------

dt.corr()

# COMMAND ----------

dt.columns = ['age', 'sex', 'chest_pain_type', 'resting_blood_pressure', 'cholesterol', 'fasting_blood_sugar', 'rest_ecg', 'max_heart_rate_achieved',
       'exercise_induced_angina', 'st_depression', 'st_slope', 'num_major_vessels', 'thalassemia', 'target']

# COMMAND ----------

dt['sex'][dt['sex'] == 0] = 'female'
dt['sex'][dt['sex'] == 1] = 'male'

dt['chest_pain_type'][dt['chest_pain_type'] == 0] = 'typical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 1] = 'atypical angina'
dt['chest_pain_type'][dt['chest_pain_type'] == 2] = 'non-anginal pain'
dt['chest_pain_type'][dt['chest_pain_type'] == 3] = 'asymptomatic'

dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
dt['fasting_blood_sugar'][dt['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

dt['rest_ecg'][dt['rest_ecg'] == 0] = 'normal'
dt['rest_ecg'][dt['rest_ecg'] == 1] = 'ST-T wave abnormality'
dt['rest_ecg'][dt['rest_ecg'] == 2] = 'left ventricular hypertrophy'

dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 0] = 'no'
dt['exercise_induced_angina'][dt['exercise_induced_angina'] == 1] = 'yes'

dt['st_slope'][dt['st_slope'] == 1] = 'upsloping'
dt['st_slope'][dt['st_slope'] == 2] = 'flat'
dt['st_slope'][dt['st_slope'] == 3] = 'downsloping'

dt['thalassemia'][dt['thalassemia'] == 1] = 'normal'
dt['thalassemia'][dt['thalassemia'] == 2] = 'fixed defect'
dt['thalassemia'][dt['thalassemia'] == 3] = 'reversable defect'

# COMMAND ----------

dt.dtypes

# COMMAND ----------

dt['sex'] = dt['sex'].astype('object')
dt['chest_pain_type'] = dt['chest_pain_type'].astype('object')
dt['fasting_blood_sugar'] = dt['fasting_blood_sugar'].astype('object')
dt['rest_ecg'] = dt['rest_ecg'].astype('object')
dt['exercise_induced_angina'] = dt['exercise_induced_angina'].astype('object')
dt['st_slope'] = dt['st_slope'].astype('object')
dt['thalassemia'] = dt['thalassemia'].astype('object')

# COMMAND ----------

dt.dtypes

# COMMAND ----------

dt = pd.get_dummies(dt, drop_first=True)

# COMMAND ----------

dt.head()

# COMMAND ----------

df.stat.corr("age","target")

# COMMAND ----------


display(df.describe())


# COMMAND ----------

# Split the dataset randomly into 70% for training and 30% for testing. 
train, test = df.randomSplit([0.7, 0.3], seed = 0)
(train.count(), test.count())
print("We have %d training examples and %d test examples." % (train.count(), test.count()))



# COMMAND ----------

display(train)

# COMMAND ----------

display(test)

# COMMAND ----------

display(train.select("target", "age"))

# COMMAND ----------

display(train.select("sex", "target"))
#sex: The person's sex (1 = male, 0 = female)

# COMMAND ----------

display(train.select("target", "cp"))
#cp: The chest pain experienced (Value 0: typical angina, Value 1: atypical angina, Value 2: non-anginal pain, Value 3: asymptomatic)

# COMMAND ----------

display(train.select("target", "trestbps"))

# COMMAND ----------

display(train.select("target", "chol"))

# COMMAND ----------

display(train.select("target", "fbs"))

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, VectorIndexer
featuresCols = df.columns
featuresCols.remove('target')
# This concatenates all feature columns into a single feature vector in a new column "rawFeatures".
vectorAssembler = VectorAssembler(inputCols=featuresCols, outputCol="rawFeatures")
# This identifies categorical features and indexes them.
vectorIndexer = VectorIndexer(inputCol="rawFeatures", outputCol="features", maxCategories=4)


# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
# ml.classification import decisiontree
# Takes the "features" column and learns to predict "target"
gbt = GBTRegressor(labelCol="target")

# COMMAND ----------

from pyspark.ml.regression import GBTRegressor
# ml.classification import decisiontree
# Takes the "features" column and learns to predict "target"
gbt = GBTRegressor(labelCol="target")

# COMMAND ----------

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import RegressionEvaluator
# Define a grid of hyperparameters to test:
#  - maxDepth: max depth of each decision tree in the GBT ensemble
#  - maxIter: iterations, i.e., number of trees in each GBT ensemble
# In this example notebook, we keep these values small.  In practice, to get the highest accuracy, you would likely want to try deeper trees (10 or higher) and more trees in the ensemble (>100).
paramGrid = ParamGridBuilder()\
  .addGrid(gbt.maxDepth, [2, 5])\
  .addGrid(gbt.maxIter, [10, 100])\
  .build()
# We define an evaluation metric.  This tells CrossValidator how well we are doing by comparing the true labels with predictions.
evaluator = RegressionEvaluator(metricName="rmse", labelCol=gbt.getLabelCol(), predictionCol=gbt.getPredictionCol())
# Declare the CrossValidator, which runs model tuning for us.
cv = CrossValidator(estimator=gbt, evaluator=evaluator, estimatorParamMaps=paramGrid)

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

# MAGIC %scala
# MAGIC val training = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-train.txt")
# MAGIC val test = spark.read.format("libsvm").load("/databricks-datasets/mnist-digits/data-001/mnist-digits-test.txt")
# MAGIC 
# MAGIC // Cache data for multiple uses.
# MAGIC training.cache()
# MAGIC test.cache()
# MAGIC 
# MAGIC println(s"We have ${training.count} training images and ${test.count} test images.")

# COMMAND ----------

# MAGIC %scala
# MAGIC import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
# MAGIC import org.apache.spark.ml.feature.StringIndexer
# MAGIC import org.apache.spark.ml.Pipeline

# COMMAND ----------

# MAGIC %scala
# MAGIC // StringIndexer: Read input column "label" (digits) and annotate them as categorical values.
# MAGIC val indexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
# MAGIC // DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column.
# MAGIC val dtc = new DecisionTreeClassifier().setLabelCol("indexedLabel")
# MAGIC // Chain indexer + dtc together into a single ML Pipeline.
# MAGIC val pipeline = new Pipeline().setStages(Array(indexer, dtc))

# COMMAND ----------

# MAGIC %scala
# MAGIC val model = pipeline.fit(training)

# COMMAND ----------

# MAGIC %scala
# MAGIC // Import the ML algorithms we will use.
# MAGIC import org.apache.spark.ml.classification.{DecisionTreeClassifier, DecisionTreeClassificationModel}
# MAGIC import org.apache.spark.ml.feature.StringIndexer
# MAGIC import org.apache.spark.ml.Pipeline

# COMMAND ----------

# MAGIC %scala
# MAGIC // StringIndexer: Read input column "label" (digits) and annotate them as categorical values.
# MAGIC val indexer = new StringIndexer().setInputCol("label").setOutputCol("indexedLabel")
# MAGIC // DecisionTreeClassifier: Learn to predict column "indexedLabel" using the "features" column.
# MAGIC val dtc = new DecisionTreeClassifier().setLabelCol("indexedLabel")
# MAGIC // Chain indexer + dtc together into a single ML Pipeline.
# MAGIC val pipeline = new Pipeline().setStages(Array(indexer, dtc))

# COMMAND ----------

# MAGIC %scala
# MAGIC val model = pipeline.fit(training)

# COMMAND ----------

# MAGIC %scala
# MAGIC val tree = model.stages.last.asInstanceOf[DecisionTreeClassificationModel]
# MAGIC display(tree)

# COMMAND ----------

# Separating features(X) and target(y)
X = dt.drop('target', axis=1)

# COMMAND ----------

print(X)

# COMMAND ----------

y = dt['target']
print(y)

# COMMAND ----------



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

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# COMMAND ----------

y_pred = classifier.predict(X_test)

# COMMAND ----------

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# COMMAND ----------

from sklearn.ensemble import RandomForestClassifier #for the model
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz #plot tree
from sklearn.metrics import roc_curve, auc #for model evaluation
from sklearn.metrics import classification_report #for model evaluation
from sklearn.metrics import confusion_matrix #for model evaluation
from sklearn.model_selection import train_test_split #for data splitting
import os
import subprocess
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)

# COMMAND ----------

estimator = model.estimators_[1]
feature_names = [i for i in X_train.columns]

y_train_str = y_train.astype('str')
y_train_str[y_train_str == '0'] = 'no disease'
y_train_str[y_train_str == '1'] = 'disease'
y_train_str = y_train_str.values

# COMMAND ----------

export_graphviz(estimator, out_file='tree.dot', 
                feature_names = feature_names,
                class_names = y_train_str,
                rounded = True, proportion = True, 
                label='root',
                precision = 2, filled = True)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

from IPython.display import Image
display(Image(filename = 'tree.png'))



# COMMAND ----------

# MAGIC %sh sudo apt-get install -y graphviz libgraphviz-dev

# COMMAND ----------

from pyspark.ml.image import ImageSchema
image_df = ImageSchema.readImages('tree.png')
display(image_df)

# COMMAND ----------


