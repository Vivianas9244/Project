// Databricks notebook source
// MAGIC %fs ls

// COMMAND ----------

// MAGIC %fs ls dbfs:/FileStore/tables

// COMMAND ----------

// MAGIC %scala 
// MAGIC val hd = spark.read.format("csv")
// MAGIC .option("header", "true")
// MAGIC .option("inferSchema", "true")
// MAGIC .load("/FileStore/tables/HeartDisease.csv")
// MAGIC 
// MAGIC display(hd)

// COMMAND ----------

