from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.mllib.evaluation import MulticlassMetrics
import sys
import pyspark.sql.functions as func
import pyspark

spark = SparkSession.builder.getOrCreate()

#Single machine prediction application
conf = SparkConf().setAppName("Wine_Quality_Pred").setMaster("local[1]")

sc = SparkContext.getOrCreate()


#Load trained model
rf = RandomForestClassifier.load("s3://akash.cs643/wineQpredmodel.model")


#Read the input data from csv
df_pyspark = spark.read.format('csv').options(header='true', inferSchema='true', delimiter=';').csv("s3://akash.cs643/ValidationDataset.csv")


#Assign all columns other than 'quality' as the feature columns. 'quality' is lable column.
featureColumns = [col for col in df_pyspark.columns if col != '""""quality"""""']


#Use VectorAssembler, which is a feature transformer that merges multiple columns into a vector column
feature_assembler = VectorAssembler(inputCols=featureColumns, outputCol='features')


#Use Pipeline, A Pipeline chains multiple Transformers and Estimators together to specify an ML workflow
spark_Pipe = Pipeline(stages=[feature_assembler, rf])

fitData = spark_Pipe.fit(df_pyspark)
transformedData = fitData.transform(df_pyspark)
transformedData = transformedData.withColumn("prediction", func.round(transformedData['prediction']))
transformedData = transformedData.withColumn('""""quality"""""', transformedData['""""quality"""""'].cast('double')).withColumnRenamed('""""quality"""""', "label")

results = transformedData.select(['prediction', 'label'])
predictionAndLabels = results.rdd

#Use MulticlassMetrics, evaluator for multiclass classification
rf_metrics = MulticlassMetrics(predictionAndLabels)

#Calculate precision, recall and F1 score (accuracy)
precision = rf_metrics.precision()
recall = rf_metrics.recall()
f1Score = rf_metrics.fMeasure()

print("Final result/statistics: ")
print("=============================")
print("F1 Score  = %s" % f1Score)
print("=============================")