from functools import reduce
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, avg, stddev, lag, max, mean, skewness, kurtosis
from pyspark.sql.window import Window
from pyspark.ml.feature import StandardScaler, VectorAssembler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

from src.data.nasa_cmaps_ds import download_nasa_ds

# Initialize Spark session
spark = SparkSession.builder.appName("SparkPipeline").getOrCreate()

cols_schema = [
    StructField("unit_id", IntegerType(), True),
    StructField("cycle", IntegerType(), True),
    StructField("os1", FloatType(), True),
    StructField("os2", FloatType(), True),
    StructField("os3", FloatType(), True),
]

sensor_cols = [f"s{i}" for i in range(1, 22)]

sensor_cols_schema = [StructField(it, FloatType(), True) for it in sensor_cols]

cols_schema.extend(sensor_cols_schema)

schema = StructType(cols_schema)

path = download_nasa_ds()
ds_file_path = str(path / "CMaps" / "train_FD002.txt")

# Load data
df_train = spark.read.csv(ds_file_path, sep=" ", header=False, schema=schema).toDF(
    *(it.name for it in cols_schema)
)

# Add RUL column
windowSpec = Window.partitionBy("unit_id")
df_train = df_train.withColumn("RUL", (max("cycle").over(windowSpec) - col("cycle")))

# Apply KMeans clustering
feature_cols = ["os1", "os2", "os3"]
feature_cols.extend(sensor_cols_schema)
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_train = assembler.transform(df_train)

scaler = StandardScaler(inputCol="features", outputCol="scaled_features")
df_train = scaler.fit(df_train).transform(df_train)

# Split data into train/test sets
unit_ids = df_train.select("unit_id").distinct().rdd.flatMap(lambda x: x).collect()
train_ids = unit_ids[: int(0.8 * len(unit_ids))]
test_ids = unit_ids[int(0.8 * len(unit_ids)) :]

train_df = df_train.filter(col("unit_id").isin(train_ids))
test_df = df_train.filter(col("unit_id").isin(test_ids))

# Train Gradient Boosted Tree Model
gbt = GBTRegressor(featuresCol="features", labelCol="RUL", maxIter=100)
model = gbt.fit(train_df)

# Evaluate model
predictions = model.transform(test_df)
evaluator = RegressionEvaluator(
    labelCol="RUL", predictionCol="prediction", metricName="rmse"
)
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse}")
