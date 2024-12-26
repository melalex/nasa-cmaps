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
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
df_train = assembler.transform(df_train)

kmeans = KMeans(k=6, seed=42)
model = kmeans.fit(df_train)
df_train = model.transform(df_train).withColumnRenamed("prediction", "cluster")


# Standardize within clusters
def standardize_within_cluster(df, features, cluster_col="cluster"):
    results = []
    for cluster_id in (
        df.select(cluster_col).distinct().rdd.flatMap(lambda x: x).collect()
    ):
        cluster_data = df.filter(col(cluster_col) == cluster_id)
        assembler = VectorAssembler(inputCols=features, outputCol="feature_vector")
        scaler = StandardScaler(inputCol="feature_vector", outputCol="scaled_features")
        assembled = assembler.transform(cluster_data)
        scaled_df = scaler.fit(assembled).transform(assembled)
        results.append(scaled_df.drop("feature_vector"))
    return reduce(DataFrame.unionAll, results)


df_train = standardize_within_cluster(df_train, sensor_cols)


# Add rolling stats and lag features
def add_roll_and_lag_features(df, sensors, window_size=5, lag_steps=5):
    windowSpec = Window.partitionBy("unit_id").orderBy("cycle")
    for sensor in sensors:
        df = df.withColumn(
            f"{sensor}_mean", avg(sensor).over(windowSpec.rowsBetween(-window_size, 0))
        )
        df = df.withColumn(
            f"{sensor}_std",
            stddev(sensor).over(windowSpec.rowsBetween(-window_size, 0)),
        )
        df = df.withColumn(f"{sensor}_lag", lag(sensor, lag_steps).over(windowSpec))
    return df


df_train = add_roll_and_lag_features(df_train, sensor_cols)

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
