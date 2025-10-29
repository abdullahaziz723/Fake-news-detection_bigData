# ----------------------------------------------------------
# Fake News Detection using PySpark (Big Data Pipeline)
# ----------------------------------------------------------

from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws, col, lower, regexp_replace
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
import pyspark.sql.functions as F

import os
os.environ["PYSPARK_PYTHON"] = "python"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python"
# ----------------------------
# Step 1: Initialize Spark
# ----------------------------
spark = SparkSession.builder \
    .appName("FakeNewsDetectionBigData") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

print("✅ Spark session started successfully!")

# ----------------------------
# Step 2: Load Dataset
# ----------------------------
CSV_PATH = "dataset/combined_news.csv"  # make sure dataset exists

df = spark.read.option("header", True).csv(CSV_PATH)
print(f"✅ Dataset loaded successfully with {df.count()} rows and {len(df.columns)} columns.")

# ----------------------------
# Step 3: Standardize Columns
# ----------------------------
# Normalize names
for c in df.columns:
    cl = c.lower()
    if cl == "title":
        df = df.withColumnRenamed(c, "title")
    elif cl == "text":
        df = df.withColumnRenamed(c, "text")
    elif cl == "label":
        df = df.withColumnRenamed(c, "label")

# Create content column
df = df.withColumn("content", concat_ws(" ", col("title"), col("text")))

# ----------------------------
# Step 4: Clean Text Data
# ----------------------------
df = df.withColumn("content", lower(col("content")))
df = df.withColumn("content", regexp_replace(col("content"), r"http\S+|www\.\S+", " "))
df = df.withColumn("content", regexp_replace(col("content"), r"[^a-z\s]", " "))
df = df.na.drop(subset=["content"])

# ----------------------------
# Step 5: Safely handle label column
# ----------------------------
# Some datasets may have strings like "FAKE" or "TRUE" instead of numbers.
# Let's convert them safely.
df = df.withColumn("label", lower(col("label")))

# Map 'fake' → 0 and 'true' → 1 if they are words
df = df.withColumn(
    "label",
    F.when(col("label").isin("fake", "0"), F.lit(0))
     .when(col("label").isin("true", "1"), F.lit(1))
     .otherwise(None)
)

# Remove rows with invalid labels
df = df.filter(col("label").isNotNull())
df = df.withColumn("label", col("label").cast("int"))

print(f"✅ Cleaned dataset ready: {df.count()} rows")

# ----------------------------
# Step 6: Build ML Pipeline
# ----------------------------
tokenizer = Tokenizer(inputCol="content", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=1 << 18)
idf = IDF(inputCol="rawFeatures", outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, idf, lr])

# ----------------------------
# Step 7: Train-Test Split
# ----------------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

print(f"Training data: {train.count()} rows | Testing data: {test.count()} rows")

# ----------------------------
# Step 8: Train Model
# ----------------------------
model = pipeline.fit(train)
print("✅ Model training completed!")

# ----------------------------
# Step 9: Evaluate
# ----------------------------
predictions = model.transform(test).select("label", "prediction", "probability")

# Binary Evaluation (AUC)
binary_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="probability", metricName="areaUnderROC")
auc = binary_eval.evaluate(predictions)

# Multiclass Evaluation (F1)
multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1 = multi_eval.evaluate(predictions)

print(f"\n📊 Model Evaluation Results:")
print(f"AUC Score: {auc:.4f}")
print(f"F1 Score: {f1:.4f}")

# ----------------------------
# Step 10: Save Model
# ----------------------------
model.write().overwrite().save("spark_fake_news_model")
print("✅ Model saved to 'spark_fake_news_model' folder.")

# ----------------------------
# Step 11: Example Prediction
# ----------------------------
sample_data = [
    ("Breaking News! NASA discovers water on Mars.",),
    ("Politician caught spreading false statements online.",)
]
sample_df = spark.createDataFrame(sample_data, ["content"])
sample_pred = model.transform(sample_df)
sample_pred.select("content", "prediction").show(truncate=False)

# ----------------------------
# Step 12: Stop Spark
# ----------------------------
spark.stop()
print("\n🏁 Spark session stopped successfully.")
