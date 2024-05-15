import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col, date_format, to_timestamp
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


def main(input_filepath):
    spark = SparkSession.builder.appName("Beer analysis").getOrCreate()

    train = spark.read.csv(
        input_filepath,
        header=True,
        inferSchema=True,
    )

    train.count()

    train = train.dropDuplicates()
    train = train.na.drop()

    train = train.withColumn("Total_Sales", col("Total_Sales").cast(FloatType()))

    train = train.withColumn(
        "Brew_Date", to_timestamp(col("Brew_Date"), "yyyy-MM-dd HH:mm:ss")
    )
    train = (
        train.withColumn("Month", date_format(col("Brew_Date"), "MM"))
        .withColumn("Day", date_format(col("Brew_Date"), "dd"))
        .withColumn("Year", date_format(col("Brew_Date"), "yyyy"))
    )

    categorical_columns = ["Beer_Style", "SKU", "Location"]

    indexers = [
        StringIndexer(inputCol=c, outputCol="{0}_indexed".format(c))
        for c in categorical_columns
    ]

    numeric_columns = [
        "Fermentation_Time",
        "Temperature",
        "pH_Level",
        "Gravity",
        "Alcohol_Content",
        "Bitterness",
        "Color",
        "Volume_Produced",
        "Quality_Score",
        "Brewhouse_Efficiency",
        "Loss_During_Brewing",
        "Loss_During_Fermentation",
        "Loss_During_Bottling_Kegging",
    ]

    assembler_inputs = [c + "_indexed" for c in categorical_columns] + numeric_columns
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    train_data, test_data = train.randomSplit([0.8, 0.2], seed=42)
    rf = DecisionTreeRegressor(featuresCol="features", labelCol="Total_Sales")
    pipeline = Pipeline(stages=indexers + [assembler, rf])

    model = pipeline.fit(train_data)
    predictions = model.transform(test_data)

    predictions.select("prediction", "Total_Sales", "features").show(5)

    evaluator = RegressionEvaluator(
        labelCol="Total_Sales", predictionCol="prediction", metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    evaluator = RegressionEvaluator(
        labelCol="Total_Sales", predictionCol="prediction", metricName="r2"
    )

    r2 = evaluator.evaluate(predictions)
    print("R Squared on test data = %g" % r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark Job Arguments")
    parser.add_argument("input_path", type=str, help="Input file path")
    args = parser.parse_args()
    main(args.input_path)
