import argparse
from pyspark.sql import SparkSession
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col, date_format, to_timestamp
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline


def main(input_filepath):
    # Initialize spark session
    spark = SparkSession.builder.appName("Beer analysis").getOrCreate()

    # Load dataset and infer schema
    train = spark.read.csv(
        input_filepath,
        header=True,
        inferSchema=True,
    )

    # Count of rows in dataset
    train.count()

    # Drop duplicates and NAs
    train = train.dropDuplicates()
    train = train.na.drop()

    # Cast the target variable to float
    train = train.withColumn("Total_Sales", col("Total_Sales").cast(FloatType()))

    # As we cannot use the datatime value directly, we split it into Year, Month and Day
    train = train.withColumn(
        "Brew_Date", to_timestamp(col("Brew_Date"), "yyyy-MM-dd HH:mm:ss")
    )
    train = (
        train.withColumn("Month", date_format(col("Brew_Date"), "MM"))
        .withColumn("Day", date_format(col("Brew_Date"), "dd"))
        .withColumn("Year", date_format(col("Brew_Date"), "yyyy"))
    )

    # Convert categorical columns to numeric values
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

    # Initialize VectorAssembler
    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

    # We trained on partial data during in the notebook. Over here we split all the datasets 80:20 as we pass in 100% of the data
    # Split the data into train and test
    train_data, test_data = train.randomSplit([0.8, 0.2], seed=42)

    # Initialize Model
    rf = RandomForestRegressor(featuresCol="features", labelCol="Total_Sales")
    pipeline = Pipeline(stages=indexers + [assembler, rf])

    # Train the model
    model = pipeline.fit(train_data)

    # Provide output predictions
    predictions = model.transform(test_data)

    predictions.select("prediction", "Total_Sales", "features").show(5)

    # Initialize evaluator
    evaluator = RegressionEvaluator(
        labelCol="Total_Sales", predictionCol="prediction", metricName="rmse"
    )

    # Compute RMSE
    rmse = evaluator.evaluate(predictions)
    print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    evaluator = RegressionEvaluator(
        labelCol="Total_Sales", predictionCol="prediction", metricName="r2"
    )

    # Compute R Square. Adjusted R square wasn't used as there is no direct implementation in pyspark
    r2 = evaluator.evaluate(predictions)
    print("R Squared on test data = %g" % r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark Job Arguments")
    parser.add_argument("input_path", type=str, help="Input file path")
    args = parser.parse_args()
    main(args.input_path)
