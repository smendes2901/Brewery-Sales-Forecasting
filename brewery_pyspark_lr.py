import argparse
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col, date_format, to_timestamp
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.regression import LinearRegression
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

    # Count of dataset
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

    # Convert categorical columns into one hot encoded for Linear Regression
    categoricalColumns = ["Beer_Style", "SKU", "Location"]

    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(
            inputCol=categoricalCol, outputCol=categoricalCol + "Index"
        )
        encoder = OneHotEncoder(
            inputCols=[stringIndexer.getOutputCol()],
            outputCols=[categoricalCol + "OHE"],
        )
        stages += [stringIndexer, encoder]

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

    assemblerInputs = [c + "OHE" for c in categoricalColumns] + numeric_columns

    # Initialize VectorAssembler
    assembler = VectorAssembler(
        inputCols=assemblerInputs,
        outputCol="features",
    )
    stages += [assembler]

    # We trained on partial data during in the notebook. Over here we split all the datasets 80:20 as we pass in 100% of the data
    # Split the data into train and test
    train_data, test_data = train.randomSplit([0.8, 0.2], seed=42)
    rf = LinearRegression(featuresCol="features", labelCol="Total_Sales")
    stages += [rf]

    pipeline = Pipeline(stages=stages)

    # Train the model
    model = pipeline.fit(train_data)

    # Provide output predictions
    predictions = model.transform(test_data)

    # Initialize evaluator
    evaluator = RegressionEvaluator(
        labelCol="Total_Sales", predictionCol="prediction", metricName="rmse"
    )
    rmse = evaluator.evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE) on test data = {rmse}")

    # Compute RMSE
    evaluator = RegressionEvaluator(
        labelCol="Total_Sales", predictionCol="prediction", metricName="r2"
    )

    # Compute RMSE
    r2 = evaluator.evaluate(predictions)
    print("R Squared on test data = %g" % r2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PySpark Job Arguments")
    parser.add_argument("input_path", type=str, help="Input file path")
    args = parser.parse_args()
    main(args.input_path)
