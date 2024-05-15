# Brewery Sales Forecasting

## Project Overview
This project involves forecasting total sales for a brewery using a dataset from Kaggle. The models used for forecasting include Linear Regression, Random Forest, and Decision Tree. The project has been executed and tested on Google Cloud Platform's DataProc service.

## Dataset
The dataset used in this project can be found at [Kaggle Brewery Operations and Market Analysis Dataset](https://www.kaggle.com/datasets/ankurnapa/brewery-operations-and-market-analysis-dataset/). It provides extensive data related to brewery operations and market analysis, which is suitable for developing predictive models.

## Requirements
- Google Cloud Platform account
- Access to GCP DataProc
- Apache Spark
- PySpark
- Python 3.x

## Installation and Setup
1. **Set up GCP DataProc Cluster:**
   Ensure that you have a GCP account and create a DataProc cluster to run PySpark jobs.

2. **Install PySpark:**
   ```bash
   pip install pyspark
    ```
## Download the Dataset:
Download the dataset from [Kaggle](https://www.kaggle.com/datasets/ankurnapa/brewery-operations-and-market-analysis-dataset/) and upload it to a bucket in Google Cloud Storage accessible by your DataProc cluster.

## Models Used

* Linear Regression: A basic model for establishing a baseline in forecasting performance.
* Random Forest: An ensemble model that uses multiple decision trees to improve the predictive accuracy and control over-fitting.
*  Decision Tree: A model that splits the data into subsets while at the same time developing a corresponding decision tree. The final decision tree can be used to make predictions.

## Results

Discuss the results from each model, including metrics like RMSE, MAE, or any other relevant evaluation metric used to assess the performance of the models.