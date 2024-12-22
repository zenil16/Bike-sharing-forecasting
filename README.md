# Bike Rental Prediction using XGBoost

This project demonstrates the process of predicting bike rentals using machine learning techniques, specifically the XGBoost algorithm. The goal is to predict the count of bike rentals based on various features such as weather, temperature, and time-related data.

## Steps

### 1. **Data Import**
   - Load dataset (`train.csv`) with datetime parsing.
   - Link to the required Datasets: https://www.kaggle.com/c/bike-sharing-demand/data?select=sampleSubmission.csv

### 2. **Data Exploration**
   - Display the first few rows and check for null values.
   - Perform basic data cleaning and feature engineering.

### 3. **Feature Engineering**
   - Extract features from the `datetime` column, including hour, day, year, month, and day of the week.
   - Drop the unnecessary `datetime` column.

### 4. **Modeling**
   - Define features (`X`) and target (`y`).
   - Split data into training and testing sets.
   - Train an `XGBRegressor` model to predict bike rentals.
   - Evaluate performance using Root Mean Squared Error (RMSE).

### 5. **Model Evaluation**
   - Calculate performance metrics like precision, recall, and accuracy using classification bins (low, medium, high demand).
   - Visualize Actual vs Predicted bike rentals with scatterplots and residual plots.

### 6. **Feature Importance**
   - Display feature importance from the trained XGBoost model.

### 7. **Prediction on Future Data**
   - Make predictions using placeholder future data (for different conditions).

## Requirements
- Python 3.x
- `pandas`
- `sklearn`
- `seaborn`
- `xgboost`
- `matplotlib`

## Installation
Install the necessary libraries using `pip`:
```bash
pip install pandas scikit-learn seaborn xgboost matplotlib
