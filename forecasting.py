## Importing the required libraries.
import pandas as pd
import sklearn as sk
import seaborn as sns

data = pd.read_csv('train.csv', parse_dates=['datetime'])

## Exploring Data.
print(data.head())
print(data.info())

##Checking for the null values.
print(data.isnull().sum())

## Feature engineering.
data['hour'] = data['datetime'].dt.hour
data['day'] = data['datetime'].dt.day
data['year'] = data['datetime'].dt.year
data['month'] = data['datetime'].dt.month
data['dayofweek'] = data['datetime'].dt.dayofweek

## dropping unnecessary columns.
data.drop('datetime', axis=1, inplace=True)

## defining features and target
x = data.drop(['count'], axis = 1) #features
y = data['count'] #target


## Spliting the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=42)

from xgboost import XGBRegressor

model = XGBRegressor(objective= 'reg:squarederror', n_estimators = 100, learning_rate = 0.1, max_depth = 6)
model.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error
y_pred = model.predict(x_test)
rmse = mean_squared_error(y_test, y_pred)
print(f'RMSE: {rmse:.2f}')


print(f"Mean of target: {y.mean()}")
print(f"Standard deviation of target: {y.std()}")
print(f"Range of target: {y.min()} to {y.max()}")


## Feature Importance
from xgboost import plot_importance
import matplotlib.pyplot as plt

plot_importance(model)
plt.show()


# Add the missing columns with placeholder values
future_data = pd.DataFrame({
    'season': [1, 2],
    'holiday': [0, 0],
    'workingday': [1, 1],
    'weather': [1, 2],
    'temp': [15.5, 16.3],
    'humidity': [50, 55],
    'windspeed': [5.0, 6.2],
    'hour': [8, 9],
    'day': [15, 15],
    'month': [1, 1],
    'year': [2024, 2024],
    'dayofweek': [2, 2],
    'atemp': [16.0, 16.5],        
    'casual': [0, 0],          
    'registered': [0, 0]        
})

# Ensure columns are in the same order as the model's expected order
expected_columns = [
    'season', 'holiday', 'workingday', 'weather', 'temp', 'atemp', 'humidity', 'windspeed',
    'casual', 'registered', 'hour', 'day', 'year', 'month', 'dayofweek'
]

future_data = future_data[expected_columns]

# Make predictions
future_predictions = model.predict(future_data)
print(future_predictions)




from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix, classification_report

# Define bins for classification (example: low, medium, high demand)
threshold1 = y.quantile(0.33)  # First tercile
threshold2 = y.quantile(0.66)  # Second tercile

# Categorize the actual values
y_train_class = pd.cut(y_train, bins=[-1, threshold1, threshold2, y.max()], labels=[0, 1, 2])
y_test_class = pd.cut(y_test, bins=[-1, threshold1, threshold2, y.max()], labels=[0, 1, 2])

# Categorize the predicted values
y_pred_class = pd.cut(y_pred, bins=[-1, threshold1, threshold2, y.max()], labels=[0, 1, 2])

# Compute precision, recall, and accuracy
precision = precision_score(y_test_class, y_pred_class, average='weighted')
recall = recall_score(y_test_class, y_pred_class, average='weighted')
accuracy = accuracy_score(y_test_class, y_pred_class)

# Print metrics
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'Accuracy: {accuracy:.2f}')

# Confusion Matrix and Classification Report
print("Confusion Matrix:")
print(confusion_matrix(y_test_class, y_pred_class))
print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))



plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label = 'Actual', alpha = 0.7)
plt.plot(y_pred, label = 'Predicted', alpha = 0.7)
plt.legend()
plt.title('Actual vs Predicted Bike Rentals')
plt.show()


## scatterplot better at showing the clear difference between actual and predicted.


plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='b')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Add a line for perfect predictions
plt.xlabel('Actual Bike Rentals')
plt.ylabel('Predicted Bike Rentals')
plt.title('Actual vs Predicted Bike Rentals')
plt.show()



## residual plot 
'''

# Calculate residuals (differences between actual and predicted values)
residuals = y_test - y_pred

# Create a residual plot
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.7, color='b')
plt.axhline(y=0, color='r', linestyle='--', lw=2)  # Horizontal line at y=0 for reference
plt.xlabel('Predicted Bike Rentals')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residuals Plot for Bike Rentals')
plt.show()


'''
