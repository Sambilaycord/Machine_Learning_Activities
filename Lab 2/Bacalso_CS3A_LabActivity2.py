import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# Load the dataset
# Dataset: Advertising dataset with columns TV, Radio, Newspaper, and Sales
# Source: https://www.kaggle.com/ashydv/advertising-dataset
data = pd.read_csv('advertising.csv')

# Display the first few rows of the dataset
print(data.head())

# Select Feature and Target
feature = 'TV'  
target = 'Sales'  

X = data[[feature]].values  
y = data[target].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Print the shape of the training and test data
print("\nTrain data size: ", X_train.shape, y_train.shape)
print("Test data size: ", X_test.shape, y_test.shape)

# Train a Linear Regression model
lr_model = LinearRegression().fit(X_train, y_train)

print("Linear Regression Model Coefficients: ", lr_model.coef_)
print("Linear Regression Model Intercept: ", lr_model.intercept_)

# Predictions
y_pred = lr_model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("\nLinear Regression Mean Squared Error is: ", mse)

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print("Linear Regression Root Mean Squared Error (RMSE) is: ", rmse)

# Calculate R-Squared
r_squared = r2_score(y_test, y_pred)
print("Linear Regression R-Squared is: ", r_squared)

# Plot regression line for the chosen feature (TV)
plt.figure(figsize=(10, 6))
sns.regplot(x=data[feature], y=data[target], line_kws={"color": "red"})
plt.title(f"Regression Line for {feature}")
plt.xlabel(feature)
plt.ylabel(target)
plt.show()

# Plot Actual vs Predicted values
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.plot(X_test, y_pred, color='green', label='Regression Line')
plt.title(f"Actual vs Predicted values for {feature}")
plt.xlabel(feature)
plt.ylabel(target)
plt.legend()
plt.show()

# Conclusion based on metrics
print("\nConclusion based on Evaluation Metrics:")
if r_squared > 0.8:
    print("The model explains a large proportion of the variance in the target variable. The model is a good fit.")
elif r_squared > 0.5:
    print("The model explains a moderate proportion of the variance in the target variable. The model is okay, but could be improved.")
else:
    print("The model explains a small proportion of the variance in the target variable. The model is not a good fit.")

if mse < 100:
    print("The model's Mean Squared Error is low, indicating better prediction performance.")
else:
    print("The model's Mean Squared Error is high, indicating the predictions are less accurate.")

if rmse < 10:
    print("The Root Mean Squared Error is low, indicating good prediction accuracy.")
else:
    print("The Root Mean Squared Error is high, suggesting poor prediction accuracy.")
