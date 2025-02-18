#Krystsal Heart M. Bacalso
#CS3A

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('LoanData_Raw_v1.0.csv')

# Remove duplicates
data_cleaned = df.drop_duplicates()

# Fill missing values with mean
data_cleaned = data_cleaned.fillna(data_cleaned.median())

# Save the cleaned data
csv_file_path = r'C:\Users\acer\Documents\Sambilaycord\LoanData_Cleaned_v1.0.csv'
data_cleaned.to_csv(csv_file_path, index=False)
print(f'CSV file LoanData_Cleaned has been created.')

# Check if there are missing values
if data_cleaned.isnull().values.any():
    print("Missing values found in the dataset")
else:
    print("No missing values found in the dataset")

# Standardize the data
scaler = StandardScaler()
X = data_cleaned.iloc[:, :-1].values  # Features (excluding target)
Y = data_cleaned.iloc[:, -1].values   # Target (default)
X_standard = scaler.fit_transform(X)

# Save the standardized data
X_standard = pd.DataFrame(X_standard, columns=data_cleaned.columns[:-1])
data_standardized = pd.concat([X_standard, pd.Series(Y, name=data_cleaned.columns[-1])], axis=1)
data_standardized.to_csv("LoanData_Standardized.csv", index=False)
print("Successfully standardized the data and saved to LoanData_Standardized.csv")