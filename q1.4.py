# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")

# Load the dataset
df = pd.read_csv("employee_wellbeing.csv",sep=',')
print("First few rows of the DataFrame:")
print(df.head())

# Check for and handle missing values
missing_values = df.isnull().values.any()
df = df.dropna(axis=0, how='any')
missing_values_after_drop = df.isnull().values.any()
print("Missing values before dropping rows:", missing_values)
print("Missing values after dropping rows:", missing_values_after_drop)

# Feature Selection: Drop columns that are irrelevant for analysis
columns_to_drop = ['EMP_ID', 'SALARY','STATUS']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop([col], axis=1)
        print(f"The '{col}' column has been dropped.")
    else:
        print(f"The '{col}' column is not present in the DataFrame.")

###MAPPING###
# Convert AGE to numeric
age_mapping = {'Less than 25': 1,'25 to 35': 2, '36 to 50': 3, '51 or more': 4}
df['AGE'] = df['AGE'].map(age_mapping)

# Convert GENDER to numeric
gender_mapping = {'Male': 1, 'Female': 2}
df['GENDER'] = df['GENDER'].map(gender_mapping)


# Convert EMPLOYMENT to numeric
employment_mapping = {'checkin_agent': 1, 'flight_attendant': 2}
df['EMPLOYMENT'] = df['EMPLOYMENT'].map(employment_mapping)

# linear regression
target_variable = 'WORK_LIFE_BALANCE_SCORE'
features = df.drop(columns=[target_variable])
X_train, X_test, y_train, y_test = train_test_split(features, df[target_variable], test_size=0.3, random_state=0)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared Score: {r2}')

pred_y_df = pd.DataFrame({'Actual Value': y_test, 'Predicted value': y_pred, 'Difference': y_test - y_pred})
print("First 25 ROW:", pred_y_df[0:25])
excel_file_path = 'pred_WLB_results.xlsx'
pred_y_df.to_excel(excel_file_path, index=False)

# Scatter plot
plt.scatter(y_test, y_pred, color='red', s=5)  
plt.xlabel('Actual Work-Life Balance Score')
plt.ylabel('Predicted Work-Life Balance Score')
plt.title('Actual vs Predicted Work-Life Balance Score')
plt.show()

# Emplyee that I add 
# Make a prediction for a specific set of input values

#AGE	GENDER	EMPLOYMENT	SUFFICIENT_INCOME	TO_DO_COMPLETED	DAILY_STRESS
# 2	       2	  1	               1                 	5	          2	           	
# CORE_CIRCLE   SUPPORTING_OTHERS 	SOCIAL_NETWORK	ACHIEVEMENT     FLOW	
# 3	               6	               1	              4	           1
# DAILY_STEPS 	SLEEP_HOURS 	LOST_VACATION	PERSONAL_AWARDS 	TIME_FOR_HOBBY	HEALTHY_DIET
#      4	           8	           8	         2	               3	            2
predicted_value = model.predict([[2,2,1,1,5,2,3,6,1,3,1,4,8,8,2,3,2]])
print("Predicted Value:", predicted_value)