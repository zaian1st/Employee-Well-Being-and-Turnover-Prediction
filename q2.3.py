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
df = pd.read_csv("employee_turnover.csv",sep=',')

# Explorative analysis 
print("First few rows of the DataFrame:")
print(df.head())

# Check for and handle missing values
missing_values = df.isnull().values.any()
df = df.dropna(axis=0, how='any')
missing_values_after_drop = df.isnull().values.any()
print("Missing values before dropping rows:", missing_values)
print("Missing values after dropping rows:", missing_values_after_drop)

##################################################
# NO Drop // ALL clolumns are important for analysis
###################################################

Department_mapping = {'sales': 1, 'accounting': 2,'hr': 3, 'checkin agent': 4,'flight attendant': 5,
                    'management': 6,'IT': 7, 'product_mng': 8,'marketing': 9, 'pilot': 10}
df['Department'] = df['Department'].map(Department_mapping)

salary_mapping = {'low': 1, 'medium': 2, 'high': 3}
df['salary'] = df['salary'].map(salary_mapping)

# Calculate the correlation matrix
correlation_matrix = df.corr()
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='YlOrBr', fmt=".3f", annot_kws={"size": 14, "weight": "bold"})
plt.title('Correlation Matrix')
plt.show()

output_excel_path = "processed_turnover.xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Processed data saved to {output_excel_path}")

#########################################################################################################
# apply the analysis but only for left emplyees 
########################################################################################################

left_employees_df = df[df['left'] == 1]
left_employees_summary_stats = left_employees_df.describe()
# Print summary statistics for left employees
print("****************************************")
print("Summary Statistics for Left Employees:")
print(left_employees_summary_stats)
# Calculate correlation between 'left' and 'salary' columns
correlation = df['left'].corr(df['salary'])
print("Correlation between 'left' and 'salary':", correlation)

salary_levels = sorted(df['salary'].unique())
slopes_staying = []
slopes_leaving = []
for salary_level in salary_levels:
    staying_count = len(df[(df['left'] == 0) & (df['salary'] == salary_level)])
    leaving_count = len(df[(df['left'] == 1) & (df['salary'] == salary_level)])
    total_count = staying_count + leaving_count
    slope_staying = staying_count / total_count
    slope_leaving = leaving_count / total_count
    slopes_staying.append(slope_staying)
    slopes_leaving.append(slope_leaving)
plt.figure(figsize=(10, 6))
plt.bar(salary_levels, slopes_staying, color='blue', alpha=0.5, label='Staying')
plt.bar(salary_levels, slopes_leaving, color='orange', alpha=0.5, label='Leaving')
for i in range(len(salary_levels)):
    plt.text(salary_levels[i], slopes_staying[i], f"{slopes_staying[i]*100:.2f}%", ha='center', va='bottom', color='black')
    plt.text(salary_levels[i], slopes_leaving[i], f"{slopes_leaving[i]*100:.2f}%", ha='center', va='bottom', color='black')
plt.plot([1, 3], [slopes_staying[0], slopes_staying[2]], color='blue')
plt.plot([1, 3], [slopes_leaving[0], slopes_leaving[2]], color='orange')
plt.xlabel('Salary Level')
plt.ylabel('Slope')
plt.title('Slopes of Staying vs Leaving Employees by Salary Level')
plt.legend()
plt.xticks(salary_levels)
plt.show()
