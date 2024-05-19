# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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



###MAPPING###
# Convert AGE to numeric
age_mapping = {'Less than 25': 1,'25 to 35': 2, '36 to 50': 3, '51 or more': 4}
df['AGE'] = df['AGE'].map(age_mapping)

# Convert GENDER to numeric
gender_mapping = {'Male': 1, 'Female': 2}
df['GENDER'] = df['GENDER'].map(gender_mapping)

# Convert STATUS to numeric
status_mapping = {'single': 1, 'in a relation': 2, 'divorced': 3, 'married': 4}
df['STATUS'] = df['STATUS'].map(status_mapping)

# Convert EMPLOYMENT to numeric
employment_mapping = {'checkin_agent': 1, 'flight_attendant': 2}
df['EMPLOYMENT'] = df['EMPLOYMENT'].map(employment_mapping)

# Convert SALARY to numeric
salary_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
df['SALARY'] = df['SALARY'].map(salary_mapping)

### TASK 1: Plot two distinct bar charts ###



# Bar chart of daily stress according to gender
plt.figure(figsize=(10, 6))
ax = sns.barplot(x='GENDER', y='DAILY_STRESS', data=df, ci=None)
plt.title('Daily Stress According to Gender')
plt.xlabel('Gender   1=Male & 2=Female')
plt.ylabel('Daily Stress')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.5f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


# Bar chart of daily stress according to job role
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='EMPLOYMENT', y='DAILY_STRESS', data=df, ci=None)
plt.title('Daily Stress According to Job Role')
plt.xlabel('Job Role:   checkin_agent=1  &  flight_attendant= 2')
plt.ylabel('Daily Stress')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.5f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


# Bar chart of Hobbies time according to gender
plt.figure(figsize=(12, 6))
ax = sns.barplot(x='GENDER', y='TIME_FOR_HOBBY', data=df, ci=None)
plt.title('Hobbies time according to gender')
plt.xlabel('Gender   1=Male & 2=Female')
plt.ylabel('Hobbies time spent')
for p in ax.patches:
    ax.annotate(f'{p.get_height():.5f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.show()


# Create a heatmap to determine the attributes highly correlated with the Work-Life Balance (WLB) score
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlOrBr", fmt=".4f")
plt.title('Correlation Heatmap')
plt.show()

# Feature Selection: Drop columns that are irrelevant for analysis
columns_to_drop = ['EMP_ID','SALARY','STATUS']
for col in columns_to_drop:
    if col in df.columns:
        df = df.drop([col], axis=1)
        print(f"The '{col}' column has been dropped.")
    else:
        print(f"The '{col}' column is not present in the DataFrame.")

output_excel_path = "processed_data.xlsx"
df.to_excel(output_excel_path, index=False)

print(f"Processed data saved to {output_excel_path}")