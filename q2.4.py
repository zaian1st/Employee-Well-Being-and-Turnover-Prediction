import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, r2_score, classification_report
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.utils.validation")
df = pd.read_csv("employee_turnover.csv", sep=',')
# Check for and handle missing values
missing_values = df.isnull().values.any()
df = df.dropna(axis=0, how='any')
missing_values_after_drop = df.isnull().values.any()
print("Missing values before dropping rows:", missing_values)
print("Missing values after dropping rows:", missing_values_after_drop)
# Convert categorical variables to numeric
Department_mapping = {'sales': 1, 'accounting': 2, 'hr': 3, 'checkin agent': 4, 'flight attendant': 5, 'management': 6, 'IT': 7, 'product_mng': 8, 'marketing': 9, 'pilot': 10}
df['Department'] = df['Department'].map(Department_mapping)
salary_mapping = {'low': 1, 'medium': 2, 'high': 3}
df['salary'] = df['salary'].map(salary_mapping)
# Define predictors (X) and target variable (y)
X = df.drop(columns=['left'])  # All columns except 'left' are predictors
y = df['left']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Create and fit the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
y_pred_rf = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_rf)
print("Accuracy:", accuracy * 100, "%")
auc_score = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])
print("AUC Score:", auc_score)
# Calculate R-squared (R^2) score
r2_rf = r2_score(y_test, y_pred_rf)
print("R^2 Score (Random Forest):", r2_rf)
# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
#  feature importances
feature_importances = pd.DataFrame(rf_model.feature_importances_,
                                index=X_train.columns,
                                columns=['Importance'])
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances['Importance'], y=feature_importances.index)
plt.xlabel('Importance Score')
plt.ylabel('Features')
plt.title('Feature Importance Plot')
plt.show()
tree_in_forest = rf_model.estimators_[0] 
max_depth_to_plot = 5 
plt.figure(figsize=(20, 10)) 
plot_tree(tree_in_forest, feature_names=X.columns, filled=True, rounded=True, class_names=["Stay", "Leave"], max_depth=max_depth_to_plot)
plt.show()