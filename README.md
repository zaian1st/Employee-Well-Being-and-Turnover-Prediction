# Employee Well-Being and Turnover Prediction

Welcome to the Employee Well-Being and Turnover Prediction project! This initiative is aimed at helping organizations improve their work environment, retain top talent, and ensure the overall well-being of their employees through predictive analytics and data-driven insights.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Well-Being Analysis](#well-being-analysis)
   - [Importance of Well-Being](#importance-of-well-being)
   - [Stress Induced by Technology](#stress-induced-by-technology)
   - [Employee Well-Being Dataset](#employee-well-being-dataset)
   - [Predictive Well-Being Algorithm](#predictive-well-being-algorithm)
3. [Turnover Analysis](#turnover-analysis)
   - [Understanding Employee Turnover](#understanding-employee-turnover)
   - [Employee Turnover Theories](#employee-turnover-theories)
   - [Employee Turnover Dataset](#employee-turnover-dataset)
   - [Turnover Prediction](#turnover-prediction)
4. [Usage](#usage)
5. [Conclusion](#conclusion)
6. [Libraries Used](#libraries-used)
7. [Algorithms](#algorithms)

## Project Overview

In today's fast-paced corporate world, maintaining employee well-being and minimizing turnover are critical to a company's success. Our project at the Schöller Endowed Chair for Information Systems, Friedrich-Alexander-Universität Erlangen-Nürnberg, aims to provide actionable insights and predictive models that organizations can use to foster a healthier and more stable work environment.

## Well-Being Analysis

### Importance of Well-Being

Employee well-being is strongly correlated with organizational performance. By monitoring and enhancing well-being, companies can reduce absenteeism, improve productivity, and create a positive workplace culture. Factors influencing well-being include job design, work hours, social support, and work-life balance.

### Stress Induced by Technology

While new systems can improve efficiency, they can also introduce technostress, leading to reduced productivity and increased workplace accidents. Effective change management and comprehensive training are essential to mitigate these effects.

### Employee Well-Being Dataset

Our dataset includes key indicators such as job role, age, gender, income sufficiency, stress levels, social support, and physical activity. We cleaned and preprocessed the data to ensure accuracy and relevance for analysis.

### Predictive Well-Being Algorithm

Using linear regression, we developed a predictive model to assess employee well-being. The model was evaluated using Mean Squared Error and R-squared Score, demonstrating its ability to accurately predict well-being scores based on various factors.

## Turnover Analysis

### Understanding Employee Turnover

Voluntary turnover is a significant concern due to its associated costs, loss of knowledge, and workflow disruptions. The equilibrium theory and job embeddedness theory provide insights into the factors leading to turnover, such as job satisfaction, social connections, and perceived barriers to change.

### Employee Turnover Theories

We explored the Unfolding Model of Turnover and Job Embeddedness Theory to understand different paths and factors influencing an employee's decision to leave.

### Employee Turnover Dataset

Our dataset includes attributes such as job satisfaction, salary, work accidents, and tenure. We conducted thorough data cleaning and analysis to identify key factors affecting turnover.

### Turnover Prediction

Using a Random Forest algorithm, we developed a turnover prediction model. The model achieved high accuracy and AUC scores, highlighting the importance of job satisfaction, tenure, and overtime hours in predicting employee turnover.

## Usage

To use the predictive models and datasets:
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the provided Python scripts (`q1.3.py`, `q1.4.py`, `q2.3.py`, `q2.4.py`) to preprocess the data and train the models.





## Libraries Used

1. Pandas: For data manipulation and analysis.
2. NumPy: For numerical operations.
3. Scikit-learn: For machine learning algorithms, including linear regression and random forest.
4. Matplotlib: For data visualization.
5. Seaborn: For statistical data visualization.

## Algorithms

1. Linear Regression: Used to predict employee well-being scores based on various factors.
2. Random Forest: Employed for predicting employee turnover, offering high accuracy and feature importance evaluation.

## Conclusion

The project offers valuable insights into employee well-being and turnover, providing organizations with tools to enhance their work environment and retain talent. By leveraging predictive analytics, companies can take proactive measures to support their employees and foster a thriving workplace culture.
