# ML Assignment 2 – Adult Income Prediction with Streamlit
## Problem statement
The Adult Income dataset contains demographic and employment information such as age, education, occupation, hours per week, and marital status, along with a binary label indicating whether the individual earns more than 50K per year. In this project, the goal is to build and compare multiple machine learning models that predict the income class (`<=50K` or `>50K`) from these attributes. 

## Dataset description

- Source: Adult/Census Income dataset from UCI Machine Learning Repository and Kaggle 
- link: https://www.kaggle.com/datasets/wenruliu/adult-income-dataset
- Number of instances: 920 data records.
- Number of features: 14 attributes plus the target i.e income.
- Target variable: `income` – binary label with values `<=50K` and `>50K`, representing whether a person’s annual income is less than or equal to 50,000 USD or greater than 50,000 USD. 


### Evaluation metrics table

After splitting the dataset into training and test sets, each model was evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC) on the test data. These values are computed in the backend and displayed in the Streamlit app.

| ML Model Name       | Accuracy | AUC    | Precision| Recall | F1     | MCC   |
|---------------------|----------|--------|----------|--------|--------|-------|
| Logistic Regression | 0.8537   | 0.9055 | 0.7411   | 0.5975 | 0.6616 | 0.575 |
| Decision Tree       | 0.8197   | 0.7569 | 0.6203   | 0.6364 | 0.6282 | 0.5094|
| kNN                 | 0.8263   | 0.8379 | 0.6561   | 0.5761 | 0.6135 | 0.5039|
| Naive Bayes         | 0.5675   | 0.7971 | 0.3505   | 0.9461 | 0.5115 | 0.352 |
| Random Forest       | 0.858    | 0.9075 | 0.7374   | 0.6317 | 0.6805 | 0.5928|
| XGBoost             | 0.876    | 0.9287 | 0.7977   | 0.6459 | 0.7138 | 0.6416|

### Observations about model performance

| ML Model Name | Observation about model performance |
|---------------------|--------------------------------------|
| Logistic Regression | Gives a good overall balance; high AUC and decent precision |
| Decision Tree | Slightly lower accuracy and MCC than Logistic regression |
| kNN | Performace is similar to Decision Tree |
| Naive Bayes | By looking at the metrics it is the weakest model. Although recall is high, precision and accuracy are low |
| Random Forest | One of the best performers. High accuracy, F1 and MCC show that combining many trees greatly improves robustness over a single tree |
| XGBoost | Overall best model. Highest accuracy, AUC and MCC, meaning it differentiates income classes most accurately |
