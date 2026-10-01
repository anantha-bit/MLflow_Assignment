\# MLflow Assignment



\## Objective



This project demonstrates how to use MLflow to track a machine learning experiment, including model parameters, evaluation metrics, and the trained machine learning model.



\## Dataset



The Iris dataset from Scikit-learn is used for this project.



The dataset contains measurements of iris flowers and three flower classes:



\- Setosa

\- Versicolor

\- Virginica



\## Machine Learning Model



A Random Forest Classifier is used for classification.



\### Model Parameters



\- `n\_estimators = 100`

\- `max\_depth = 3`

\- `random\_state = 42`



\## MLflow Tracking



The following information is tracked using MLflow:



\### Parameters



\- Number of estimators

\- Maximum tree depth



\### Metric



\- Accuracy



\### Model



The trained Random Forest model is logged using MLflow.



\## Result



The model achieved:



\*\*Accuracy: 1.0000 (100%)\*\*



\## Experiment



The MLflow experiment is named:



`Iris\_MLflow\_Assignment`



\## How to Run



Install the required libraries:



```bash

pip install mlflow pandas scikit-learn

