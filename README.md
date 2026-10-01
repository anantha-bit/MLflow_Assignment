# MLflow Assignment

## Project Overview

This project demonstrates the use of **MLflow** for tracking a machine learning experiment.

A **Random Forest Classifier** is trained using the Iris dataset, and MLflow is used to record the model parameters, evaluation metric, and trained model.

---

## Objectives

- Train a machine learning classification model.
- Track model parameters using MLflow.
- Track model performance using MLflow metrics.
- Log the trained model using MLflow.
- View and manage experiments through the MLflow UI.

---

## Dataset

The **Iris dataset** from Scikit-learn is used.

The dataset contains measurements of iris flowers and has three classes:

- Setosa
- Versicolor
- Virginica

The features used are:

- Sepal length
- Sepal width
- Petal length
- Petal width

---

## Machine Learning Model

The project uses a:

**Random Forest Classifier**

### Model Parameters

| Parameter | Value |
|---|---:|
| Number of Estimators | 100 |
| Maximum Depth | 3 |
| Random State | 42 |

---

## MLflow Tracking

The MLflow experiment is named:

```text
Iris_MLflow_Assignment
