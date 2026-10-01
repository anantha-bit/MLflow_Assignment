import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# 1. Load the Iris dataset
data = load_iris()

X = data.data
y = data.target


# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 3. Define model parameters
n_estimators = 100
max_depth = 3


# 4. Create the model
model = RandomForestClassifier(
    n_estimators=n_estimators,
    max_depth=max_depth,
    random_state=42
)


# 5. Start an MLflow experiment
mlflow.set_experiment("Iris_MLflow_Assignment")

with mlflow.start_run():

    # 6. Train the model
    model.fit(X_train, y_train)

    # 7. Make predictions
    predictions = model.predict(X_test)

    # 8. Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # 9. Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)

    # 10. Log metric
    mlflow.log_metric("accuracy", accuracy)

    # 11. Log the trained model
    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")