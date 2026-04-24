"""
Script to train machine learning model.
"""
# Add the necessary imports for the starter code.
from sklearn.model_selection import train_test_split
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, model_slices
import joblib

# Add code to load in the data.
DATA_PATH = "starter/data/census.csv"
data = pd.read_csv(DATA_PATH)

# Optional enhancement, use K-fold cross validation
# instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

# Train and save a model.
# Input of model is the label in cat_features, output is the label salary
model_trained = train_model(X_train, y_train)
# save the model
joblib.dump({
    "model": model_trained,
    "encoder": encoder,
    "lb": lb
}, "starter/model/model.joblib")

# activate model slices
feature = test["education"]
model_slices(model_trained, X_test, y_test, feature)
