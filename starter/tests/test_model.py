"""
Pytest for model train and inference
"""
import os
import numpy as np
from starter import train_model
import pandas as pd
from starter.ml.data import process_data
from starter.ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
import logging
import joblib

def test_model_train():
    """ Test of model training and saving"""
    train_model
    assert os.path.exists("starter/model/model.joblib"), "model is not saved"

def test_model_inference():
    """ Test of model inference"""
    model_trained = joblib.load("starter/model/model.joblib")
    model = model_trained["model"]
    encoder = model_trained["encoder"]
    lb = model_trained["lb"]
    
    DATA_PATH = "starter/data/census.csv"
    data = pd.read_csv(DATA_PATH)

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

    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

    pred = inference(model,X_test)
    assert isinstance(pred, np.ndarray), "pred not successful"

def test_model_metric():
    """ Test of model metric"""
    model_trained = joblib.load("starter/model/model.joblib")
    model = model_trained["model"]
    encoder = model_trained["encoder"]
    lb = model_trained["lb"]
    
    DATA_PATH = "starter/data/census.csv"
    data = pd.read_csv(DATA_PATH)

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

    X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

    pred = inference(model,X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, pred)

    assert isinstance(precision, float), "metric not successful"
    assert isinstance(recall, float), "metric not successful"
    assert isinstance(fbeta, float), "metric not successful"
    assert precision>0.5, "precision too low"

def test_model_slice():
    """ test of model slice and save in slice_output.txt with feature education"""
    assert os.path.exists("starter/starter/slice_output.txt"), "slice output is not saved"

