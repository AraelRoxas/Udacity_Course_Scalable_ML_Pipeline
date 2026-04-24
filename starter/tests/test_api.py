"""
Pytest for api
"""
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_get_root():
    """ test the response from GET"""
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Welcome"}


def test_predict_high():
    """ test the predict from model, the result should be >50k"""
    example = {
        'age': 52,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 209642,
        'education': 'HS-grad',
        'education-num': 9,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'ca50Kpital-loss': 0,
        'hours-per-week': 45,
        'native-country': 'United-States',
        'salary': '>50K'}

    response = client.post("/predict", json=example)

    assert response.status_code == 200
    assert response.json()["prediction"] == ">50K"


def test_predict_low():
    """ test the predict from model, the result should be <=50k"""
    example = {
        'age': 50,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 83311,
        'education': 'Bachelors',
        'education-num': 13,
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital-gain': 0,
        'ca50Kpital-loss': 0,
        'hours-per-week': 13,
        'native-country': 'United-States',
        'salary': '<=50K'}

    response = client.post("/predict", json=example)

    assert response.status_code == 200
    assert response.json()["prediction"] == "<=50K"
