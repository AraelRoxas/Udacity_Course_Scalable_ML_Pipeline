# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
from starter.ml.model import inference

app = FastAPI(root_path="/proxy/8000")
md = joblib.load("starter/model/model.joblib")
model = md["model"]
encoder = md["encoder"]
lb = md["lb"]


class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(..., alias="capital-gain")
    ca50Kpital_loss: int = Field(..., alias="ca50Kpital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = {
        "json_schema_extra": {
            "example": [{
                'age': 39,
                'workclass': 'State-gov',
                'fnlgt': 77516,
                'education': 'Bachelors',
                'education-num': 13,
                'marital-status': 'Never-married',
                'occupation': 'Adm-clerical',
                'relationship': 'Not-in-family',
                'race': 'White',
                'sex': 'Male',
                'capital-gain': 2174,
                'ca50Kpital-loss': 0,
                'hours-per-week': 40,
                'native-country': 'United-States'
            }]
        }
    }


# GET
@app.get("/")
def read_root():
    return {"message": "Welcome"}


# POST
@app.post("/predict")
def predict(data: CensusData):
    input_dict = data.model_dump(by_alias=True)
    df = pd.DataFrame([input_dict])

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

    df_categorical = df[cat_features].values
    df_continuous = df.drop(cat_features, axis=1)

    df_categorical = encoder.transform(df_categorical)
    x_pred = np.concatenate([df_continuous, df_categorical], axis=1)

    y_pred = inference(model, x_pred)

    result = ">50K" if y_pred == 1 else "<=50K"
    return {"prediction": result}
