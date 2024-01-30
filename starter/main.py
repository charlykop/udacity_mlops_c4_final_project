import os
from typing import Union, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pickle
import pandas as pd
import json
from ml.data import process_data
from ml.model import inference

class Data(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int 
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int 
    capital_loss: int 
    hours_per_week: int 
    native_country: str 


    model_config = {
        "json_schema_extra" : {
            "examples": [
                {
                    "age": 29,
                    "workclass": "State-gov",
                    "fnlgt": 77516,
                    "education": "Bachelors",
                    "education_num": 13,
                    "marital_status": "Never-married",
                    "occupation": "Adm-clerical",
                    "relationship": "Not-in-family",
                    "race": "White",
                    "sex": "Male",
                    "capital_gain": 2174,
                    "capital_loss": 0,
                    "hours_per_week": 40,
                    "native_country": "United-States"
                }
            ]
            
        }
    }
        # allow_population_by_field_name = True
    

# load model, encoder and LabelBinarizer
current_dir = os.path.dirname(__file__)
with open(os.path.join(current_dir + '/model/trained_model.pkl'), 'rb') as file:
    model = pickle.load(file)
with open(os.path.join(current_dir + '/model/label_binarizer.pkl'), 'rb') as file:
    lb = pickle.load(file)
with open(os.path.join(current_dir + '/model/encoder.pkl'), 'rb') as file:
    encoder = pickle.load(file)


app = FastAPI(
    title = "Salery Prediction",
    description = "An API that predicts the salery of adults.",
    version = "1.0.0",
)

# Welcome user
@app.get("/")
async def welcome_user():
    return "Welcome to this API."

# POST: model inference
@app.post("/model_inference/")
async def return_predictions(data: Data):

    data_set = {key.replace('_', '-'): [value] for key, value in data.__dict__.items()}

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

    df = pd.DataFrame(data_set, index=[0])
    print(df)
    print(type(encoder))

    X_test, y_test, encoder_test, lb_test = process_data(
        df, categorical_features=cat_features, training=False, 
        encoder=encoder, lb=lb
    )
    print(X_test)
    preds = inference(model, X_test)
    print(preds)
    if preds[0] == 0:
        predictions = "<=50K"
    else:
        predictions = ">50K"
    
    return predictions

if __name__ == '__main__':
    pass

