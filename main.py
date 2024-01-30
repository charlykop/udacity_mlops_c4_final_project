import os
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd
from ml.model import inference, process_data

class Data(BaseModel):
    age: int = Field(examples=[29])
    workclass: str = Field(examples=["State-gov"])
    fnlgt: int = Field(examples=[77516])
    education: str = Field(examples=["Bachelors"])
    education_num: int = Field(examples=[13])
    marital_status: str = Field(examples=["Never-married"])
    occupation: str = Field(examples=["Adm-clerical"])
    relationship: str = Field(examples=["Not-in-family"])
    race: str = Field(examples=["White"])
    sex: str = Field(examples=["Male"])
    capital_gain: int = Field(examples=[2174])
    capital_loss: int = Field(examples=[0])
    hours_per_week: int = Field(examples=[40])
    native_country: str = Field(examples=["United-States"])


    # model_config = {
    #     "json_schema_extra" : {
    #         "examples": [
    #             {
    #                 "age": 29,
    #                 "workclass": "State-gov",
    #                 "fnlgt": 77516,
    #                 "education": "Bachelors",
    #                 "education_num": 13,
    #                 "marital_status": "Never-married",
    #                 "occupation": "Adm-clerical",
    #                 "relationship": "Not-in-family",
    #                 "race": "White",
    #                 "sex": "Male",
    #                 "capital_gain": 2174,
    #                 "capital_loss": 0,
    #                 "hours_per_week": 40,
    #                 "native_country": "United-States"
    #             }
    #         ]
            
    #     }
    # }
    

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

    X_test, _, _, _ = process_data(
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

