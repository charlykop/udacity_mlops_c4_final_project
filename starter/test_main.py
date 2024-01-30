from fastapi.testclient import TestClient
from main import app

import json
import pandas as pd
import pickle
import os
from ml.data import process_data
from ml.model import inference

client = TestClient(app)

# Test GET method
def test_get_response():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == "Welcome to this API."

# Test POST method
def test_post_prediction_lower_50():

    data = {"age": 29,
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


    r = client.post("/model_inference/", data=json.dumps(data))
    print(r.status_code)
    print(r.json())


    assert r.status_code == 200
    assert r.json() == "<=50K"

# Test POST method
def test_post_prediction_higher_50():

    data = {"age": 42,
            "workclass": "Private",
            "fnlgt": 159449,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 5178,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
    }


    r = client.post("/model_inference/", data=json.dumps(data))
    print(r.status_code)
    print(r.json())


    assert r.status_code == 200
    assert r.json() == ">50K"

if __name__ == '__main__':
    print(test_post_prediction_lower_50())





