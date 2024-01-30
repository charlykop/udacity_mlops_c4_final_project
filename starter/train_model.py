# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
import pickle
from ml.data import process_data
from ml.model import train_model, compute_model_metrics, inference, compute_sliced_performance

# Add code to load in the data. 
data = pd.read_csv('data/census.csv')

# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

# # Proces the test data with the process_data function.

X_test, y_test, encoder_test, lb_test = process_data(
    test, categorical_features=cat_features, label="salary", training=False, 
    encoder=encoder, lb=lb
)

# # Train and save a model.
model = train_model(X_train, y_train)
print(type(model))
pickle.dump(model, open('model/trained_model.pkl', 'wb'))
pickle.dump(encoder, open('model/encoder.pkl', 'wb'))
pickle.dump(lb, open('model/label_binarizer.pkl', 'wb'))
preds = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, preds)
print(f'precision: {precision}, recall: {recall}, F-beta: {fbeta}')

# # Performance of model based on slices
for feature in cat_features:    
    performance_sliced = compute_sliced_performance(test, feature=feature,label="salary", model=model, categorical_features=cat_features, encoder=encoder, lb=lb)
    with open("slice_output.txt", 'a') as file:
        for item in performance_sliced: 
            file.write(item + "\n")

