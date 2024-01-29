import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from aequitas.group import Group
from aequitas.bias import Bias
from aequitas.fairness import Fairness
from aequitas.plotting import Plot
from .data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    # Train model
    logit = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='auto', n_jobs=None, penalty='l2',
                    random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                    warm_start=False)
    model = logit.fit(X_train, y_train)

    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.linear_model._logistic.LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(X)

    return preds

def compute_sliced_performance(df, label, feature, categorical_features, model, encoder, lb):
    """ Evaluates data slices using precision, recall, and F1.

    Inputs
    ------
    df : pd.DataFrame
        Dataframe containing the features and label
    label : str
        Name of the label column in `X`
    feature: str
        Feature to be sliced
    categorical_features: list[str]
        List containing the names of the categorical features
    model : sklearn.linear_model._logistic.LogisticRegression
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer.
    Returns
    -------
    sliced_performance: list
        Performance of feature in slices. 
    """

    sliced_performance = []
    categories = df[feature].unique()
    for category in categories:
        df_category = df.loc[df[feature] == category]
        X_test, y_test, encoder_test, lb_test = process_data(
            df_category, categorical_features=categorical_features, label=label, training=False, 
            encoder=encoder, lb=lb
        )
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        sliced_performance.append(f"Feature '{feature}' - Category: '{category}': count {len(df_category)}, precision {precision}, recall {recall}, fbeta: {fbeta}")
        
    return sliced_performance