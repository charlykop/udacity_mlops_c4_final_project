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

    sliced_predictions = []
    categories = df[feature].unique()
    for category in categories:
        df_category = df.loc[df[feature] == category]
        X_test, y_test, encoder_test, lb_test = process_data(
            df_category, categorical_features=categorical_features, label=label, training=False, 
            encoder=encoder, lb=lb
        )
        preds = inference(model, X_test)
        precision, recall, fbeta = compute_model_metrics(y_test, preds)
        sliced_predictions.append(f"Feature '{feature}' - Category: '{category}': count {len(df_category)}, precision {precision}, recall {recall}, fbeta: {fbeta}")
        
    return sliced_predictions
        




        

# def compute_sliced_performance(df, label, categorical_features, preds, lb, evaluated_slice = 'education', alpha = 0.05):
    # y = df[label]
    # X = df.drop([label], axis=1)
    # X = X[categorical_features]
    # print(X.columns)
    # try:
    #     y = lb.transform(y.values).ravel()
    #     # Catch the case where y is None because we're doing inference.
    # except AttributeError:
    #     pass
    # df_aq = X.reset_index(drop=True).copy()
    # df_aq['label_value'] = y
    # df_aq['score'] = preds

    # # print(pd.crosstab(df_aq.score, df_aq.sex))
    # # print(pd.crosstab(df_aq.label_value, df_aq.sex))
    # group = Group()
    # xtab, idxs = group.get_crosstabs(df_aq)
    # print(xtab.head())
    # bias = Bias()
    # bias_df = bias.get_disparity_major_group(xtab, original_df=df_aq, alpha=alpha, mask_significance=True)
    # print(bias_df.head())
    # print(bias_df.columns)
    # fairness = Fairness()
    # fairness_df = fairness.get_group_value_fairness(bias_df)
    # print(fairness_df.head())
    # overall_fairness = fairness.get_overall_fairness(fairness_df)
    # print(overall_fairness)
    # ap = Plot()
    # metrics = ['fpr', 'for', 'fnr']
    # disparity_metrics = [f"{metric}_disparity" for metric in metrics]
    # bias_df = bias_df[["attribute_name", "attribute_value"] + metrics + disparity_metrics]
    # eps = 0.000001
    # bias_df[disparity_metrics] = bias_df[disparity_metrics].fillna(0.0) + eps
    # bias_df = bias_df[["attribute_name", "attribute_value"] + metrics + disparity_metrics]
    # print(bias_df)
    # p = ap.plot_disparity_all(bias_df,  attributes = evaluated_slice, min_group_size= 0.01, significance_alpha=alpha)

