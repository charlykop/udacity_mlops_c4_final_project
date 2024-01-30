# Model Card
Carlotta Kopietz created the model. 
## Model Details
The model is logistic regression using the default hyperparameters in scikit-learn 1.3.2. 
## Intended Use
This model should be used to predict whether a person makes over 50K a year. The user are students and researchers. 
## Training Data
The data was obtained from the UCI Machine Learning Repository (https://archive.ics.uci.edu/dataset/20/census+income). The target class is salary.
The data was extracted by Barry Becker from the 1994 Census database. A set of reasonably clean records was extracted using the following conditions: ((AAGE>16) && (AGI>100) && (AFNLWGT>1)&& (HRSWK>0)) (done by data provider). 
Addition all spaces where removed. A 80 to 20 split was used to break this into a train and test set. No stratification was done. To use the data for training a One Hot Encoder was used on the features and a label binarizer was used on the labels.
## Evaluation Data
As validation data 20% of the census data were used. 
## Metrics
The following metrics were used to evaluate the model: 

- Precision: 0.7063903281519862
- Recall: 0.26370083816892326
- F-beta: 0.384037558685446

## Ethical Considerations
The data is not well balanced from the features and therefore not representative for e.g.:

- Native country is strongly unbalanced. The native countriy 'United-States' is good representated while all others are underpresentated. 
- Number of 'White' (part of race) is over representated. 
- ...

## Caveats and Recommendations
The database is from 1994 and therefore not representative for today. It would be recommend to use data just for learning purposes like learning to train a ML model.
