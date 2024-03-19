import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ML_Pipeline.utils import read_data
from sklearn.model_selection import train_test_split
from ML_Pipeline.models import model_zoo, evaluate_models
from ML_Pipeline.hyperparameter import model, parameters
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from ML_Pipeline.feature_eng import AddFeatures
from ML_Pipeline.scaler import CustomScaler
from ML_Pipeline.encoding import CategoricalEncoder
from sklearn.metrics import roc_auc_score, f1_score, recall_score, confusion_matrix, classification_report
import joblib

# Read the data
df = read_data("../input/Churn_Modelling.csv")

# Define target variable and columns to remove
target_var = ['Exited']
cols_to_remove = ['RowNumber', 'CustomerId']

# Drop non-essential columns and separate target variable
if 'Exited' in df.columns:
    df.drop(columns=['Exited'], inplace=True)
for col in cols_to_remove:
    if col in df.columns:
        df.drop(cols_to_remove, axis=1, inplace=True)

# Split data into train, validation, and test sets
df_train_val, df_test, y_train_val, y_test = train_test_split(df, df[target_var].values.ravel(),
                                                              test_size=0.1, random_state=42)
df_train, df_val, y_train, y_val = train_test_split(df_train_val, y_train_val,
                                                    test_size=0.12, random_state=42)

# Spot-check models
models = model_zoo()
# Evaluate models on different metrics
# results_recall = evaluate_models(X, y, models, metric='recall')
# results_f1 = evaluate_models(X, y, models, metric='f1')

# Hyperparameter tuning
X_train = df_train.drop(columns=['Exited'], axis=1)
X_val = df_val.drop(columns=['Exited'], axis=1)
# Randomized search
# search = RandomizedSearchCV(model, parameters, n_iter=20, cv=5, scoring='f1')
# search.fit(X_train, y_train.ravel())
# Grid search
grid = GridSearchCV(model, parameters, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train.ravel())

# Ensemble models
# Prepare data for error analysis
X_train = df_train.drop(columns=['Exited'], axis=1)
X_val = df_val.drop(columns=['Exited'], axis=1)

# Define models with best parameters
lgb1 = LGBMClassifier(boosting_type='dart', class_weight={0: 1, 1: 1}, min_child_samples=20, n_jobs=-1,
                      importance_type='gain', max_depth=4, num_leaves=31, colsample_bytree=0.6, learning_rate=0.1,
                      n_estimators=21, reg_alpha=0, reg_lambda=0.5)

lgb2 = LGBMClassifier(boosting_type='dart', class_weight={0: 1, 1: 3.93}, min_child_samples=20, n_jobs=-1,
                      importance_type='gain', max_depth=6, num_leaves=63, colsample_bytree=0.6, learning_rate=0.1,
                      n_estimators=201, reg_alpha=1, reg_lambda=1)

lgb3 = LGBMClassifier(boosting_type='dart', class_weight={0: 1, 1: 3.0}, min_child_samples=20, n_jobs=-1,
                      importance_type='gain', max_depth=6, num_leaves=63, colsample_bytree=0.6, learning_rate=0.1,
                      n_estimators=201, reg_alpha=1, reg_lambda=1)

# Create pipelines for each model
model_1 = Pipeline(steps=[('categorical_encoding', CategoricalEncoder()),
                           ('add_new_features', AddFeatures()),
                           ('classifier', lgb1)])

model_2 = Pipeline(steps=[('categorical_encoding', CategoricalEncoder()),
                           ('add_new_features', AddFeatures()),
                           ('classifier', lgb2)])

model_3 = Pipeline(steps=[('categorical_encoding', CategoricalEncoder()),
                           ('add_new_features', AddFeatures()),
                           ('classifier', lgb3)])

# Fit each model
model_1.fit(X_train, y_train.ravel())
model_2.fit(X_train, y_train.ravel())
model_3.fit(X_train, y_train.ravel())

# Get prediction probabilities from each model
m1_pred_probs_val = model_1.predict_proba(X_val)
m2_pred_probs_val = model_2.predict_proba(X_val)
m3_pred_probs_val = model_3.predict_proba(X_val)

# Get predictions for the best model
m3_preds = np.where(m3_pred_probs_val[:, 1] >= 0.5, 1, 0)

# Model averaging predictions
m1_m2_preds = np.where(((0.1 * m1_pred_probs_val[:, 1]) + (0.9 * m2_pred_probs_val[:, 1])) >= 0.5, 1, 0)

# Evaluate best model on validation set
roc_auc_score(y_val, m3_preds)
recall_score(y_val, m3_preds)
confusion_matrix(y_val, m3_preds)

# Train final model
best_f1_lgb = LGBMClassifier(boosting_type='dart', class_weight={0: 1, 1: 3.0}, min_child_samples=20, n_jobs=-1,
                              importance_type='gain', max_depth=6, num_leaves=63, colsample_bytree=0.6,
                              learning_rate=0.1, n_estimators=201, reg_alpha=1, reg_lambda=1)

final_model = Pipeline(steps=[('categorical_encoding', CategoricalEncoder()),
                               ('add_new_features', AddFeatures()),
                               ('classifier', best_f1_lgb)])

# Fit final model on train dataset
final_model.fit(X_train, y_train)

# Save final model
joblib.dump(final_model, '../output/final_churn_model_f1_0_45.sav')

# Testing
# Load model object
model = joblib.load('../output/final_churn_model_f1_0_45.sav')
X_test = df_test.drop(columns=['Exited'], axis=1)

# Predict target probabilities and values on test data
test_probs = model.predict_proba(X_test)[:, 1]
test_preds = np.where(test_probs > 0.45, 1, 0)

# Test set metrics
roc_auc_score(y_test, test_preds)
recall_score(y_test, test_preds)
confusion_matrix(y_test, test_preds)
classification_report(y_test, test_preds)

# Add predictions and probabilities in the original test dataframe
test = df_test.copy()
test['predictions'] = test_preds
test['pred_probabilities'] = test_probs

# Extract high churn list
high_churn_list = test[test.pred_probabilities > 0.7].sort_values(by='pred_probabilities', ascending=False).reset_index().drop(columns=['index', 'Exited', 'predictions'], axis=1)
high_churn_list.to_csv('../output/high_churn_list.csv', index=False)

print("DONE")
