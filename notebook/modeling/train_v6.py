import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, cohen_kappa_score
)
import xgboost as xgb
import joblib

# from skopt import BayesSearchCV
# from skopt.space import Integer, Real


# File paths
DATA_DIR = '../../data'
TRAIN_DATA = f'{DATA_DIR}/fraudTrain.csv'
TEST_DATA = f'{DATA_DIR}/fraudTest.csv'

# Load the dataset
print("Loading dataset...")
train_df = pd.read_csv(TRAIN_DATA)
test_df = pd.read_csv(TEST_DATA)

# Data preprocessing
print("Preprocessing dataset...")
train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['dob'] = pd.to_datetime(train_df['dob'])
train_df['unix_time'] = pd.to_datetime(train_df['unix_time'], unit='s')

test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])
test_df['dob'] = pd.to_datetime(test_df['dob'])
test_df['unix_time'] = pd.to_datetime(test_df['unix_time'], unit='s')

# Select features and target
categorical_features = ['merchant', 'category', 'gender', 'state']
numerical_features = ['cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 'merch_lat', 'merch_long']

X = train_df[numerical_features]
y = train_df['is_fraud']

X_t = test_df[numerical_features]
y_t = test_df['is_fraud']

# Split the dataset
print("Splitting dataset...")
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4, random_state=42)
X_test,y_test = X_t, y_t

# Function to evaluate regression models
def evaluate_regression_model(model, X_train, y_train, X_test, y_test):
    print(f"Evaluating regression model: {model.__class__.__name__}")
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'Training RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'Test RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'Training MAE': mean_absolute_error(y_train, y_pred_train),
        'Test MAE': mean_absolute_error(y_test, y_pred_test),
        'Training R2': r2_score(y_train, y_pred_train),
        'Test R2': r2_score(y_test, y_pred_test)
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

# Function to evaluate classification models
def evaluate_classifier(model, X_eval, y_eval):
    print(f"Evaluating classification model: {model.__class__.__name__}")
    
    y_pred = model.predict(X_eval)

    metrics = {        
        'Test Accuracy': accuracy_score(y_eval, y_pred),        
        'Test F1-Score': f1_score(y_eval, y_pred),
        'Test Precision': precision_score(y_eval, y_pred),        
        'Test Recall': recall_score(y_eval, y_pred),        
        'Test AUC': roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")
    
    test_auc = roc_auc_score(y_eval, model.predict_proba(X_eval)[:, 1])
    return test_auc


# Classification Models
print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
print("training")
evaluate_classifier(log_reg, X_train, y_train)
print("validation")
best_auc = evaluate_classifier(log_reg, X_valid, y_valid)
best_model = log_reg

print("\n--- Random Forest Classifier ---")
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
print("training")
evaluate_classifier(rf_classifier, X_train, y_train)
print("validation")
auc= evaluate_classifier(rf_classifier, X_valid, y_valid)
if auc > best_auc:
    best_auc = auc
    best_model = rf_classifier

print("\n--- XGBoost Classifier ---")
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_classifier.fit(X_train, y_train)
print("training")
evaluate_classifier(xgb_classifier, X_train, y_train)
print("validation")
auc= evaluate_classifier(xgb_classifier, X_valid, y_valid)
if auc > best_auc:
    best_auc = auc
    best_model = xgb_classifier

# Final Evaluation with the Best Model
print("\n--- Final Model Evaluation: Best Random Forest ---")
auc= evaluate_classifier(best_model, X_test, y_test)


OUTPUT_MODEL_FILE='../../output/best_model.pkl'

# Save the best model
print("Saving the best model...")
joblib.dump(best_model, OUTPUT_MODEL_FILE)
print(f"Model saved as '{OUTPUT_MODEL_FILE}'")