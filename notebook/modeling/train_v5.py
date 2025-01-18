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

from skopt import BayesSearchCV
from skopt.space import Integer, Real


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
def evaluate_classifier(model, X_train, y_train, X_test, y_test):
    print(f"Evaluating classification model: {model.__class__.__name__}")
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'Training Accuracy': accuracy_score(y_train, y_pred_train),
        'Test Accuracy': accuracy_score(y_test, y_pred_test),
        'Training F1-Score': f1_score(y_train, y_pred_train),
        'Test F1-Score': f1_score(y_test, y_pred_test),
        'Training Precision': precision_score(y_train, y_pred_train),
        'Test Precision': precision_score(y_test, y_pred_test),
        'Training Recall': recall_score(y_train, y_pred_train),
        'Test Recall': recall_score(y_test, y_pred_test),
        'Training AUC': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
        'Test AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value:.3f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Pred Negative', 'Pred Positive'], 
                yticklabels=['True Negative', 'True Positive'])
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix - {model.__class__.__name__}')
    plt.show()

# Hyperparameter Optimization for Random Forest
print("Optimizing hyperparameters for Random Forest...")
rfc = RandomForestClassifier(random_state=42)
# param_distributions = {
#     'n_estimators': [50, 100, 200],
#     'max_depth': [None, 10, 20, 30],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4]
# }

# random_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_distributions, n_iter=20, 
#                                    cv=3, scoring='roc_auc', verbose=2, n_jobs=-1, random_state=42)
# random_search.fit(X_train, y_train)
# best_model = random_search.best_estimator_
# print("Best Random Forest Model:")
# print(random_search.best_params_)


# Define the parameter search space
param_space = {
    'n_estimators': Integer(50, 300),
    'max_depth': Integer(5, 30),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 5)
}
# Bayesian optimization with BayesSearchCV
print("Optimizing hyperparameters for Random Forest using Bayesian optimization...")
bayes_search = BayesSearchCV(
    estimator=rfc,
    search_spaces=param_space,
    n_iter=10,  # Number of iterations
    cv=3,  # Cross-validation
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1  # Print progress to console
)

bayes_search.fit(X_train, y_train)
# Retrieve the best model and parameters
best_model = bayes_search.best_estimator_
print("Best Random Forest Model with Bayesian Optimization:")
print(bayes_search.best_params_)



# Evaluate the best model
print("\n--- Evaluating Best Model ---")
evaluate_classifier(best_model, X_train, y_train, X_test, y_test)

OUTPUT_MODEL_FILE='../../output/best_random_forest_model.pkl'

# Save the best model
print("Saving the best model...")
joblib.dump(best_model, OUTPUT_MODEL_FILE)
print(f"Model saved as '{OUTPUT_MODEL_FILE}'")

# Regression Models
print("\n--- Linear Regression ---")
linear_model = LinearRegression()
evaluate_regression_model(linear_model, X_train, y_train, X_test, y_test)

print("\n--- Ridge Regression ---")
ridge_model = Ridge(alpha=1.0)
evaluate_regression_model(ridge_model, X_train, y_train, X_test, y_test)

print("\n--- Lasso Regression ---")
lasso_model = Lasso(alpha=0.1)
evaluate_regression_model(lasso_model, X_train, y_train, X_test, y_test)

print("\n--- Decision Tree Regressor ---")
decision_tree = DecisionTreeRegressor(random_state=42)
evaluate_regression_model(decision_tree, X_train, y_train, X_test, y_test)

print("\n--- XGBoost Regressor ---")
xgb_regressor = xgb.XGBRegressor(n_estimators=100, random_state=42)
evaluate_regression_model(xgb_regressor, X_train, y_train, X_test, y_test)

# Classification Models
print("\n--- Logistic Regression ---")
log_reg = LogisticRegression(max_iter=1000)
evaluate_classifier(log_reg, X_train, y_train, X_test, y_test)

print("\n--- Random Forest Classifier ---")
evaluate_classifier(rf_classifier, X_train, y_train, X_test, y_test)

print("\n--- XGBoost Classifier ---")
xgb_classifier = xgb.XGBClassifier(n_estimators=100, random_state=42)
evaluate_classifier(xgb_classifier, X_train, y_train, X_test, y_test)

# Final Evaluation with the Best Model
print("\n--- Final Model Evaluation: Best Random Forest ---")
evaluate_classifier(best_model, X_train, y_train, X_test, y_test)
