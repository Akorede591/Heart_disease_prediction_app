import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib # For saving and loading the model

# --- 1. Simulate Data Loading (Replace with your actual data loading) ---
# In a real scenario, you would load your 'heart.csv' or similar dataset.
# For demonstration, let's create a dummy dataset that mimics the structure
# and feature names mentioned in your Chapter 3.

# Define the features based on Table 3.1 and your project's context
# Note: 'target' is the dependent variable
features_list = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
]

# Create a dummy DataFrame (replace this with your actual data loading)
# Example: df = pd.read_csv('your_heart_disease_dataset.csv')
# For this script, we'll generate random data for demonstration purposes.
np.random.seed(42) # for reproducibility
n_samples = 1025
dummy_data = {
    'age': np.random.randint(20, 80, n_samples),
    'sex': np.random.randint(0, 2, n_samples),
    'cp': np.random.randint(0, 4, n_samples), # 0-3
    'trestbps': np.random.randint(90, 180, n_samples),
    'chol': np.random.randint(150, 400, n_samples),
    'fbs': np.random.randint(0, 2, n_samples),
    'restecg': np.random.randint(0, 3, n_samples), # 0-2
    'thalach': np.random.randint(100, 200, n_samples),
    'exang': np.random.randint(0, 2, n_samples),
    'oldpeak': np.random.uniform(0.0, 4.0, n_samples).round(1),
    'slope': np.random.randint(0, 3, n_samples), # 0-2
    'ca': np.random.randint(0, 4, n_samples), # 0-3
    'thal': np.random.randint(0, 3, n_samples), # 0-2 (assuming 0,1,2 for thal)
    'target': np.random.randint(0, 2, n_samples) # 0 or 1
}
df = pd.DataFrame(dummy_data)

print("--- Data Loading and Initial Inspection ---")
print("Shape of the raw dataset:", df.shape)
print("First 5 rows of the raw dataset:")
print(df.head())

# --- 2. Data Pre-processing (as per Chapter 3) ---

# 2.1 Data Inspection and Cleaning (already done by creating DataFrame)
# 2.2 Handling Missing Values (your project states no missing values)
print("\n--- Data Pre-processing ---")
print("Checking for missing values (as per Chapter 3, should be none):")
print(df.isnull().sum())

# 2.3 Outlier Detection and Treatment (Winsorization using IQR)
# Identify numerical columns for outlier treatment (excluding binary and categorical that will be one-hot encoded)
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numerical_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
print("Outliers treated using Winsorization for numerical columns.")

# 2.4 Categorical Encoding (One-hot encoding for 'cp', 'restecg', 'slope', 'thal')
# Binary features 'sex', 'fbs', 'exang' are already 0/1, no need to encode.
categorical_cols_to_encode = ['cp', 'restecg', 'slope', 'thal']
df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=False) # Keep all categories for MI scores
print("Categorical features one-hot encoded.")
print("Shape after encoding:", df_encoded.shape)

# Separate features (X) and target (y)
X = df_encoded.drop('target', axis=1)
y = df_encoded['target']

# --- Save the full list of columns after one-hot encoding (before scaling) ---
# This list is crucial for ensuring consistency in the Streamlit app's input
full_encoded_columns = X.columns.tolist()
joblib.dump(full_encoded_columns, 'full_encoded_columns.joblib')
print(f"Full list of encoded columns saved as 'full_encoded_columns.joblib'")


# 2.5 Feature Scaling (Min-Max Normalization)
# Apply scaling to all numerical columns, which are now all columns in X after one-hot encoding.
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("Features scaled using Min-Max Normalization.")

# --- Save the list of columns that the scaler was fitted on (which are all columns of X_scaled) ---
# This is crucial for app.py to ensure input to scaler matches training
scaler_fit_columns = X_scaled.columns.tolist()
joblib.dump(scaler_fit_columns, 'scaler_fit_columns.joblib')
print(f"List of columns used for scaler fit saved as 'scaler_fit_columns.joblib'")


# 2.6 Train-Test Split (80% train, 20% test, stratified)
# Use X_scaled for train-test split as it's the output of the scaling step
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split into training (80%): {X_train.shape} and testing (20%): {X_test.shape}")

# --- 3. Mutual Information Feature Selection (as per Chapter 3) ---
print("\n--- Mutual Information Feature Selection ---")

# Compute MI scores
# Ensure X_train is a DataFrame to keep column names
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
print("Computed MI Scores:")
print(mi_series)

# Select top 'k' features based on cross-validation result (k=10 from Chapter 4)
k_optimal = 10
selector = SelectKBest(mutual_info_classif, k=k_optimal)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features_names = X_train.columns[selector.get_support()]
print(f"\nSelected Top {k_optimal} Features:")
print(selected_features_names.tolist())

# --- Save the exact columns that the selector expects as input (i.e., X_train's columns) ---
# This is `X_train.columns` which should be identical to `scaler_fit_columns`
joblib.dump(X_train.columns.tolist(), 'selector_input_columns.joblib')
print(f"List of columns used for selector input saved as 'selector_input_columns.joblib'")


# --- 4. Naïve Bayes Classifier Implementation (as per Chapter 3) ---
print("\n--- Naïve Bayes Classifier Implementation ---")

model = GaussianNB()

# --- 5. Cross-Validation (as per Chapter 4) ---
# Using Stratified K-Fold for robust evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_selected, y_train, cv=cv, scoring='accuracy')

print(f"Mean Cross-Validation Accuracy (5-fold) on training set with {k_optimal} features: {np.mean(cv_scores):.4f}")
print(f"Individual CV Accuracies: {cv_scores.round(4)}")

# Train the final model on the entire selected training data
model.fit(X_train_selected, y_train)
print("Final Gaussian Naïve Bayes model trained on selected features.")

# --- 6. Performance Evaluation on Test Set (as per Chapter 4) ---
print("\n--- Performance Evaluation on Test Set ---")

y_pred = model.predict(X_test_selected)
y_proba = model.predict_proba(X_test_selected)[:, 1] # Probability of positive class

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive']))

# --- 7. Save the Trained Model ---
model_filename = 'gaussian_naive_bayes_model.joblib'
joblib.dump(model, model_filename)
print(f"\nTrained model saved as '{model_filename}'")

# --- 8. Save the Scaler and Feature Selector ---
scaler_filename = 'minmax_scaler.joblib'
joblib.dump(scaler, scaler_filename)
print(f"MinMaxScaler saved as '{scaler_filename}'")

selector_filename = 'mi_feature_selector.joblib'
joblib.dump(selector, selector_filename)
print(f"Mutual Information Feature Selector saved as '{selector_filename}'")

print("\n--- Integration with Streamlit App ---")
print("To integrate this model into your Streamlit app:")
print("""
1.  **Run this script** to generate the necessary `.joblib` files.
2.  **Modify your Streamlit app (`app.py`)** to load these files and use them for prediction.
""")

