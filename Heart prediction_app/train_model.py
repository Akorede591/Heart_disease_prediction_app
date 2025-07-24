import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.naive_bayes import GaussianNB
import pickle # Changed from joblib to pickle

# --- 1. Simulate Data Loading (Replace with your actual data loading) ---
# For demonstration, let's create a dummy dataset similar to UCI Heart Disease.
# In a real scenario, you would load your 'heart_disease_uci.csv' or similar.
data = {
    'age': np.random.randint(29, 77, 100),
    'sex': np.random.randint(0, 2, 100),
    'cp': np.random.randint(0, 4, 100), # chest pain type
    'trestbps': np.random.randint(90, 200, 100), # resting blood pressure
    'chol': np.random.randint(120, 564, 100), # serum cholestoral
    'fbs': np.random.randint(0, 2, 100), # fasting blood sugar
    'restecg': np.random.randint(0, 3, 100), # resting electrocardiographic results
    'thalach': np.random.randint(71, 202, 100), # maximum heart rate achieved
    'exang': np.random.randint(0, 2, 100), # exercise induced angina
    'oldpeak': np.random.uniform(0, 6.2, 100), # ST depression induced by exercise
    'slope': np.random.randint(0, 3, 100), # slope of the peak exercise ST segment
    'ca': np.random.randint(0, 4, 100), # number of major vessels
    'thal': np.random.randint(0, 3, 100), # thal
    'target': np.random.randint(0, 2, 100) # target variable (heart disease)
}
df = pd.DataFrame(data)

# Introduce some missing values for demonstration of handling
# df.loc[10, 'chol'] = np.nan
# df.loc[20, 'trestbps'] = np.nan

print("Original DataFrame Head:")
print(df.head())

# --- 2. Data Pre-processing (Based on Chapter 3.2 - Adapted for app.py) ---

# Handle missing values (e.g., imputation with median/mean)
for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

# Outlier Detection and Treatment (IQR method)
for col in ['trestbps', 'chol', 'thalach', 'oldpeak']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

# Separate features (X) and target (y)
X = df.drop('target', axis=1)
y = df['target']

# Identify categorical and numerical features for preprocessing
numerical_features = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalach', 'exang', 'oldpeak', 'ca']
categorical_features_to_encode = ['cp', 'restecg', 'slope', 'thal']

# Apply One-Hot Encoding for categorical features
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
encoded_features = encoder.fit_transform(X[categorical_features_to_encode])
encoded_feature_names = encoder.get_feature_names_out(categorical_features_to_encode)

# Combine original numerical features with one-hot encoded features
X_combined = pd.DataFrame(encoded_features, columns=encoded_feature_names, index=X.index)
X_combined = pd.concat([X[numerical_features], X_combined], axis=1)

# Store the full list of columns after one-hot encoding
full_encoded_columns = X_combined.columns.tolist()
with open('full_encoded_columns.pkl', 'wb') as file: # Changed to pickle and .pkl
    pickle.dump(full_encoded_columns, file)
print("'full_encoded_columns.pkl' saved.")

print("\nCombined DataFrame Head (after One-Hot Encoding):")
print(X_combined.head())

# Apply Min-Max Scaling to ALL columns of the combined DataFrame (numerical + OHE)
scaler = MinMaxScaler()
X_scaled_array = scaler.fit_transform(X_combined)
X_scaled_df = pd.DataFrame(X_scaled_array, columns=full_encoded_columns)

# Store the columns that the scaler was fitted on
scaler_fit_columns = full_encoded_columns
with open('scaler_fit_columns.pkl', 'wb') as file: # Changed to pickle and .pkl
    pickle.dump(scaler_fit_columns, file)
print("'scaler_fit_columns.pkl' saved.")

print("\nScaled DataFrame Head:")
print(X_scaled_df.head())

# --- 3. Mutual Information Feature Selection (Chapter 3.3) ---

# Split data into training and testing sets (80/20 split as per document)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)

# Calculate Mutual Information scores for feature selection
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X_train_scaled.columns).sort_values(ascending=False)

print("\nMutual Information Scores (from scaled features):")
print(mi_scores_series)

num_features_to_select = 10 # Adjust based on your analysis of MI scores
mi_feature_selector = SelectKBest(mutual_info_classif, k=num_features_to_select)
mi_feature_selector.fit(X_train_scaled, y_train)

# Transform the training and testing data using the fitted selector
X_train_selected = mi_feature_selector.transform(X_train_scaled)
X_test_selected = mi_feature_selector.transform(X_test_scaled)

# Get the names of the columns that the selector expects as input
selector_input_columns = X_scaled_df.columns.tolist()
with open('selector_input_columns.pkl', 'wb') as file: # Changed to pickle and .pkl
    pickle.dump(selector_input_columns, file)
print("'selector_input_columns.pkl' saved.")

# Save the SelectKBest object itself
with open('mi_feature_selector.pkl', 'wb') as file: # Changed to pickle and .pkl
    pickle.dump(mi_feature_selector, file)
print("'mi_feature_selector.pkl' saved.")

print(f"\nSelected {num_features_to_select} features using Mutual Information.")

# --- 4. Naïve Bayes Classifier Implementation (Chapter 3.4) ---

model = GaussianNB()
model.fit(X_train_selected, y_train)

print("\nModel training complete.")

# --- 5. Saving the Trained Model and Preprocessors ---

# Save the trained Naïve Bayes model
with open('gaussian_naive_bayes_model.pkl', 'wb') as file: # Changed to pickle and .pkl
    pickle.dump(model, file)
print("\n'gaussian_naive_bayes_model.pkl' saved.")

# Save the MinMaxScaler
with open('minmax_scaler.pkl', 'wb') as file: # Changed to pickle and .pkl
    pickle.dump(scaler, file)
print("'minmax_scaler.pkl' saved.")

print("\n--- Next Steps ---")
print("You can now run your Streamlit application using `streamlit run app.py`.")

