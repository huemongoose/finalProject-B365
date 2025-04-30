import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

# Load the dataset
df = pd.read_csv("alzheimers_disease_data.csv")

# Basic info
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics (for all attributes)
print("\n=== Numerical Features Summary ===")
print(df.describe())

# Separate features (X) and target (y)
X = df.drop(columns=['Diagnosis', 'PatientID', 'DoctorInCharge'])
y = df['Diagnosis']

# Select top 10 features using ANOVA F-score
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# Get selected feature names
selected_features = X.columns[selector.get_support()]
print("\nTop 10 features:", selected_features.tolist())


# Create a DataFrame with only these features + Diagnosis
df_top10 = df[selected_features.tolist() + ['Diagnosis']]


# Generate numerical summaries
print("\n--- Basic Statistics for Top 10 Features ---")
print(df_top10.describe(include='all').round(2))
