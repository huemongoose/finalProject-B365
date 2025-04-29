import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("alzheimer.csv")

# Basic info
print("=== Dataset Overview ===")
print(f"Shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Summary statistics
print("\n=== Numerical Features Summary ===")
print(df.describe())

# Categorical features analysis
print("\n=== Categorical Features ===")
print("Group (Diagnosis) Distribution:")
print(df['Group'].value_counts())
print("\nGender Distribution:")
print(df['Gender'].value_counts())
print("\nClinical Dementia Rating Distribution:")
print(df['Clinical_Dementia_Rating'].value_counts())

# Visualization
plt.figure(figsize=(15, 10))

# 1. Mental State Exam vs. Group (Cognitive Score by Diagnosis)
plt.subplot(2, 2, 1)
sns.boxplot(x='Group', y='Mental_State_Exam_Score', data=df)
plt.title("Mental State Exam Scores by Diagnosis")

# 2. Normalized Whole Brain Volume vs. Group (Brain Atrophy)
plt.subplot(2, 2, 2)
sns.boxplot(x='Group', y='Normalized_Whole_Brain_Volume', data=df)
plt.title("Brain Volume by Diagnosis")

# 3. Age Distribution
plt.subplot(2, 2, 3)
sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Age Distribution")

# 4. Dementia Rating vs. Cognitive Score
plt.subplot(2, 2, 4)
sns.scatterplot(x='Mental_State_Exam_Score', y='Clinical_Dementia_Rating', hue='Group', data=df)
plt.title("Cognitive Score vs. Dementia Rating")

plt.tight_layout()
plt.show()

# Correlation analysis (for numerical features)
numerical_cols = ['Age', 'Years_Of_Education', 'Mental_State_Exam_Score', 
                  'Clinical_Dementia_Rating', 'Estimated_Total_Intracranial_Volume', 
                  'Normalized_Whole_Brain_Volume', 'Atlas_Scaling_Factor']
plt.figure(figsize=(8, 6))
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()