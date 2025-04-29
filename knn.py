import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score



# Load and preprocess data
data = pd.read_csv("alzheimers_disease_data.csv")  
data = data.dropna()

# Separate features (X) and target (y)
X = data.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1)
y = LabelEncoder().fit_transform(data['Diagnosis']) 

# Split into 75% train and 25% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Create and train KNN classifier (using default k=5)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = knn.predict(X_test)

# Print results
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))