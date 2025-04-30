import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer

##### Data Preprocessing #####

# Load and preprocess data
data = pd.read_csv("alzheimers_disease_data.csv")  

# Separate features (X) and target (y)
X = data.drop(['Diagnosis', 'DoctorInCharge', 'PatientID'], axis=1)  
y = LabelEncoder().fit_transform(data['Diagnosis'])  

# Impute missing values with median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
X = pd.DataFrame(X_imputed, columns=X.columns)

# Select Top 10 Features 
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)
selected_features = X.columns[selector.get_support()]

# Filter dataset to keep only selected features
X = X[selected_features]

# Split data (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



##### Finding the best K #####

# Hyperparameter search for K
param_grid = {'n_neighbors': range(1, 21)}  
knn = KNeighborsClassifier(metric='euclidean')

# 4-fold cross-validation grid search
grid_search = GridSearchCV(
    knn, param_grid, cv=4, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)


##### Evaluation #####

# Results
print(f"\nBest K: {grid_search.best_params_['n_neighbors']}")
print(f"Best CV accuracy: {grid_search.best_score_:.2f}")

# Evaluate on test set
best_knn = grid_search.best_estimator_
test_accuracy = best_knn.score(X_test_scaled, y_test)
print(f"Test accuracy (K={best_knn.n_neighbors}): {test_accuracy:.2f}")

# Generate predictions and print classification report
y_pred = best_knn.predict(X_test_scaled) 
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

