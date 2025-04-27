import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report

# Load and preprocess data
data = pd.read_csv("alzheimer.csv")  
data = data.dropna()  
data['M/F'] = data['M/F'].map({'M': 0, 'F': 1})

# Separate features (X) and target (y)
X = data.drop('Group', axis=1)  
y = LabelEncoder().fit_transform(data['Group'])  

# Split data (75% train, 25% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter search for K
param_grid = {'n_neighbors': range(1, 21)}  
knn = KNeighborsClassifier(metric='euclidean')

# 4-fold cross-validation grid search
grid_search = GridSearchCV(
    knn, param_grid, cv=4, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Results
print(f"Best K: {grid_search.best_params_['n_neighbors']}")
print(f"Best CV accuracy: {grid_search.best_score_:.2f}")

# Evaluate on test set
best_knn = grid_search.best_estimator_
test_accuracy = best_knn.score(X_test_scaled, y_test)
print(f"Test accuracy (K={best_knn.n_neighbors}): {test_accuracy:.2f}")

# Generate predictions and print classification report
y_pred = best_knn.predict(X_test_scaled) 
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Converted', 'Demented', 'Nondemented']))

