import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold




def normalization(data):
    df = pd.read_csv(data)
    df = df.dropna()
    results = df.iloc[:, 0]
    values = df.iloc[:, 1:]
    values['M/F'] = values['M/F'].map({'M': 0, 'F': 1})

    fit = LabelEncoder()
    fittedResults = fit.fit_transform(results)
    
    return fittedResults, values


def KNN(K, data):
    results, values = normalization(data)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(values)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    knn = KNeighborsClassifier(K, metric='euclidean')
    model = knn.fit(X_pca, results)

    return model, scaler, pca, X_pca, results

    #rkf = RepeatedKFold(n_splits=4, n_repeats=4, random_state=42)
    #scores = cross_val_score(knn, X, y, cv=rkf)

def predictKNN(K, data, prediction):
    model, scaler, pca,_,_ = KNN(K, data)
    _, values_df = normalization(data)
    feature_names = values_df.columns

    prediction_df = pd.DataFrame([prediction], columns=feature_names)

    scaled = scaler.transform(prediction_df)
    transformed = pca.transform(scaled)

    classification =  model.predict(transformed)

    if classification == [2]:
        return ["Nondemented"]
    elif classification == [1]:
        return["Demented"]
    else:
        return ["Converted"]

def plotKNN(K, data):
    _, _, _, X_pca, results = KNN(K,data)
    
    

    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=results)

    labels_map = {0: "Converted", 1: "Demented", 2: "Nondemented"}
    handles, _ = scatter.legend_elements()
    labels = [labels_map[i] for i in np.unique(results)]
    plt.legend(handles, labels, title="Classes")

    plt.title("KNN")
    plt.grid(True)
    plt.show()







data = "alzheimer.csv"
plotKNN(3, "alzheimer.csv")


print(predictKNN(3,data, np.array([0,68,16,1,7,1,1714,0.682,1.024])))
print(predictKNN(3,data, np.array([1,92,14,1,27,0.5,1423,0.696,1.234])))
print(predictKNN(3,data, np.array([1,20,14,1,27,0.5,1423,0.696,1.234])))