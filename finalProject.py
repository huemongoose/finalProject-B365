import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler



def normalization(data):

    df = pd.read_csv(data)
    df = df.dropna()
    results = df.iloc[:,0]
    values = df.iloc[:,1:]
    values['M/F'] = values['M/F'].map({'M': 0, 'F': 1})

    fit = LabelEncoder()
    scaling = StandardScaler()

    fittedResults = fit.fit_transform(results)
    scaledValues = scaling.fit_transform(values)

    return fittedResults,scaledValues














normalizedData = normalization("alzheimer.csv")
