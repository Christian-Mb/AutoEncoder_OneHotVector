from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import pandas as pd

dataset =pd.read_csv('C:\\Users\\Christian Mada-Mbari\\Desktop\\test.csv')
print(dataset)
X = dataset.iloc[:, :-1].values
le = LabelEncoder()
for i in range(12):
    X[:,i] = le.fit_transform(X[:,i])

ohe = OneHotEncoder(categorical_features = [0])
X = ohe.fit_transform(X).toarray()

print(X)


