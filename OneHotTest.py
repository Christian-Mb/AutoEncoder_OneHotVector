import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
#Create a test dataframe
df = pd.DataFrame([
       ['green', 'Chevrolet', 2017],
       ['blue', 'BMW', 2015],
       ['yellow', 'Lexus', 2018],
])
df.columns = ['color', 'make', 'year']

le_color = LabelEncoder()
le_make = LabelEncoder()
df['color_encoded'] = le_color.fit_transform(df.color)
df['make_encoded'] = le_make.fit_transform(df.make)

print(df)

color_ohe = OneHotEncoder()
make_ohe = OneHotEncoder()
X = color_ohe.fit_transform(df.color_encoded.values.reshape(-1,1)).toarray()
Xm = make_ohe.fit_transform(df.make_encoded.values.reshape(-1,1)).toarray()

dfOneHot = pd.DataFrame(X, columns = ["Color_"+str(int(i)) for i in range(X.shape[1])])
df = pd.concat([df, dfOneHot], axis=1)

dfOneHot = pd.DataFrame(Xm, columns = ["Make"+str(int(i)) for i in range(X.shape[1])])
df = pd.concat([df, dfOneHot], axis=1)

color_lb = LabelBinarizer()
make_lb = LabelBinarizer()
X = color_lb.fit_transform(df.color.values)
Xm = make_lb.fit_transform(df.make.values)


