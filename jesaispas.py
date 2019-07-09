import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
dataset = pd.read_csv("C:\\Users\\Christian Mada-Mbari\\Desktop\\test.csv")
dataset.head()
df_x = dataset["Type"]
print("Type : \n " ,df_x)
print(df_x)
cv = CountVectorizer()
x_train, x_test = train_test_split(df_x, test_size=0.2, random_state=4)
x_traincv = cv.fit_transform(["the mouse run up the clock",
                              "the mouse ran down",
                              "good morning",
                              "how are you ?",
                              "I'm so gleeful",
                              "Moving in average"])
print(x_traincv.toarray())
matrice = x_traincv.toarray()
matrice_sparse = sparse.csr_matrix(matrice)
print("matrice non sparse : \n",x_traincv, "fin matrice non sparse")
print("matrice sparse : \n",matrice_sparse, "fin matrice sparse")
print(sparse.csr_matrix.todense(matrice))
print(cv.get_feature_names())
cv1 = CountVectorizer()
x_traincv=cv1.fit_transform(x_train)
data = x_traincv.toarray()
print(data)

print("la taille de data: ",len(data))
print("la taille de dfx : ", len(df_x))



print("test : ",data[25])
print(x_train.iloc[25])

print("generation : ",cv1.inverse_transform(data[25]))
print("phrase : ", x_train.iloc[25])