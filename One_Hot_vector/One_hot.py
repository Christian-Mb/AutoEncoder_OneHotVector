from connection import Connect
from sklearn.feature_extraction.text import CountVectorizer
from scipy import sparse


class One_Hot_Vector :

    def __init__(self):
        db = Connect.db
        print("appel du constructeur")



    def oneHot(self):

        mycursor = db.cursor()
        mycursor.execute("SELECT distinct Theme FROM theme")
        result = mycursor.fetchall()  # requête qui récupères toutes les données de la requête

        list = []

        for x in result:  # méthode pour supprimer les parenthèses (ne fonctionne pas sans)

            x = str(x)
            x = x[2:]
            x = x[:-3]
            print(" \n", x)
            list.append(x)

        vectorizer = CountVectorizer()
        data = vectorizer.fit_transform(list)  # transformation du texte en vecteur (dans toute la liste)
        matrice = data.toarray()
        self.test = matrice
        matrice_sparse = sparse.csr_matrix(matrice)  # transforme les matrices en matrice sparse
        return matrice