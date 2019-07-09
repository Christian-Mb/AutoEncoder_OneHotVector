from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer

vectorizer = CountVectorizer()

corpus = [
    "The mouse ran up the clock",
]

#onehot = Binarizer()
#corpus = onehot.fit_transform(corpus)
vector = vectorizer.fit_transform(corpus)

corpus = vectorizer.fit_transform(vector.toarray())
onehot = Binarizer()
corpus = onehot.fit_transform(corpus.toarray())
print(vector.toarray())

