from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
NGRAM_RANGE = (1, 2)
TOP_K = 20000
TOKEN_MODE = 'word'
MIN_DOCUMENT_FREQUENCY = 2

# create CountVectorizer object
kwargs = {
    'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
    'dtype': 'int32',
    'strip_accents': 'unicode',
    'decode_error': 'replace',
    'analyzer': TOKEN_MODE,  # Split text into word tokens.
    'min_df': MIN_DOCUMENT_FREQUENCY,
}
vectorizer = TfidfVectorizer(**kwargs)
corpus = ["the house had a tiny little mouse",

          "the end of the mouse story"
          ]
# learn the vocabulary and store CountVectorizer sparse matrix in X
X = vectorizer.fit_transform(corpus)
# columns of X correspond to the result of this method
vectorizer.get_feature_names() == (
    ['the mouse ran up down', 'first', 'four', 'is', 'little',
     'made', 'the', 'of', 'had', 'the mouse ran up the clock',
     'story', 'a', 'three'])
# retrieving the matrix in the numpy form
X.toarray()

print("premiere partie \n", X.toarray())
# transforming a new document according to learn vocabulary
vectorizer.transform(['The mouse ran up the clock']).toarray()
print(vectorizer.transform(['The mouse ran up the clock']).toarray())
from sklearn.feature_extraction.text import TfidfTransformer

counts = vectorizer.fit_transform(corpus)
print("\n frequence : \n", counts)
transformer = TfidfTransformer(smooth_idf=False, use_idf=True)

tfidf = transformer.fit_transform(counts)

tfidf.toarray()
print("\n tfidf : \n", tfidf.toarray())
