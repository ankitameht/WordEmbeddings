# with NLTK
from collections import defaultdict
from src.utils import tokenize
from nltk.text import TextCollection

def frequency_vectorize_nltk(doc):
    features = defaultdict(int)
    for token in tokenize(doc):
        features[token] +=1
    return features

def one_hot_vectorize_nltk(doc):
    vectors = {
        token: True
        for token in tokenize(doc)
    }
    return vectors

def tf_idf_vectorize_nltk(corpus):
    print(corpus)
    #corpus = [tokenize(doc) for doc in corpus]
    texts  = TextCollection(corpus)
    print(texts)
    for doc in corpus:
        yield {
            term: texts.tf_idf(term, doc)
            for term in doc
        }

#In Scikit-Learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import Binarizer
from sklearn.feature_extraction.text import TfidfVectorizer

def frequency_vectorize_scikitLearn(corpus):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform(corpus)
    return vectors

def one_hot_vectorize_scikitLearn(corpus):
    freq = CountVectorizer()
    corpus = freq.fit_transform(corpus)

    onehot = Binarizer()
    vector = onehot.fit_transform(corpus.toarray())
    return vector

def tf_idf_vectorize_scikitLearn(corpus):
    tfidf = TfidfVectorizer()
    corpus = tfidf.fit_transform(corpus)
    return corpus

# The Gensim way
import gensim

def frequency_vectorize_gensim(corpus):
    corpus = [tokenize(doc) for doc in corpus]
    id2word = gensim.corpora.Dictionary(corpus)
    vectors = [
        id2word.doc2bow(doc) for doc in corpus
    ]
    return vectors

def one_hot_vectorize_gensim(corpus):
    corpus = [tokenize(doc) for doc in corpus]
    id2word = gensim.corpora.Dictionary(corpus)
    vectors = [
        [(token[0], 1) for token in id2word.doc2bow(doc)]
        for doc in corpus
    ]
    return vectors

def tf_idf_vectorize_gensim(corpus):
    corpus = [tokenize(doc) for doc in corpus]
    lexicon = gensim.corpora.Dictionary(corpus)
    tfidf = gensim.models.TfidfModel(dictionary=lexicon, normalize=True)
    vectors = [tfidf[lexicon.doc2bow(doc)] for doc in corpus]
    return vectors

if __name__ == "__main__":
    string_sample1 =  "What is the step by step guide to invest in share market in india?"
    string_sample2 = "What is the step by step guide to invest in share market?"
    corpus = [string_sample1, string_sample2]


    # Method 1
    vectors = map(frequency_vectorize_nltk, corpus)
    for v in vectors:
        print(v)

    # Method 2
    vectors = frequency_vectorize_scikitLearn(corpus)
    print(type(vectors))

     # Method 3
    vectors = frequency_vectorize_gensim(corpus)
    print(vectors)

    # Method 4
    vectors = map(one_hot_vectorize_nltk, corpus)
    for v in vectors:
        print(v)

    # Method 5
    vectors = one_hot_vectorize_scikitLearn(corpus)
    print(vectors)

    # Method 6
    vectors = one_hot_vectorize_gensim(corpus)
    print(vectors)
    
    # Method 7
    vectors = tf_idf_vectorize_nltk(corpus)
    for v in vectors:
        print(v)


    # Method 8
    vectors = tf_idf_vectorize_scikitLearn(corpus)
    print(vectors)

    # Method 9
    vectors = tf_idf_vectorize_gensim(corpus)
    print(vectors)