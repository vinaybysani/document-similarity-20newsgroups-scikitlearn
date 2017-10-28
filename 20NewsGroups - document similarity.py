import sklearn
import timeit
import numpy as np
import sklearn.datasets
import sklearn.metrics.pairwise
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import plotly as py
import plotly.graph_objs as go

import nltk
# nltk.download()

from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

categories = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
              'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles',
              'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
              'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc',
              'talk.religion.misc']
cat_len = len(categories)
all_group_docs = []
category_docs_count = []
all_group_vects = []

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])

def heat_map(cos_mat):
    trace = go.Heatmap(z=cos_mat, x=categories, y=categories)
    data = [trace]
    test = py.offline.plot(data)
    # py.image.save_as(test, filename='a-simple-plot.png', )


def category_docsCount():
    for i in categories:
        each_group_docs = fetch_20newsgroups(subset='all', categories=[i], shuffle=True, random_state=42)
        # print(len(each_group_docs.data))
        category_docs_count.append(len(each_group_docs.data))


def read_data():
    # strings merge - for each category (iteration)
    for i in categories:
        each_group = fetch_20newsgroups(subset='all', categories=[i], shuffle=True, random_state=42)
        each_group_docs = []

        # iterate over each document in specific group, append its document
        for i in each_group.data:
            each_group_docs.append(i)

        all_group_docs.append(each_group_docs)

        # print(len(all_group_docs[2]))


# dict_vect = DictVectorizer()
def vectorizer(choice):

    x = []  # all docs in all groups into single list
    for i in all_group_docs:
        x += i


    if (choice == "bagofwords"):
        vectorizer = CountVectorizer()
        vect = vectorizer.fit_transform(x)
    elif (choice == "tokenized" or choice == "tfidf"):
        vectorizer = StemmedCountVectorizer()
        vect = vectorizer.fit_transform(x)
        if (choice == "tfidf"):
            transformer = TfidfTransformer()
            vect = transformer.fit_transform(vect)



    # divide big matrix into individual matrices
    x = 0
    for i in category_docs_count:
        all_group_vects.append(vect[x:x + i])
        x += i

    # shape and avg.non-zero entries
    print(vect.get_shape()) # shape
    nonZeros = 0
    for i in range(len(categories)):
        nonZeros += vect[i].count_nonzero()
    print(nonZeros/len(categories)) # avg non-zeros
    # exit()

    cos_mat = []
    for i in range(cat_len):
        l = []
        for j in range(cat_len):
            l.append(np.mean(sklearn.metrics.pairwise.cosine_similarity(all_group_vects[i], all_group_vects[j])))
        cos_mat.append(l)
    return(cos_mat)



def get_most_least_similar_groups(cos_mat):
    max_val = -1  # impossible value
    max_indices = []
    min_val = 2  # impossible value
    min_indices = []
    for i in range(len(categories)):
        for j in range(len(categories)):
            if (i == j):
                continue
            val = cos_mat[i][j]
            if (val > max_val):
                max_val = val
                max_indices = [i, j]
            if (val < min_val):
                min_val = val
                min_indices = [i, j]

    print("most similar Newsgroups - ", max_indices, max_val, categories[max_indices[0]], categories[max_indices[1]])
    print("Least similar Newsgroups - ", min_indices, min_val, categories[min_indices[0]], categories[min_indices[1]])


read_data()
category_docsCount()
# cos_mat = vectorizer("bagofwords")
# cos_mat = vectorizer("tokenized")
cos_mat = vectorizer("tfidf")
# heat_map(cos_mat)
get_most_least_similar_groups(cos_mat)