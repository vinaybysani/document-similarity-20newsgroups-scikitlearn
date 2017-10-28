# document-similarity-20newsgroups-scikitlearn
Famous 20 newsgroups  documents are compared for similarity using scikitlearn library

Used the following vector representations
1.	Bag of words (Occurences Count)
2.	Tokenized representation
3.	Term frequency - Inverse document frequency (TF-IDF)

Document similarity(Cosine) between 20 news groups has been calculated using the above 3 representations. It is observed and verified that TF-IDF representaion has been the most effective way to find similarity between documents

## Tradeoffs of three representations
-	**Bag of words** is the easiest representation of all, considering occurrence count alone.
-	**Tokenized representation** makes use of stemming. This is a better representation than the Bag of words, since it tries to reduce any kind of similar words (of same root), which do not necessarily provide us useful information on document classification.
-	**TF-IDF** is by far the best representation in terms of relevance. It makes use of term frequencies. It gives us good estimate on classifying two documents of different length. It also considers weights for words, rare words having more weight and commonly used words having low weight.

![alt text](https://github.com/vinaybysani/document-similarity-20newsgroups-scikitlearn/blob/master/images/count.png)