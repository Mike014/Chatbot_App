# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer

# Definition of the BagOfWords class
class BagOfWords:
    def __init__(self):
        # Initialization of the CountVectorizer
        self.vectorizer = CountVectorizer()
        
    def fit(self, documents):
        # Fit the vectorizer on the documents
        self.vectorizer.fit(documents)

    def fit_transform(self, documents):
        # Fit the vectorizer on the documents and transform them into Bag-of-Words features
        return self.vectorizer.fit_transform(documents)

    def transform(self, documents):
        # Transform new documents into Bag-of-Words features
        return self.vectorizer.transform(documents)

# Example usage of the BagOfWords class
if __name__ == "__main__":
    # List of example documents
    documents = [
        "This is an example of document",  # Document 1
        "This is another one."             # Document 2
    ]
    # Create an instance of the BagOfWords class
    bow = BagOfWords()
    # Fit and transform the documents into a Bag-of-Words matrix
    bow_matrix = bow.fit_transform(documents)
    # Print the Bag-of-Words matrix as an array
    print(bow_matrix.toarray())

# Expected output:
# [[1 0 1 1 1 1 0 1]
#  [0 1 0 0 1 0 1 1]]

# Explanation of the output:
# The resulting matrix has one row for each document and one column for each unique word in the vocabulary.
# The columns represent the words: ['an', 'another', 'document', 'example', 'is', 'of', 'one', 'this']
# Each value in the matrix represents the count of occurrences of the corresponding word in the document.
# For example, in the first document "This is an example of document":
# - 'an' appears 1 time
# - 'another' appears 0 times
# - 'document' appears 1 time
# - 'example' appears 1 time
# - 'is' appears 1 time
# - 'of' appears 1 time
# - 'one' appears 0 times
# - 'this' appears 1 time
# So, the corresponding row is [1, 0, 1, 1, 1, 1, 0, 1]
