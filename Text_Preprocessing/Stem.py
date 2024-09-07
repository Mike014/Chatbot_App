import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

nltk.download('wordnet')

class Stemmer():
    # word = "running"
    def __init__(self):
        pass
    
    def porter_stem(self, word):
        stemmer = PorterStemmer()
        return stemmer.stem(word)
    # Output: run

    def lancaster_stem(self, word):
        stemmer = LancasterStemmer()
        return stemmer.stem(word)
    # Output: run
    
    def snowball_stem(self, word):
        stemmer = SnowballStemmer('english')
        return stemmer.stem(word)
    # Output: run
    
    def lemmatize(self, word):
        lemmatizer = WordNetLemmatizer()
        return lemmatizer.lemmatize(word)
    # Output: running
    
# # Test the Stemmer class
# if __name__ == "__main__":
    
#     stemmer = Stemmer()
    
    
#     print("porter_stem:", stemmer.porter_stem(word))  
#     print("lancaster_stem:", stemmer.lancaster_stem(word))  
#     print("snowball_stem:", stemmer.snowball_stem(word))  
#     print("lemmatize:", stemmer.lemmatize(word))  



