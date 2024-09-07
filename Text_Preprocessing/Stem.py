import nltk
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer

# Ensure the necessary data for lemmatization is downloaded
nltk.download('wordnet')

class Stemmer:
    # Example word for stemming and lemmatization
    # word = "running"
    
    def __init__(self):
        pass
    
    def porter_stem(self, word):
        # Initialize the Porter stemmer
        stemmer = PorterStemmer()
        # Return the stemmed word
        return stemmer.stem(word)
    # Output: run

    def lancaster_stem(self, word):
        # Initialize the Lancaster stemmer
        stemmer = LancasterStemmer()
        # Return the stemmed word
        return stemmer.stem(word)
    # Output: run
    
    def snowball_stem(self, word):
        # Initialize the Snowball stemmer for English
        stemmer = SnowballStemmer('english')
        # Return the stemmed word
        return stemmer.stem(word)
    # Output: run
    
    def lemmatize(self, word):
        # Initialize the WordNet lemmatizer
        lemmatizer = WordNetLemmatizer()
        # Return the lemmatized word
        return lemmatizer.lemmatize(word)
    # Output: running
    
# # Test the Stemmer class
# if __name__ == "__main__":
#     stemmer = Stemmer()
#     print("porter_stem:", stemmer.porter_stem(word))  
#     print("lancaster_stem:", stemmer.lancaster_stem(word))  
#     print("snowball_stem:", stemmer.snowball_stem(word))  
#     print("lemmatize:", stemmer.lemmatize(word))
