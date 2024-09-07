from nltk.tokenize import word_tokenize, RegexpTokenizer, sent_tokenize, PunktSentenceTokenizer, WhitespaceTokenizer
from nltk.util import ngrams
import nltk

# Ensure the necessary data for tokenization is downloaded
nltk.download('punkt')

class Tokenizer:
    def __init__(self):
        pass
    
    def word_tokenize(self, text):
        # Tokenize the text into words
        return word_tokenize(text)
    # Output: ['This', 'is', 'a', 'sentence', '.']
    
    def regex_tokenize(self, text):
        # Tokenize the text using a regular expression
        tokenizer = RegexpTokenizer(r'\w+')
        return tokenizer.tokenize(text)
    # Output: ['This', 'is', 'a', 'sentence']

    def sent_tokenize(self, text):
        # Tokenize the text into sentences
        return sent_tokenize(text)
    # Output: ['This is a sentence.', 'And this is another one.']
    
    def punkt_tokenize(self, text):
        # Tokenize the text into sentences using Punkt tokenizer
        tokenizer = PunktSentenceTokenizer()
        return tokenizer.tokenize(text)
    # Output: ['This is a sentence.', 'And this is another one.']
    
    def whitespace_tokenize(self, text):
        # Tokenize the text based on whitespace
        tokenizer = WhitespaceTokenizer()
        return tokenizer.tokenize(text)
    # Output: ['This', 'is', 'a', 'sentence.']
    
    def ngrams(self, text, n):
        # Generate n-grams from the text
        return ngrams(text, n)
    # Output: [('This', 'is'), ('is', 'a'), ('a', 'sentence.')] with n=2
    
# Test the Tokenizer class
# if __name__ == "__main__":  
#     tokenizer = Tokenizer()
    
#     sentence = "This is a sentence. And this is another one."
#     print("word_tokenize:", tokenizer.word_tokenize(sentence))  
#     print("regex_tokenize:", tokenizer.regex_tokenize(sentence))  
#     print("sent_tokenize:", tokenizer.sent_tokenize(sentence))  
#     print("punkt_tokenize:", tokenizer.punkt_tokenize(sentence))  
#     print("whitespace_tokenize:", tokenizer.whitespace_tokenize(sentence))  
#     print("bigrams:", list(tokenizer.ngrams(sentence.split(), 2)))  
#     print("trigrams:", list(tokenizer.ngrams(sentence.split(), 3)))  
#     print("4-grams:", list(tokenizer.ngrams(sentence.split(), 4)))