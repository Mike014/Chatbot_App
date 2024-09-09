# text_preprocessing.py

import nltk
import numpy as np
from Text_Preprocessing.Corpus import NLTKCorpora
from Text_Preprocessing.Re import Regex
from Text_Preprocessing.Stem import Stemmer
from Text_Preprocessing.Tokenize import Tokenizer

# Ensure the necessary data for POS tagging is downloaded
nltk.download('averaged_perceptron_tagger')

class TextPreprocessing:
    def __init__(self):
        self.corpora = NLTKCorpora()
        self.regex = Regex()
        self.stemmer = Stemmer()
        self.tokenizer = Tokenizer()

    def preprocess_text(self, text):
        # Sentence tokenization
        sentences = self.tokenizer.sent_tokenize(text)

        # Word tokenization
        words = self.tokenizer.word_tokenize(text)

        # Example feature extraction: length of text, number of words, number of sentences
        features = [
            len(text),  # Length of the text
            len(words),  # Number of words
            len(sentences)  # Number of sentences
        ]

        # Ensure the features array has the expected number of features (e.g., 10)
        while len(features) < 10:
            features.append(0)  # Pad with zeros if necessary

        return np.array(features)  # Return an array of numerical features

    def pos_tagging(self, words):
        return nltk.pos_tag(words)

# Example usage of the TextPreprocessing class
if __name__ == "__main__":
    text = "Hi, this is not a sentence, and neither a word"
    preprocessor = TextPreprocessing()
    result = preprocessor.preprocess_text(text)
    print("Final Result:", result)

