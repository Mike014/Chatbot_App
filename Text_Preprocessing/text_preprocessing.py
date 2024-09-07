# text_preprocessing.py

import nltk
from .Corpus import NLTKCorpora
from .Re import Regex
from .Stem import Stemmer
from .Tokenize import Tokenizer

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
        print("Sentences:", sentences)

        # Word tokenization
        words = self.tokenizer.word_tokenize(text)
        print("Words:", words)

        # Regex tokenization
        regex_words = self.tokenizer.regex_tokenize(text)
        print("Regex Words:", regex_words)

        # Whitespace tokenization
        whitespace_words = self.tokenizer.whitespace_tokenize(text)
        print("Whitespace Words:", whitespace_words)

        # Porter stemming
        porter_stems = [self.stemmer.porter_stem(word) for word in words]
        print("Porter Stems:", porter_stems)

        # Lancaster stemming
        lancaster_stems = [self.stemmer.lancaster_stem(word) for word in words]
        print("Lancaster Stems:", lancaster_stems)

        # Snowball stemming
        snowball_stems = [self.stemmer.snowball_stem(word) for word in words]
        print("Snowball Stems:", snowball_stems)

        # Lemmatization
        lemmas = [self.stemmer.lemmatize(word) for word in words]
        print("Lemmas:", lemmas)

        # Stopwords
        stopwords = self.corpora.get_all_stopwords()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        print("Filtered Words:", filtered_words)

        # N-grams
        bigrams = list(self.tokenizer.ngrams(words, 2))
        trigrams = list(self.tokenizer.ngrams(words, 3))
        print("Bigrams:", bigrams)
        print("Trigrams:", trigrams)

        # Syntactic analysis (POS tagging)
        pos_tags = self.pos_tagging(words)
        print("POS Tags:", pos_tags)

        return {
            "sentences": sentences,
            "words": words,
            "regex_words": regex_words,
            "whitespace_words": whitespace_words,
            "porter_stems": porter_stems,
            "lancaster_stems": lancaster_stems,
            "snowball_stems": snowball_stems,
            "lemmas": lemmas,
            "filtered_words": filtered_words,
            "bigrams": bigrams,
            "trigrams": trigrams,
            "pos_tags": pos_tags
        }

    def pos_tagging(self, words):
        return nltk.pos_tag(words)

# Example usage of the TextPreprocessing class
if __name__ == "__main__":
    text = "Hi, this is not a sentence, and neither a word"
    preprocessor = TextPreprocessing()
    result = preprocessor.preprocess_text(text)
    print("Final Result:", result)

