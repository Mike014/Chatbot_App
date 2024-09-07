# text_preprocessing.py

from .Corpus import NLTKCorpora
from .Re import Regex
from .Stem import Stemmer
from .Tokenize import Tokenizer

class TextPreprocessing:
    def __init__(self):
        self.corpora = NLTKCorpora()
        self.regex = Regex()
        self.stemmer = Stemmer()
        self.tokenizer = Tokenizer()

    def preprocess_text(self, text):
        # Tokenizzazione delle frasi
        sentences = self.tokenizer.sent_tokenize(text)
        print("Sentences:", sentences)

        # Tokenizzazione delle parole
        words = self.tokenizer.word_tokenize(text)
        print("Words:", words)

        # Tokenizzazione con regex
        regex_words = self.tokenizer.regex_tokenize(text)
        print("Regex Words:", regex_words)

        # Tokenizzazione con whitespace
        whitespace_words = self.tokenizer.whitespace_tokenize(text)
        print("Whitespace Words:", whitespace_words)

        # Stemming con Porter
        porter_stems = [self.stemmer.porter_stem(word) for word in words]
        print("Porter Stems:", porter_stems)

        # Stemming con Lancaster
        lancaster_stems = [self.stemmer.lancaster_stem(word) for word in words]
        print("Lancaster Stems:", lancaster_stems)

        # Stemming con Snowball
        snowball_stems = [self.stemmer.snowball_stem(word) for word in words]
        print("Snowball Stems:", snowball_stems)

        # Lemmatizzazione
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
            "trigrams": trigrams
        }

# Esempio di utilizzo della classe TextPreprocessing
if __name__ == "__main__":
    text = "This is a sentence. And this is another one."
    preprocessor = TextPreprocessing()
    result = preprocessor.preprocess_text(text)
    print(result)
