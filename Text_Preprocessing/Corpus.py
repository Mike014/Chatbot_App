import nltk
import nltk.corpus as corpus
from nltk.corpus import stopwords

# Scarica tutte le risorse necessarie
nltk.download('all')

class NLTKCorpora:
    def __init__(self):
        pass

    def get_brown_categories(self):
        # Brown Corpus: Get categories from the Brown Corpus
        return corpus.brown.categories()

    def get_gutenberg_files(self):
        # Gutenberg Corpus: Get file IDs from the Gutenberg Corpus
        return corpus.gutenberg.fileids()

    def get_webtext_files(self):
        # Web Text Corpus: Get file IDs from the Web Text Corpus
        return corpus.webtext.fileids()

    def get_reuters_categories(self):
        # Reuters Corpus: Get categories from the Reuters Corpus
        return corpus.reuters.categories()

    def get_inaugural_files(self):
        # Inaugural Address Corpus: Get file IDs from the Inaugural Address Corpus
        return corpus.inaugural.fileids()

    def get_movie_reviews_categories(self):
        # Movie Reviews Corpus: Get categories from the Movie Reviews Corpus
        return corpus.movie_reviews.categories()

    def get_shakespeare_files(self):
        # Shakespeare Corpus: Get file IDs from the Shakespeare Corpus
        return corpus.shakespeare.fileids()

    def get_wordnet_synsets(self, word):
        # WordNet: Get synsets for a given word from WordNet
        return corpus.wordnet.synsets(word)

    def get_names_sample(self, sample_size=10):
        # Names Corpus: Get a sample of names from the Names Corpus
        return corpus.names.words()[:sample_size]

    def get_stopwords_sample(self, language='english', sample_size=10):
        # Stopwords: Get a sample of stop words in a given language
        return corpus.stopwords.words(language)[:sample_size]

    def get_swadesh_words(self, language='en'):
        # Swadesh Corpus: Get words from the Swadesh Corpus in a given language
        return corpus.swadesh.words(language)

    def get_words_sample(self, sample_size=10):
        # Words Corpus: Get a sample of words from the Words Corpus
        return corpus.words.words()[:sample_size]

    def get_conll2000_sample(self, sample_size=1):
        # Conll2000 Corpus: Get a sample of chunked sentences from the Conll2000 Corpus
        return corpus.conll2000.chunked_sents()[:sample_size]

    def get_conll2002_sample(self, sample_size=1):
        # Conll2002 Corpus: Get a sample of sentences from the Conll2002 Corpus
        return corpus.conll2002.sents()[:sample_size]

    def get_treebank_sample(self, sample_size=1):
        # Treebank Corpus: Get a sample of sentences from the Treebank Corpus
        return corpus.treebank.sents()[:sample_size]

    def get_udhr_languages(self):
        # UDHR Corpus: Get available languages from the UDHR Corpus
        return corpus.udhr.fileids()
    
    def get_all_stopwords(self, language='english'):
        # Stopwords: Get all stopwords in English
        return set(stopwords.words(language))
   
# Esempio di utilizzo della classe
# if __name__ == "__main__":
#     corpora = NLTKCorpora()
#     print("Brown Corpus Categories:", corpora.get_brown_categories())
#     print("Gutenberg Corpus Files:", corpora.get_gutenberg_files())
#     print("Web Text Corpus Files:", corpora.get_webtext_files())
#     print("Reuters Corpus Categories:", corpora.get_reuters_categories())
#     print("Inaugural Address Corpus Files:", corpora.get_inaugural_files())
#     print("Movie Reviews Corpus Categories:", corpora.get_movie_reviews_categories())
#     print("Shakespeare Corpus Files:", corpora.get_shakespeare_files())
#     print("WordNet Synsets for 'car':", corpora.get_wordnet_synsets('car'))
#     print("Names Corpus Sample:", corpora.get_names_sample())
#     print("Stopwords Sample:", corpora.get_stopwords_sample())
#     print("Swadesh Corpus Sample:", corpora.get_swadesh_words())
#     print("Words Corpus Sample:", corpora.get_words_sample())
#     print("Conll2000 Corpus Sample:", corpora.get_conll2000_sample())
#     print("Conll2002 Corpus Sample:", corpora.get_conll2002_sample())
#     print("Treebank Corpus Sample:", corpora.get_treebank_sample())
#     print("UDHR Corpus Languages:", corpora.get_udhr_languages())
#     # print("Europarl Corpus Files:", corpora.get_europarl_files())
