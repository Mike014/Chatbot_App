# Project Summary

## Project Description
The Chatbot with GUI project is an application that uses the `tkinter` library to create a graphical user interface (GUI) that allows users to interact with various natural language processing (NLP) functionalities. The application includes features for text tokenization, stemming, lemmatization, access to various corpora, the use of regular expressions, and part-of-speech (POS) tagging. These functionalities are integrated into a simple and intuitive GUI.

## Project Purpose
This project is a prototype of a larger and more complex system aimed at providing advanced tools for natural language processing. The goal is to demonstrate basic capabilities and provide a solid foundation on which to build additional features and improvements.

## Main Files

### [main.py](#main.py-context)
The `main.py` file contains the main code for the chatbot application with a graphical interface. It uses the `tkinter` library to create a GUI that allows users to interact with the chatbot. The `Chatbot` class manages the creation of user interface widgets and the functionalities of tokenization, stemming, corpus management, regex, and POS tagging. The buttons in the GUI allow users to perform these operations and display the results in the text area.

### [text_preprocessing.py](#text_preprocessing.py-context)
The `text_preprocessing.py` file contains the `TextPreprocessing` class, which integrates various text preprocessing functionalities. This class uses methods from other modules to perform sentence and word tokenization, regex tokenization, whitespace tokenization, stemming (Porter, Lancaster, Snowball), lemmatization, stopword removal, n-gram generation, and POS tagging.

### [Corpus.py](#corpus.py-context)
The `Corpus.py` file defines the `NLTKCorpora` class, which provides methods to access various corpora available in the `nltk` library. It includes methods to obtain categories, files, and samples from corpora such as Brown, Gutenberg, Web Text, Reuters, Inaugural Address, Movie Reviews, Shakespeare, WordNet, Names, Stopwords, Swadesh, Words, Conll2000, Conll2002, Treebank, and UDHR. This file is useful for working with predefined linguistic data.

### [Re.py](#re.py-context)
The `Re.py` file defines the `Regex` class, which encapsulates common operations with regular expressions using Python's `re` module. The class includes methods to compile patterns, search, find all occurrences, perform full matches, match at the start of the text, replace patterns, and split text. This file is useful for performing text manipulation operations based on patterns.

### [Stem.py](#stem.py-context)
The `Stem.py` file defines the `Stemmer` class, which provides methods to perform stemming and lemmatization of words using various algorithms available in the `nltk` library. It includes methods for stemming with Porter, Lancaster, and Snowball, and for lemmatization with WordNet. This file is useful for reducing words to their roots or base forms.

### [Tokenize.py](#tokenize.py-context)
The `Tokenize.py` file defines the `Tokenizer` class, which provides methods to tokenize text into words, sentences, and n-grams using various tokenization techniques available in the `nltk` library. It includes methods for word tokenization, regex tokenization, sentence tokenization, Punkt sentence tokenization, whitespace tokenization, and generating n-grams. This file is useful for breaking down text into smaller components for further processing.
