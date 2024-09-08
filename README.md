# Project Summary

## Project Description

- The Chatbot with GUI project is an application that uses the `tkinter` library to create a graphical user interface (GUI) that allows users to interact with various natural language processing (NLP) functionalities. The application includes features for text tokenization, stemming, lemmatization, access to various corpora, the use of regular expressions, and part-of-speech (POS) tagging. These functionalities are integrated into a simple and intuitive GUI.

## Project Purpose

- This project is a prototype of a larger and more complex system aimed at providing advanced tools for natural language processing. The goal is to demonstrate basic capabilities and provide a solid foundation on which to build additional features and improvements.

## Current Chatbot Functionalities

The chatbot currently:

1. **Preprocesses Text**:
   - Uses various techniques for tokenization, stemming, lemmatization, and stopword removal to prepare the text.

2. **Transforms Text**:
   - Converts the text into a numerical representation using the Bag of Words model.

3. **Predicts Classes**:
   - Uses a neural model to make predictions based on the preprocessed and transformed text.

4. **User Interface**:
   - Provides a graphical interface with Tkinter to input text, process it, and display the results.

## How to Run It

1. **Install Dependencies**:
   - Ensure you have Python installed.
   - Install the required libraries: `tkinter`, `nltk`, `tensorflow`, `scikit-learn`, `imblearn`, `matplotlib`.

2. **Run the Application**:
   - Execute the `main.py` file to start the chatbot's graphical interface.
   - Enter text in the input area and click the "Process Text" button to see the preprocessing, transformation, and prediction results.
   - Use the "Clear" button to clear the text area and the "Quit" button to close the application.

In summary, the chatbot takes a text input, preprocesses it, transforms it into a numerical representation, makes a prediction with a neural model, and displays the results to the user.
