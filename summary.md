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

4. **Generates Responses**:
   - Uses the pre-trained DialoGPT model to generate responses based on user input.

5. **User Interface**:
   - Provides a graphical interface with Tkinter to input text, process it, and display the results.

## Pre-Trained Model Used

- **DialoGPT**: The DialoGPT model is a variant of the GPT-2 (Generative Pre-trained Transformer 2) model developed by OpenAI. It has been trained on a large corpus of Reddit conversations to generate coherent and contextually relevant responses in a dialogue. DialoGPT is particularly suitable for chatbot applications due to its ability to understand and generate natural language fluently.

## How to Use It

1. **Install Dependencies**:
   - Ensure you have Python installed.
   - Install the required libraries: `tkinter`, `nltk`, `tensorflow`, `scikit-learn`, `imblearn`, `matplotlib`, `transformers`, `torch`.

```bash
pip install <libraries>
``` 

2. **Download and Run the Application**:
   - Open Git Bash terminal.
   - Clone the repository:
     
```bash
git clone https://github.com/Mike014/Chatbot_App.git
```   

2. **Navigate to the project directory, Run the App:**:
   - Run the application

```bash
cd Chatbot_App
python main.py
```
   - Enter text in the input area and click the "Process Text" button to see the preprocessing, transformation, and prediction results.
   - Use the "Clear" button to clear the text area and the "Quit" button to close the application.

## Modules and Libraries Used

### [main.py](#main.py-context)
- **Description**: Manages the chatbot's graphical interface and coordinates the interaction between various modules.
- **Libraries**: `tkinter`, `pickle`, `numpy`.

### [language_models\__init__.py](#language_models\__init__.py-context)
- **Description**: Initializes the language model modules.
- **Modules**: `BagOfWords`, `NeuralModel`, `DialoGPTModel`.

### [bag_of_words.py](#bag_of_words.py-context)
- **Description**: Implements the Bag of Words model for numerical text representation.
- **Libraries**: `sklearn.feature_extraction.text.CountVectorizer`.

### [dialogpt_model.py](#dialogpt_model.py-context)
- **Description**: Implements a dialogue model based on DialoGPT.
- **Libraries**: `transformers`, `torch`.

### [neural_model.py](#neural_model.py-context)
- **Description**: Implements a neural model for text classification.
- **Libraries**: `tensorflow`, `nltk`, `sklearn`, `imblearn`, `numpy`, `matplotlib`.

### [text_generation\__init__.py](#text_generation\__init__.py-context)
- **Description**: Initializes the text generation modules.

### [seq2seq_model.py](#seq2seq_model.py-context)
- **Description**: Implements a Seq2Seq model for text generation.
- **Libraries**: `tensorflow.keras.layers`, `tensorflow.keras.models`.

### [teacher_forcing.py](#teacher_forcing.py-context)
- **Description**: Implements the Teacher Forcing technique for training Seq2Seq models.
- **Libraries**: `numpy`.

### [word_embeddings.py](#word_embeddings.py-context)
- **Description**: Manages word representations using embeddings.
- **Libraries**: `tensorflow.keras.preprocessing.text`, `tensorflow.keras.preprocessing.sequence`.

### [Text_Preprocessing\__init__.py](#text_preprocessing\__init__.py-context)
- **Description**: Initializes the text preprocessing modules.
- **Modules**: `Tokenizer`, `Stemmer`, `Regex`, `NLTKCorpora`, `TextPreprocessing`.

### [text_preprocessing.py](#text_preprocessing.py-context)
- **Description**: Implements various text preprocessing techniques.
- **Libraries**: `nltk`, `Text_Preprocessing.Corpus`, `Text_Preprocessing.Re`, `Text_Preprocessing.Stem`, `Text_Preprocessing.Tokenize`.

In summary, the chatbot takes a text input, preprocesses it, transforms it into a numerical representation, makes a prediction with a neural model, generates a response with DialoGPT, and displays the results to the user.

