# -*- coding: utf-8 -*-
import sys
import os

# Set terminal encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords from NLTK
nltk.download('stopwords')

class NeuralModel:
    def __init__(self, vocab_size, embedding_dim, max_length):
        # Initialize the tokenizer with a vocabulary size limit
        self.tokenizer = Tokenizer(num_words=vocab_size)
        self.max_length = max_length
        
        # Define the model architecture
        self.model = Sequential([
            Embedding(vocab_size, embedding_dim, input_length=max_length),
            Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))),
            Dropout(0.5),
            Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01))),
            Dropout(0.5),
            LSTM(16, kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid', kernel_regularizer=l2(0.01))
        ])
        
        # Build the model with the input shape
        self.model.build(input_shape=(None, max_length))
        
        # Compile the model with binary cross-entropy loss and Adam optimizer
        self.model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

    def preprocess_text(self, texts):
        # Set of English stopwords
        stop_words = set(stopwords.words('english'))
        processed_texts = []
        
        for text in texts:
            # Remove non-word characters
            text = re.sub(r'\W', ' ', text)
            # Replace multiple spaces with a single space
            text = re.sub(r'\s+', ' ', text)
            # Convert text to lowercase
            text = text.lower()
            # Remove stopwords
            text = ' '.join([word for word in text.split() if word not in stop_words])
            processed_texts.append(text)
        
        return processed_texts

    def fit(self, texts, labels, epochs=50, batch_size=16):
        # Preprocess the input texts
        texts = self.preprocess_text(texts)
        # Convert texts to sequences of integers
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        padded_sequences = tf.convert_to_tensor(padded_sequences, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(padded_sequences.numpy(), labels.numpy(), test_size=0.2, random_state=42)
        
        # Convert NumPy arrays to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
        
        # Apply SMOTE to oversample the minority class
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train.numpy(), y_train.numpy())
        
        # Convert NumPy arrays to TensorFlow tensors
        X_train_res = tf.convert_to_tensor(X_train_res, dtype=tf.float32)
        y_train_res = tf.convert_to_tensor(y_train_res, dtype=tf.float32)
        
        # Callbacks for early stopping and learning rate reduction
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        
        # Train the model
        history = self.model.fit(X_train_res, y_train_res, epochs=epochs, batch_size=batch_size, 
                                 validation_data=(X_test, y_test), callbacks=[early_stopping, reduce_lr])
        
        # Plot training history
        self.plot_history(history)

    def plot_history(self, history):
        # Plot training and validation accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot training and validation loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.show()

    def predict(self, texts):
        # Preprocess the input texts
        texts = self.preprocess_text(texts)
        # Convert texts to sequences of integers
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        padded_sequences = tf.convert_to_tensor(padded_sequences, dtype=tf.float32)
        # Predict using the trained model
        return self.model.predict(padded_sequences)

    def evaluate(self, texts, labels):
        # Preprocess the input texts
        texts = self.preprocess_text(texts)
        # Convert texts to sequences of integers
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Pad sequences to ensure uniform length
        padded_sequences = pad_sequences(sequences, maxlen=self.max_length)
        padded_sequences = tf.convert_to_tensor(padded_sequences, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        
        # Predict using the trained model
        predictions = self.model.predict(padded_sequences)
        predictions = (predictions > 0.5).astype(int)
        
        # Calculate precision, recall, and F1 score
        precision = precision_score(labels, predictions)
        recall = recall_score(labels, predictions)
        f1 = f1_score(labels, predictions)
        
        # Print evaluation metrics
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')

    def save_weights(self, filepath):
        """Salva i pesi del modello nel file specificato."""
        self.model.save_weights(filepath)
        print(f"Pesi salvati in {filepath}")

    def load_weights(self, filepath):
        """Carica i pesi del modello dal file specificato."""
        self.model.load_weights(filepath)
        print(f"Pesi caricati da {filepath}")

# Example usage of the NeuralModel class
if __name__ == "__main__":
    texts = [
        "This is an example of document.", 
        "This is another one",
        "Yet another example document.",
        "More examples to train the model.",
        "This is a positive example.",
        "This is a negative example.",
        "Positive example here.",
        "Negative example here."
    ]
    labels = [0, 1, 0, 1, 1, 0, 1, 0]  # Example labels
    vocab_size = 1000
    embedding_dim = 50
    max_length = 10

    neural_model = NeuralModel(vocab_size, embedding_dim, max_length)
    neural_model.fit(texts, labels, epochs=10, batch_size=2)
    neural_model.save_weights('neural_model.weights.h5')
    neural_model.load_weights('neural_model.weights.h5')
    predictions = neural_model.predict(["This is a new example"])
    
    # Print predictions directly
    for prediction in predictions:
        print(prediction)
    
    # Evaluate the model
    neural_model.evaluate(texts, labels)
