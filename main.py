# -*- coding: utf-8 -*-
import sys
import os
import tkinter as tk
import pickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Add the correct path for the word_embeddings module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Text_Preprocessing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'language_models')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'text_generation')))

from text_preprocessing import TextPreprocessing
from language_models.bag_of_words import BagOfWords
from language_models.neural_model import NeuralModel
from text_generation.seq2seq_model import create_seq2seq_model
from text_generation.word_embeddings import create_word_embeddings
from text_generation.teacher_forcing import teacher_forcing
from language_models.dialogpt_model import DialoGPTModel

class Chatbot:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Chatbot")
        
        # Create the UI widgets
        self.create_widgets()
        
        # Initialize the text preprocessor
        self.preprocessor = TextPreprocessing()
        
        # Initialize the BagOfWords model
        self.bow = BagOfWords()
        
        # Initialize the NeuralModel
        vocab_size = 1000
        embedding_dim = 50
        max_length = 10
        self.neural_model = NeuralModel(vocab_size, embedding_dim, max_length)
        
        # Initialize the Seq2Seq model
        input_dim = 10
        output_dim = 10
        latent_dim = 256
        self.seq2seq_model = create_seq2seq_model(input_dim, output_dim, latent_dim)
        
        # Compile the Seq2Seq model
        self.seq2seq_model.compile(optimizer='adam', loss='categorical_crossentropy')
        
        # Initialize the word tokenizer
        self.word_tokenizer = None
        
        # Initialize the DialoGPT model
        self.dialogpt_model = DialoGPTModel()
        
        # Load or train the models
        self.load_or_train_models()

    def load_or_train_models(self):
        try:
            # Check if the models are already saved
            if os.path.exists('bow_model.pkl') and os.path.exists('neural_model.weights.h5') and os.path.exists('seq2seq_model.weights.h5') and os.path.exists('word_tokenizer.pkl'):
                # Load the BagOfWords model
                with open('bow_model.pkl', 'rb') as f:
                    self.bow = pickle.load(f)
                # Load the NeuralModel
                self.neural_model.load_weights('neural_model.weights.h5')
                # Load the Seq2Seq model
                self.seq2seq_model.load_weights('seq2seq_model.weights.h5')
                # Load the word tokenizer
                with open('word_tokenizer.pkl', 'rb') as f:
                    self.word_tokenizer = pickle.load(f)
            else:
                raise ValueError("Models not found, training new models.")
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Training new models...")
            texts = [
                "Hello, how are you?", 
                "I am fine, thank you.",
                "What is your name?",
                "My name is Chatbot.",
                "How can I help you?",
                "Tell me a joke.",
                "Why did the chicken cross the road?",
                "To get to the other side."
            ]
            labels = [0, 1, 0, 1, 1, 0, 1, 0]  # Example labels
            
            # Train the BagOfWords model
            self.bow.fit(texts)
            
            # Train the neural model
            self.neural_model.fit(texts, labels, epochs=10, batch_size=2)
            
            # Create word embeddings
            vocab_size = 50
            max_length = 10
            padded_sequences, self.word_tokenizer = create_word_embeddings(texts, vocab_size, max_length)
            
            # Ensure word_tokenizer is created
            if self.word_tokenizer is None:
                print("Error: Word tokenizer was not created correctly.")
                return
            
            # Train the Seq2Seq model
            input_data = np.random.rand(8, 5, 10)  # Random input data
            target_data = np.random.rand(8, 5, 10)  # Random target data
            teacher_forcing(self.seq2seq_model, input_data, target_data, epochs=10, batch_size=2)
            
            # Save the models
            with open('bow_model.pkl', 'wb') as f:
                pickle.dump(self.bow, f)
            self.neural_model.save_weights('neural_model.weights.h5')
            self.seq2seq_model.save_weights('seq2seq_model.weights.h5')
            with open('word_tokenizer.pkl', 'wb') as f:
                pickle.dump(self.word_tokenizer, f)

    def clear(self):
        self.entry.delete(0, tk.END)
        self.text_area.delete(1.0, tk.END)

    def decode_sequence(self, input_seq, max_length):
        decoder_input = np.zeros((1, 1, 10))  # Dimensione corretta
        decoded_sentence = ''
        stop_condition = False
        sampled_word = ''
        
        for i in range(max_length):
            output_tokens = self.seq2seq_model.predict([input_seq, decoder_input])
            if isinstance(output_tokens, list):
                output_tokens = output_tokens[0]
            
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_word = self.word_tokenizer.index_word.get(sampled_token_index, '')
            
            if sampled_word == '<end>' or sampled_word == '' or sampled_word in decoded_sentence.split():
                stop_condition = True
            else:
                decoded_sentence += ' ' + sampled_word
            
            decoder_input = np.zeros((1, 1, 10))  # Dimensione corretta
            decoder_input[0, 0, sampled_token_index] = 1.0
            
            if stop_condition:
                break
        
        return decoded_sentence.strip()

    def generate_dialogpt_response(self, user_input):
        return self.dialogpt_model.generate_response(user_input)

    def process_text(self):
        user_input = self.entry.get()
        preprocessed_result = self.preprocessor.preprocess_text(user_input)
        bow_features = self.bow.transform([user_input])
        prediction = self.neural_model.predict([user_input])
        
        if self.word_tokenizer is None:
            self.text_area.insert(tk.END, "Error: Word tokenizer was not initialized correctly.\n")
            return
        
        # Generate response using DialoGPT
        response = self.generate_dialogpt_response(user_input)
        
        self.text_area.insert(tk.END, f"User Input: {user_input}\n")
        self.text_area.insert(tk.END, f"Generated Response: {response}\n")

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Welcome to Chatbot")
        self.label.pack(pady=20)

        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)

        self.process_button = tk.Button(self.root, text="View Response", command=self.process_text)
        self.process_button.pack(pady=10)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=10)

        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(pady=10)

        self.text_area = tk.Text(self.root, height=10, width=50)
        self.text_area.pack(pady=10)

def main():
    root = tk.Tk()
    app = Chatbot(root)
    root.mainloop()

if __name__ == "__main__":
    main()
