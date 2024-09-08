# main.py

import sys
import os
import tkinter as tk

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'Text_Preprocessing')))

from text_preprocessing import TextPreprocessing
from language_models.bag_of_words import BagOfWords
from language_models.neural_model import NeuralModel

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
        
        # Example training data
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
        
        # Train the BagOfWords model
        self.bow.fit(texts)
        
        # Train the neural model
        self.neural_model.fit(texts, labels, epochs=10, batch_size=2)

    def clear(self):
        # Clear the content of the text area
        self.text_area.delete('1.0', tk.END)

    def process_text(self):
        # Get user input from the text entry
        user_input = self.entry.get()
        
        # Preprocess the user's text
        preprocessed_result = self.preprocessor.preprocess_text(user_input)
        
        # Transform the text using BagOfWords
        bow_features = self.bow.transform([user_input])
        
        # Predict using the neural model
        prediction = self.neural_model.predict([user_input])
        
        # Insert the result into the text area
        self.text_area.insert(tk.END, f"Preprocessed Text: {preprocessed_result}\n")
        self.text_area.insert(tk.END, f"Bag of Words Features: {bow_features.toarray()}\n")
        self.text_area.insert(tk.END, f"Prediction: {prediction}\n")

    def create_widgets(self):
        # Create a welcome label
        self.label = tk.Label(self.root, text="Welcome to Chatbot")
        self.label.pack(pady=20)

        # Create a text entry for user input
        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)

        # Create a text area to display results
        self.text_area = tk.Text(self.root, height=10, width=50)
        self.text_area.pack(pady=10)

        # Create a button to process the text
        self.process_button = tk.Button(self.root, text="Process Text", command=self.process_text)
        self.process_button.pack(pady=10)

        # Create a button to clear the text area
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=10)

        # Create a button to quit the application
        self.quit_button = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit_button.pack(pady=10)

def main():
    # Create the main window
    root = tk.Tk()
    
    # Create the Chatbot application
    app = Chatbot(root)
    
    # Start the main event loop
    root.mainloop()

if __name__ == "__main__":
    main()
