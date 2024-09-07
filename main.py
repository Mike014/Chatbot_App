import tkinter as tk
from Text_Preprocessing.text_preprocessing import TextPreprocessing
import os 

class Chatbot:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("Chatbot")
        
        # Create the UI widgets
        self.create_widgets()
        
        # Initialize the text preprocessor
        self.preprocessor = TextPreprocessing()

    def clear(self):
        # Clear the content of the text area
        self.text_area.delete('1.0', tk.END)

    def process_text(self):
        # Get user input from the text entry
        user_input = self.entry.get()
        
        # Preprocess the user's text
        result = self.preprocessor.preprocess_text(user_input)
        
        # Insert the result into the text area
        self.text_area.insert(tk.END, str(result))

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
