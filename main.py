import tkinter as tk
from Text_Preprocessing.text_preprocessing import TextPreprocessing
import os 

class Chabot():
    def __init__(self, root):
        self.root = root
        self.root.title("Chabot")
        self.create_widgets()
        self.preprocessor = TextPreprocessing()

    def clear(self):
        self.text_area.delete('1.0', tk.END)

    def process_text(self):
        user_input = self.entry.get()
        result = self.preprocessor.preprocess_text(user_input)
        self.text_area.insert(tk.END, str(result))

    def create_widgets(self):
        self.label = tk.Label(self.root, text="Welcome to Chabot")
        self.label.pack(pady=20)

        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)

        self.text_area = tk.Text(self.root, height=10, width=50)
        self.text_area.pack(pady=10)

        self.process_button = tk.Button(self.root, text="Process Text", command=self.process_text)
        self.process_button.pack(pady=10)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=10)

        self.quit_button = tk.Button(self.root, text="Chiudi", command=self.root.quit)
        self.quit_button.pack(pady=10)

def main():
    root = tk.Tk()  # Create the main window
    app = Chabot(root)  # Create the Chatbot application
    root.mainloop()  # Start the main event loop

if __name__ == "__main__":
    main()
