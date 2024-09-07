import tkinter as tk
from Text_Preprocessing.Tokenize import Tokenizer
from Text_Preprocessing.Re import Regex
from Text_Preprocessing.Stem import Stemmer
from Text_Preprocessing.Corpus import NLTKCorpora


class Chabot():
    def __init__(self, root): 
        self.root = root
        self.root.title("Chabot")
        self.tokenizer = Tokenizer()
        self.regex = Regex()
        self.stemmer = Stemmer()
        self.corpus = NLTKCorpora()
        self.create_widgets()  
    
    def create_widgets(self):
        self.label = tk.Label(self.root, text="Welcome to Chabot")
        self.label.pack(pady=20)
        
        self.entry = tk.Entry(self.root, width=50)
        self.entry.pack(pady=10)
        
        self.text_area = tk.Text(self.root, height=10, width=50)
        self.text_area.pack(pady=10)
        
        self.button = tk.Button(self.root, text="Chiudi", command=self.root.quit)
        self.button.pack(pady=10)
        
        # create button clear
        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(pady=10)
       
        self.tokenize_button = tk.Button(self.root, text="Tokenize", command=self.tokenize)
        self.tokenize_button.pack(pady=10)
        
        self.stem_button = tk.Button(self.root, text="Stem", command=self.stem)
        self.stem_button.pack(pady=10)
        
        self.corpus_button = tk.Button(self.root, text="Corpus", command=self.corpora)
        self.corpus_button.pack(pady=10)
        
        self.regex_button = tk.Button(self.root, text="Regex", command=self.run_regex)
        self.regex_button.pack(pady=10)
        print("Regex button created")  # Debug
        
    def clear(self):
        self.text_area.delete('1.0', tk.END)
        
    def tokenize(self):
        sentence = self.entry.get() 
        words = self.tokenizer.word_tokenize(sentence)
        sentences = self.tokenizer.sent_tokenize(sentence)
        result = f"Words: {words}\nSentences: {sentences}\n"
        self.text_area.insert(tk.END, result)
        
    def stem(self):
        self.user_sentence = self.entry.get()
        words = [word for word in self.user_sentence.split()]
        porter_stems = [self.stemmer.porter_stem(word) for word in words]
        lancaster_stems = [self.stemmer.lancaster_stem(word) for word in words]
        snowball_stems = [self.stemmer.snowball_stem(word) for word in words]
        lemmatized_words = [self.stemmer.lemmatize(word) for word in words]
        result = f"Porter: {porter_stems}\nLancaster: {lancaster_stems}\nSnowball: {snowball_stems}\nLemmatized: {lemmatized_words}\n"
        self.text_area.insert(tk.END, result)
        
    def corpora(self):
        self.user_sentence = self.entry.get()
        words = [word for word in self.user_sentence.split()]
        stopwords_list = set(self.corpus.get_all_stopwords())
        stopwords_in_sentence = [word for word in words if word.lower() in stopwords_list]
        result = f"Stopwords in sentence: {stopwords_in_sentence}\n"
        self.text_area.insert(tk.END, result)
        
    def run_regex(self):
        self.user_sentence = self.entry.get()
        words = [word for word in self.user_sentence.split()]
        pattern = r'\b\w{4,}\b'  
        words_with_4_chars = [word for word in words if self.regex.fullmatch(pattern, word)]
        print(f"Words with 4 or more characters: {words_with_4_chars}")  # Debug
        result = f"Words with 4 or more characters: {words_with_4_chars}\n"
        self.text_area.insert(tk.END, result)
        
        
def main():
    root = tk.Tk() # Create the main window 
    app = Chabot(root) # Create the Chatbot application
    root.mainloop() # Start the main event loop

if __name__ == "__main__":
    main()
