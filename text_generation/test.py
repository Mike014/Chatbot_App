# -*- coding: utf-8 -*-

import sys
import os
import numpy as np

# Aggiungi il percorso del modulo text_preprocessing e language_models al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Text_Preprocessing')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'language_models')))

from text_generation.seq2seq_model import create_seq2seq_model
from Text_Preprocessing.Tokenize import Tokenizer
from teacher_forcing import teacher_forcing
from word_embeddings import create_word_embeddings

def test_seq2seq_with_preprocessing():
    input_dim = 10
    output_dim = 10
    latent_dim = 256

    # Crea il modello Seq2Seq
    model = create_seq2seq_model(input_dim, output_dim, latent_dim)
    model.summary()

    # Compila il modello
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Esempio di testo
    texts = ["this is an example of text", "another example text"]

    # Pre-elaborazione del testo
    tokenizer = Tokenizer()
    tokenized_texts = [tokenizer.word_tokenize(text) for text in texts]

    # Stampa i testi tokenizzati
    print(tokenized_texts)

    # Creazione di word embeddings
    vocab_size = 50
    max_length = 10
    padded_sequences, word_tokenizer = create_word_embeddings(texts, vocab_size, max_length)

    # Stampa le sequenze con padding
    print("Padded Sequences:")
    print(padded_sequences)

    # Creazione di dati di input e target per il test di teacher_forcing
    input_data = np.random.rand(2, 5, input_dim)  # Dati di input casuali
    target_data = np.random.rand(2, 5, output_dim)  # Dati di target casuali

    # Esegui il test di teacher_forcing
    teacher_forcing(model, input_data, target_data, epochs=1, batch_size=1)

if __name__ == "__main__":
    test_seq2seq_with_preprocessing()
