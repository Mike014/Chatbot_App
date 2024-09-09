# Implementazione del modello Seq2Seq utilizzando Keras
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.models import Model

def create_seq2seq_model(input_dim, output_dim, latent_dim):
    # Encoder
    encoder_inputs = Input(shape=(None, input_dim))
    encoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(None, output_dim))
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

    # Attention
    attention = Attention()([decoder_outputs, encoder_outputs])
    decoder_concat_input = Dense(latent_dim, activation="tanh")(attention)

    # Dense layer
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Model
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return model

def test_create_seq2seq_model():
    input_dim = 10
    output_dim = 10
    latent_dim = 256

    model = create_seq2seq_model(input_dim, output_dim, latent_dim)
    model.summary()

if __name__ == "__main__":
    test_create_seq2seq_model()







