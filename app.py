import streamlit as st
import tensorflow as tf
import json

# Import your model classes and utility functions
from model import Encoder, Decoder, BahdanauAttention
from utils import preprocess_sentence

# --- Load Model and Artifacts ---
@st.cache_resource
def load_model():
    # Define hyperparameters
    units = 512
    embedding_dim = 256
    
    # Load vocabularies
    with open('hi_vec_vocab.json', 'r') as f:
        hi_vocab = json.load(f)
    with open('en_vec_vocab.json', 'r') as f:
        en_vocab = json.load(f)

    # Create and configure vectorization layers
    hi_vec = tf.keras.layers.TextVectorization(max_tokens=len(hi_vocab), standardize=None)
    en_vec = tf.keras.layers.TextVectorization(max_tokens=len(en_vocab), standardize=None)
    
    hi_vec.set_vocabulary(hi_vocab)
    en_vec.set_vocabulary(en_vocab)
    
    BATCH_SIZE = 1

    # Create new model instances
    encoder = Encoder(len(hi_vocab), embedding_dim, units, BATCH_SIZE)
    decoder = Decoder(len(en_vocab), embedding_dim, units, BATCH_SIZE)
    
    # --- THIS IS THE FIX ---
    # Build the models by calling them with dummy inputs. This forces Keras to
    # create all the layer weights, making them ready for loading.
    
    # Build the encoder
    dummy_input = tf.zeros((BATCH_SIZE, 1), dtype=tf.int64)
    initial_hidden_state = encoder.initialize_hidden_state()
    _, _ = encoder(dummy_input, initial_hidden_state)
    
    # Build the decoder
    dummy_enc_output = tf.zeros((BATCH_SIZE, 1, units))
    _ , _, _ = decoder(dummy_input, initial_hidden_state, dummy_enc_output)
    
    # Now that the models are "built", we can load the weights.
    encoder.load_weights('encoder.weights.h5')
    decoder.load_weights('decoder.weights.h5')
    
    return encoder, decoder, hi_vec, en_vec

# --- Translation Function ---
def translate(hindi_sentence, encoder, decoder, hi_vec, en_vec):
    units = encoder.enc_units
    
    hindi_sentence = preprocess_sentence(hindi_sentence)
    inputs = hi_vec([hindi_sentence])
    inputs = inputs.to_tensor()

    hidden = [tf.zeros((1, units)), tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    start_token_index = en_vec.get_vocabulary().index('[START]')
    dec_input = tf.expand_dims([start_token_index], 0)

    result_sentence = ''
    
    for t in range(50):
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        predicted_word = en_vec.get_vocabulary()[predicted_id]

        if predicted_word == '[END]':
            break
            
        result_sentence += predicted_word + ' '
        dec_input = tf.expand_dims([predicted_id], 0)

    return result_sentence.strip()

# --- Streamlit App Interface ---

st.set_page_config(page_title="Hindi to English Translator", layout="centered")
st.title("🧠 Hindi-to-English NMT")
st.write("This app uses a sequence-to-sequence model with attention to translate Hindi text into English.")

# Load the model
encoder, decoder, hi_vec, en_vec = load_model()

# Get user input
hindi_input = st.text_area("Enter a sentence in Hindi:", "आप कैसे हैं?", height=100)

if st.button("Translate"):
    if hindi_input:
        with st.spinner("Translating..."):
            translation = translate(hindi_input, encoder, decoder, hi_vec, en_vec)
        st.success("Translated Sentence:")
        st.write(translation)
    else:
        st.warning("Please enter a sentence to translate.")
