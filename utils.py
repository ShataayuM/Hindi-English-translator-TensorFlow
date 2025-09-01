import re
import unicodedata

def preprocess_sentence(sentence):
    """Cleans and preprocesses a single sentence."""
    s = str(sentence)
    s = re.sub(r"([?.!,¿])", r" \1 ", s)
    s = re.sub(r'[" "]+', " ", s)
    s = re.sub(r"[^a-zA-Z?.!,¿\u0900-\u097F]+", " ", s)
    s = s.strip()
    return s
