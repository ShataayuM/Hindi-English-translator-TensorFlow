# Hindi-to-English Neural Machine Translation with TensorFlow & Streamlit

This project is a deep learning-based neural machine translation (NMT) system that translates sentences from Hindi to English. The model is built from scratch using TensorFlow and Keras, and the interactive web interface is powered by Streamlit.

**Live Demo:** [https://hindi-english-translator-tensorflow-44wd4tbfyf5xri6bxum9x.streamlit.app](https://hindi-english-translator-tensorflow-44wd4ttbfyf5xn6bxum9xx.streamlit.app/)

![Screenshot of the running Streamlit App](https://i.imgur.com/wY2Y47S.png)

---

## 🛠️ Tech Stack
- **TensorFlow & Keras**: For building and training the sequence-to-sequence model.
- **Pandas**: For data loading and initial manipulation.
- **Streamlit**: For creating and deploying the interactive web application.
- **Python**: The core programming language.

---

## 📜 Model Architecture
The translation model uses a classic **Encoder-Decoder architecture** with a **Bahdanau Attention mechanism**.

- **Encoder**: An LSTM layer processes the input Hindi sentence and encodes it into context vectors (its hidden state and cell state).
- **Attention Mechanism**: The attention layer allows the decoder to focus on relevant parts of the Hindi input sentence when generating each English word. This is crucial for improving translation quality, especially for longer sentences.
- **Decoder**: An LSTM layer takes the encoder's context vectors and the previously generated word to produce the next word in the English translation.

---

## 📊 Dataset
The model was trained on the **IIT Bombay English-Hindi Parallel Corpus**, which contains a large set of sentence pairs. The data was sourced from Kaggle.

- **Dataset Link**: [IIT Bombay English-Hindi Corpus](https://www.kaggle.com/datasets/dhruvagg/hindi-english-parallel-corpus)
- A subset of 50,000 sentence pairs was used for this training session.

---

## 🚀 How to Run this Project

Follow these steps to run the translation app on your local machine.

### 1. Clone the Repository
```bash
git clone [https://github.com/ShatayuM/Hindi-English-translator-TensorFlow.git](https://github.com/ShatayuM/Hindi-English-translator-TensorFlow.git)
cd Hindi-English-translator-TensorFlow
