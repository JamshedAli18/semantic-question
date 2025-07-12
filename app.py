import streamlit as st
import re
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('duplicate_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

MAX_LEN = 50  # same as used during training

# Text cleaning
def clean(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

# Streamlit UI
st.title("Duplicate Question Prediction")

q1 = st.text_input("Enter Question 1")
q2 = st.text_input("Enter Question 2")

if st.button("Predict"):
    combined = clean(q1) + " " + clean(q2)
    seq = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    pred = model.predict(padded)[0][0]

    st.write("### Result:")
    st.write("🔁 Duplicate" if pred >= 0.5 else "❌ Not Duplicate")
    st.write(f"**Probability:** {pred:.2f}")
