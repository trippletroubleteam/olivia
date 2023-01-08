import utils
from pathlib import Path
import streamlit as st
import os
if st.secrets.get("OPENAI_KEY"):
    os.environ["OPENAI_KEY"] = st.secrets['OPENAI_KEY']

file = Path("./model2/bert_model.h5")
if not file.is_file():
    print("File does not exist, downloading")
    os.mkdir("model2")
    utils.downloadFile("bert_model.h5", output_path="./model2/bert_model.h5")

import model4
from app import generate_completion, therapy_prompt, counselor_prompt
import nltk

nltk.download("stopwords")
model = model4.load_model()


st.title("UnisonAI NLP Analysis of Conversations")


text = st.text_input("Enter atleast 5 sentences talking about how you feel")

if text:
    score = model4.test(model, text)

    st.text("User is at a " + str(round(score/2, 2)) + "%" + " risk of being depressed.")

    st.progress(score/200)

    st.subheader("Olivia's Reponse:")

    st.write(generate_completion(therapy_prompt(text)))


    st.subheader("Parent Suggestion:")

    st.write(generate_completion(counselor_prompt(text)))
