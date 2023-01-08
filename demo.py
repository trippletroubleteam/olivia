import streamlit as st
import os
os.environ["OPENAI_KEY"] == st.secrets["OPENAI_KEY"]
import model4
import nltk
nltk.download("stopwords")
from app import *

model = model4.load_model()

st.title("UnisonAI NLP Analysis of Conversations")


text = st.text_input("Enter atleast 5 sentences talking about how you feel")

score = model4.test(model, text)

st.text("User is at a " + str(round(score/2, 2)) + "%" + " risk of being depressed.")

st.progress(score/200)

st.subheader("Olivia's Reponse:")

st.write(generate_completion(therapy_prompt(text)))


st.subheader("Parent Suggestion:")

st.write(generate_completion(counselor_prompt + text))
