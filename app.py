from flask import Flask, redirect, request, url_for
import os
import model4
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
app = Flask(__name__)
model = model4.load_model()


@app.route("/olivia")
def olivia():
    text = request.args.get('message')
    if not text:
        return "No text provided"

    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(text),
            temperature=0.6,
            max_tokens=250,
        )

    return {
        "olivia": clean_completion(response.choices[0].text),
        "depression_score": model4.test(model, text),
        "sentence": text,
        }

def generate_prompt(text):
    return f"""You are a therapist who helps people with depression, somone says the following text to you, respond in a kind and comforting way but do not prompt for a response {text}"""

def clean_completion(text):
    return text.strip("\n")


if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
