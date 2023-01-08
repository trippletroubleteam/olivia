from flask import Flask, redirect, request, url_for
import os
import model4
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_KEY")
app = Flask(__name__)
model = model4.load_model()

congrats_prompt = "In one concise sentence, congratulate a parent on how well they are raising their child."
help_prompt = "You are a smart family counselor tasked with advising a parent to check up on their child who is not doing too well in one concise sentence."
danger_prompt = "You are tasked with alerting and explaining to a parent that their child is in danger of hurting themselves and must seek help in one concise sentence."
counselor_prompt = "You are a smart family counselor and a child has just said the following to you, in one concise sentence tell the parents how they should react based on this information. "

@app.route("/olivia")
def olivia():
    text = request.args.get('message')
    if not text:
        return "No text provided"
    text = clean_completion(text)

    depression_score = model4.test(model, text)

    return {
        "olivia": generate_completion(therapy_prompt(text)),
        "depression_score": depression_score/200,
        "sentence": text,
        "suggestion": generate_completion(counselor_prompt + text)
        }


def generate_completion(prompt, *, temperature=0.6, max_tokens=512, **kwargs):
    response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

    return clean_completion(response.choices[0].text)


def therapy_prompt(text):
    return f"You are a smart therapist who helps people with depression, somone says the following to you, respond in a kind and comforting way but do not prompt for a response {text}"


def clean_completion(text):
    return text.replace("\n", "")


def generate_suggestion(percent):
    if percent == 0:
        response = generate_completion(therpay_prompt)
    elif percent > 0 and percent < 100:
        response = generate_completion(therpay_prompt)
    elif percent > 100:
        response = generate_completion(therpay_prompt)

    return clean_completion(response)


if __name__ == "__main__":
    app.run(port=5000, debug=True, host="0.0.0.0")
