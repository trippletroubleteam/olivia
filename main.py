# flask application to take in messages by the user and feeding it into gpt3 to generate a response that acts like a therapist who helps people with depression

import openai
from flask import Flask, redirect, request, url_for
import os

app = Flask(__name__)
openai.api_key = os.getenv("key")


@app.route("/", methods=("POST", "GET"))
def index():
    if request.method == "POST":
        text = request.form["text"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(text),
            temperature=0.6,
            max_tokens=250,
        )

        return response.choices[0].text

    result = request.args.get("result")
    return result


def generate_prompt(text):
    return f"""You are a therapist who helps people with depression, somone says the following text to you, respond in a kind and comforting way but do not prompt for a response.

            {text}
            """


app.run(port=5000, debug=True, host="0.0.0.0")
