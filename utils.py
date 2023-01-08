import toml
import json
import streamlit as st
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from pathlib import Path

#Converts a JSON file, such as a credentials file, to a TOML
def json_to_toml(input_file, output_file="secrets.toml"):
    with open(input_file) as json_file:
        json_text = json_file.read()

    config = {"textkey": json_text}
    toml_config = toml.dumps(config)

    with open(output_file, "w") as target:
        target.write(toml_config)


def downloadFile(download_path, output_path=None):
    if not output_path:
        output_path = download_path

    if not firebase_admin._apps:
        key_dict = json.loads(st.secrets["textkey"])
        cred = credentials.Certificate(key_dict)
        firebase_admin.initialize_app(cred)

    bucket = storage.bucket("unison-3ae0e.appspot.com")

    blob = bucket.blob(download_path)
    blob.download_to_filename(output_path)

    print(f"Downloaded {download_path} and saved too: {output_path}")
