import requests
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--encoder", help="Encodes models with different PyTorch devices", default="cuda")
parser.add_argument("-na", "--new-auth", help="Fetches a new auth token for Reddit's API", action="store_true")
args = parser.parse_args()
load_dotenv()

username = "garbage22507"
client_id = "yKxWouPF79VSwyXRWL4jJQ"
base_headers = {'user-agent': f'Windows PC:resolve:v1.0.0 (by /u/{username})'}
newAuth = args.new_auth
model = SentenceTransformer("bert-base-nli-mean-tokens")
post_limit = 10000

def getAuthToken():
    auth = requests.auth.HTTPBasicAuth(client_id, os.getenv("REDDIT_SECRET"))

    data = {'grant_type': 'password',
            'username': username,
            'password': os.getenv("REDDIT_PASSWORD")}


    res = requests.post('https://www.reddit.com/api/v1/access_token', auth=auth, data=data, headers=base_headers).json()

    TOKEN = res.get('access_token')
    if not TOKEN:
        raise Exception(res)

    print(TOKEN)
    return TOKEN


def writeAuthToken(token):
    password = os.getenv("REDDIT_PASSWORD")
    secret = os.getenv("REDDIT_SECRET")

    with open(".env", "w+") as file:
        file.write(f"REDDIT_SECRET={secret}\nREDDIT_PASSWORD={password}\nREDDIT_API_KEY={token}")

    load_dotenv()


def df_from_response(data_list, res):
    count = 0
    for post in res['data']['children']:
        data_list.append({
            'title': post['data']['title'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'ups': post['data']['ups'],
            'downs': post['data']['downs'],
            'score': post['data']['score'],
            "id": post['data']['id'],
            'created_utc': datetime.fromtimestamp(post['data']['created_utc']).strftime('%Y-%m-%dT%H:%M:%SZ'),
            'kind': post['kind'],
            'text': post['data']['selftext']
            })
        count += 1
    return data_list, count


def fetchData(sub, total, newAuth=False):
    print("New Auth:", newAuth)

    if newAuth:
        token = getAuthToken()
        writeAuthToken(token)
    else:
        token = os.environ.get("REDDIT_API_KEY")
        print(token)

    headers = {**base_headers, **{'Authorization': f"bearer {token}"}}
    params = {"limit": "100", "t": "all"}
    data_list = []
    totalIters = int(total/100)

    for i in range(totalIters):
        # make request
        res = requests.get(f"https://oauth.reddit.com/r/{sub}", headers=headers, params=params)
        #print(res.text)
        res = res.json()

        if not res['data'].get('children'):
            break

        data_list, count = df_from_response(data_list, res)
        row = data_list[len(data_list)-1]


        fullname = row['kind'] + '_' + row['id']

        params['after'] = fullname

        print(f"Added {count} entries")
        print(f"Total Entries: {len(data_list)} entries")
        print("\n")


    print("Total Entries:", len(data_list))

    df = pd.DataFrame.from_records(data_list)
    df = df[['text']]
#    df = df.rename(columns={"title": "text"})

    df = df.iloc[2:]

    print(df.head())

    return df


def createRedditEmbeddings(df, output_path=None):
    df_posts = df[['text']]
    posts = df_posts['text'].to_list()

    model = SentenceTransformer("bert-base-nli-mean-tokens")

    print(f"Encoding the corpus with {len(df.index)} entries. This might take a while...")
    post_embeddings = model.encode(posts, show_progress_bar=True, device=args.encoder)

    if output_path:
        print(f"Saving embeddings to {output_path}")

        with open(output_path, "wb") as file:
            pickle.dump({'posts': posts, 'embeddings': post_embeddings}, file)

    return post_embeddings


def fromCSV():
    df = pd.read_csv('reddit.csv')
    df = df.iloc[2:]
    df = df[df.is_depression != 0]
    df = df[['clean_text']]
    df = df.rename(columns={"clean_text": "text"})

    print(df.head())
    createRedditEmbeddings(df, output_path="reddit_embeddings.pkl")


def main():
    queries = ["SuicideWatch/hot", "depression/hot", "depression/new", "depression/top", "SuicideWatch/top", "SuicideWatch/new", ]
    df = pd.DataFrame()
    for query in queries:
        df1 = fetchData(query, post_limit, newAuth=newAuth)
        df = pd.concat([df, df1])
    df.reset_index(inplace=True)
    embeddings = createRedditEmbeddings(df, output_path="reddit_embeddings.pkl")


if __name__ == "__main__":
    fromCSV()
