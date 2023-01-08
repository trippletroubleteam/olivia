from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np

path = "depression.txt"
model = SentenceTransformer("bert-base-nli-mean-tokens")


def loadRedditEmbeddings(path="reddit_embeddings.pkl"):
    print(f"Loading embeddings from {path}")
    with open(path, "rb") as file:
        cache_data = pickle.load(file)
        posts = cache_data.get('posts')
        embeddings = cache_data.get('embeddings')

        if not posts or not embeddings.any():
            raise Exception("Embedding pkl is not formatted correctly.")

    return posts, embeddings


def predict(sentence, embeddings):
    embeddings.shape

    test_sentence = model.encode(sentence)
    test_sentence.shape

    score = cosine_similarity(
    [test_sentence],
    embeddings
    )

    return score


def getSimilarity(text, embeddings):
    scores = predict(text, embeddings)[0].tolist()
    max = np.max(scores)
    index = scores.index(max)

    return np.average(scores), max, index



def analayzeText(text):
    posts, embeddings = loadRedditEmbeddings()
    average, max, index = getSimilarity(text, embeddings)
    return average, max, posts[index]



def main():
    with open(path) as file:
        text = file.read()

    average, max, post = analayzeText(text)

    print("Submitted Text:", text)
    print("Depression Probability:", (average * 100))
    print("Most similar with post", post)
    print("With a score of", max * 100)

if __name__ == "__main__":
    main()
