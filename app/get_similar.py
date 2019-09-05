#
# Author: Scott Sievert (https://github.com/stsievert)
#
import nltk
import numpy as np
from app.models import InferSent
import torch
from time import time
from typing import TypeVar, List
from scipy.spatial.distance import cdist, squareform
from toolz import topk

model = None
ArrayLike = TypeVar("ArrayLike")


def initialize(download=False):
    global model
    if download:
        nltk.download("punkt")

    model_version = 2
    MODEL_PATH = "encoder/infersent%s.pkl" % model_version
    params_model = {
        "bsize": 64,
        "word_emb_dim": 300,
        "enc_lstm_dim": 2048,
        "pool_type": "max",
        "dpout_model": 0.0,
        "version": model_version,
    }
    if model is not None:
        return model
    model = InferSent(params_model)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        W2V_PATH = (
            "GloVe/glove.840B.300d.txt"
            if model_version == 1
            else "fastText/crawl-300d-2M.vec"
        )
        model.set_w2v_path(W2V_PATH)
        model.build_vocab_k_words(K=100_000)
        return model
    except:
        raise ValueError("Run sim_init.sh")


def get_embedding(sents: List[str], verbose=False) -> ArrayLike:
    model = initialize(download=False)
    embeddings = model.encode(sents, bsize=128, tokenize=False, verbose=verbose)
    return embeddings


def get_closest_indices(new_sent, feats, k=3):
    """
    Parameters
    ----------
    new_sent : str
        New sentence
    feats : array-like, 2d
        Features for different sentences.

    Returns
    -------
    idx : int
        Index of ``feats`` that is closest to ``new_sent``.

    """
    model = initialize(download=False)
    feat = model.encode([new_sent], bsize=128, tokenize=False)
    dists = cdist(feat, feats, "cosine")
    assert dists.shape[0] == 1
    dists = dists[0]
    vals = topk(k, -dists)
    idxs = [k for k, v in enumerate(dists) if -v in vals]
    return idxs[::-1]


def get_most_similar(new, sentences):
    feats = get_embedding(sentences)
    idxs = get_closest_indices(new, feats, k=3)  # 0.0936 secs
    o = sentences[idxs[0]]
    tw = sentences[idxs[1]]
    th = sentences[idxs[2]]
    return [o, tw, th]


if __name__ == "__main__":
    initialize(download=False)  # 9.87 without download

    sents = [
        "Everyone really likes the newest benefits",
        "The Government Executive articles housed on the website are not able to be searched .",
        "I like him for the most part , but would still enjoy seeing someone beat him .",
        "My favorite restaurants are always at least a hundred miles away from my house .",
        "I know exactly .",
    ]
    assert len(sents) == 5
    feats = get_embedding(sents)  # 0.6497 secs (0.12 secs / sentence)

    new_sent = "I'm not really sure..."
    idxs = get_closest_indices(new_sent, feats, k=2)  # 0.0936 secs

    closest_sentence = sents[idxs[0]]
    assert closest_sentence == "I know exactly ."
    second_closest = sents[idxs[1]]
    assert second_closest == 'Everyone really likes the newest benefits'
