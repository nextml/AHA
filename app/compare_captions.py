#
# Author: Scott Sievert (https://github.com/stsievert)
#
import os
import yaml
from io import StringIO
from functools import lru_cache
from typing import Dict, TypeVar

import requests
import pandas as pd
import joblib

from app import caption_features

# ArrayLike = Union[np.ndarray, pd.DataFrame, dask.dataframe.DataFrame,
#                   cupy.ndaray, ...]
ArrayLike = TypeVar("ArrayLike")

ESTs = joblib.load("./models.joblib")
nlp = None

# fmt: off
cols = [
    "ENT_CARDINAL", "ENT_DATE", "ENT_FAC", "ENT_GPE", "ENT_LOC",
    "ENT_MONEY", "ENT_NORP", "ENT_ORDINAL", "ENT_ORG", "ENT_PERSON",
    "ENT_PRODUCT", "ENT_QUANTITY", "ENT_TIME", "ENT_WORK_OF_ART",
    "POS_ADJ", "POS_ADP", "POS_ADV", "POS_AUX", "POS_CCONJ", "POS_DET",
    "POS_INTJ", "POS_NOUN", "POS_NUM", "POS_PART", "POS_PRON",
    "POS_PROPN", "POS_PUNCT", "POS_SYM", "POS_VERB", "POS_X", "TAG_''",
    "TAG_-LRB-", "TAG_-RRB-", "TAG_.", "TAG_:", "TAG_CC", "TAG_CD",
    "TAG_DT", "TAG_EX", "TAG_FW", "TAG_HYPH", "TAG_IN", "TAG_JJ",
    "TAG_JJR", "TAG_JJS", "TAG_MD", "TAG_NFP", "TAG_NN", "TAG_NNP",
    "TAG_NNPS", "TAG_NNS", "TAG_PDT", "TAG_POS", "TAG_PRP", "TAG_PRP$",
    "TAG_RB", "TAG_RBR", "TAG_RBS", "TAG_RP", "TAG_TO", "TAG_UH",
    "TAG_VB", "TAG_VBD", "TAG_VBG", "TAG_VBN", "TAG_VBP", "TAG_VBZ",
    "TAG_WDT", "TAG_WP", "TAG_WRB", "TAG_XX", "TAG_``", "is_currency",
    "is_digit", "is_lower", "is_oov", "is_punct", "is_stop", "is_title",
    "is_upper", "joke_quarter", "joke_words", "like_num",
    "max_len_noun_phrase", "max_perplexity", "mean_perplexity",
    "median_perplexity", "min_perplexity", "num_alpha", "num_chars",
    "num_noun_chunks", "num_noun_phrases", "num_proper_nouns",
    "num_sentences", "num_sents", "num_stop", "num_syllables",
    "num_tokens", "num_words", "readability_ARI", "readability_flesch",
    "sentiment_polarity", "sentiment_subjectivity", "sim_anomaly_max",
    "sim_context_max", "sim_diff_90_percentile", "sim_diff_max",
    "sim_diff_mean", "sim_diff_median"
]
# fmt: on


def _get_model(contest):
    return ESTs[contest]


def initialize():
    global nlp
    if nlp is None:
        nlp = caption_features.load()


def _get_NLP(words):
    @lru_cache()
    def _helper(words):
        return nlp(" ".join(list(words)))

    return _helper(tuple(words))


def predict(diff: ArrayLike, contest: int) -> Dict[str, float]:
    """
    Parameters
    ----------
    diff : array-like
        Difference of feature vectors for captions. i.e., ``diff = f1 -
        f2`` where ``f1 = get_features(caption: str, contest: int)``
    contest : int
        Which contest this prediction if for

    Returns
    -------
    info : Dict[str, float]
        With keys ``funnier`` and ``proba``. ``funnier == 1`` indicates
        the left caption in ``diff`` is funnier. ``proba`` is estimate
        of the probability it's in that class.

    Notes
    -----
    Future work is to add a ``feature_importance`` key to the return
    dictionary.

    """
    est = ESTs[contest]
    diff = [diff]
    label = est.predict(diff)
    proba = est.predict_proba(diff)[:, 1]
    return {"funnier": label.item(), "proba": proba.item()}


@lru_cache()
def get_features(c: str, contest: int):
    """
    Parameters
    ----------
    caption : str
        The caption to get features for
    contest : int
        Which contest is this testing on?

    Returns
    -------
    features : pd.Series
        Features for the caption.
        These features have the same keys regardless of which
        ``contest`` is passed (and in the same orde too).

    """
    context, anom = get_meta(contest)
    C = _get_NLP(context)
    A = _get_NLP(anom)
    features = caption_features.stats(c, C, A)

    f = pd.Series({k: features.get(k, 0) for k in cols})
    f.sort_index(inplace=True)
    return f


def get_cached_df(contest, alg_label, verbose=True):
    fname = f"{contest}-round2-{alg_label}-responses.csv"
    DIR = "input-data/"
    if fname not in os.listdir(DIR):
        url = f"https://github.com/nextml/caption-contest-data/raw/master/contests/responses/{fname}.zip"
        cmd = f"wget {url} -O input-data/{fname}.zip"
        if verbose:
            print("$", cmd)
        os.system(cmd)
        cmd = f"cd input-data/; unzip {fname}.zip"
        if verbose:
            print("$", cmd)
        os.system(cmd)
    df = pd.read_csv(DIR + fname)
    if "Unnamed: 0" in df:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    return df


@lru_cache()
def get_meta(contest):
    base = "https://raw.githubusercontent.com/nextml/caption-contest-data/master/"
    fname = "contests/metadata/anomalies.yaml"
    r = requests.get(base + fname)
    with StringIO(r.text) as f:
        anoms = yaml.load(f, Loader=yaml.FullLoader)

    fname = "contests/metadata/contexts.yaml"
    r = requests.get(base + fname)
    with StringIO(r.text) as f:
        contexts = yaml.load(f, Loader=yaml.FullLoader)
    return contexts[contest], anoms[contest]


@lru_cache()
def compare_captions(c1, c2, contest):
    f1 = get_features(c1, contest)
    f2 = get_features(c2, contest)
    diff = f1 - f2
    info = predict(diff, contest)
    info.update(
        {
            "caption_pos": c1,
            "caption_neg": c2,
            "contest": contest
        }
    )
    return info


def rank_captions(caps, contest):
    out = []
    for c1 in caps:
        for c2 in caps:
            if c1 == c2:
                continue
            out.append(compare_captions(c1, c2, contest))
    out = pd.DataFrame(out)
    pairwise = out.pivot_table(
        index="caption_neg",
        columns="caption_pos",
        values="funnier",
    )
    borda = pairwise.sum(skipna=True)
    borda.sort_values(ascending=False, inplace=True)
    borda += borda.min()
    borda /= borda.max()
    return borda


if __name__ == "__main__":
    initialize()  # 15.98s

    contest = 530
    captions = [
        "The latest polls show you hanging on by a thread.",
        "The Queen says she wants half of everything.",
        "It appears your character is getting cut from Season 7.",
        "It's a recall notice from the Acme Twine and Rope company.",
        "First off, we need to renew your life insurance policy.",
        "It's a gift from your eldest son.",
    ]
    p = rank_captions(captions, contest)
    #  f1 = get_features(top_caption, contest)  # 0.686s
    #  f2 = get_features("foo", contest)  # 0.468s
    #  assert (f1.index == f2.index).all()

    #  diff = f1 - f2
    #  info = predict(diff, contest)  # 0.00252s
    #  print(info)
