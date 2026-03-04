import re
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from collections import Counter
from sklearn.metrics import f1_score

nltk.download("punkt_tab")
nltk.download("stopwords")

def token_count(text):
    return len(word_tokenize(text))

STOPWORDS = set(stopwords.words("english"))
def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens

def get_ngram_freq(df_subset, n=1, min_freq=0):
    counter = Counter()
    for tokens in df_subset["tokens"]:
        counter.update(ngrams(tokens, n))

    counter = Counter({
        k: v for k, v in counter.items() if v >= min_freq
    })
    return counter

def compute_log_odds_ngrams(freq_pos, freq_neg, alpha=1, min_freq=10):
    # vocab = set(freq_pos.keys()).union(set(freq_neg.keys()))

    vocab = { #to remove noise
        w for w in set(freq_pos) | set(freq_neg)
        if (freq_pos[w] + freq_neg[w]) >= min_freq
    }

    log_odds = {}

    total_pos = sum(freq_pos.values())
    total_neg = sum(freq_neg.values())

    for ngram in vocab:
        pos = freq_pos.get(ngram, 0) + alpha
        neg = freq_neg.get(ngram, 0) + alpha 

        log_odds[" ".join(ngram)] = np.log(
            (pos/total_pos)/
            (neg/total_neg)
        )

    return log_odds

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"&\w+;", "", text)
    text = re.sub(r"[\d]+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lexical_features(text, top_ngrams, log_odds_dict):
    feats = np.zeros(len(top_ngrams), dtype=np.float32)
    for i, ng in enumerate(top_ngrams):
        if ng in text:
            feats[i] = log_odds_dict[ng]
    return feats