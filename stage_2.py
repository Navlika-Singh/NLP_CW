import pandas as pd 
import numpy as np 
import re
from collections import Counter

import matplotlib.pyplot as plt 
import seaborn as sns

import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer 
from sklearn.manifold import TSNE 
import umap

nltk.download("punkt_tab")
nltk.download("stopwords")

sns.set(style="whitegrid")

from utils.load_data import DontPatronizeMe
from utils.helper import token_count, clean_and_tokenize, get_ngram_freq, compute_log_odds_ngrams, preprocess_text

#load data
train_data = DontPatronizeMe(
    data_path="/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/data",
    split="train",
    split_file="/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/data/splits/train_semeval_parids-labels.csv"
)
df = train_data.load_task1()
print(df.head()) #step 1
exit()

# class distribution
class_dist = (
    df["label"]
    .value_counts()
    .rename_axis("label")
    .reset_index(name="count")
)
class_dist["percentage"] = class_dist["count"] / class_dist["count"].sum()

print(class_dist)

plt.figure(figsize=(5, 4))
sns.barplot(x="label", y="percentage", data=class_dist)
plt.xticks([0, 1], ["Non-PCL", "PCL"])
plt.title("Class Distribution (Train Set)")
plt.ylabel("Percentage of samples")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/class_distribution_percentage.png")

#token length stats
df["token_count"] = df["text"].apply(token_count)

length_stats = (
    df.groupby("label")["token_count"]
    .describe(percentiles=[0.95])
    .round(2)
)

print(length_stats)

#vocab
df["tokens"] = df["text"].astype(str).apply(lambda x: x.split())
all_tokens = [token.lower() for tokens in df["tokens"] for token in tokens]
vocab = set(all_tokens)
vocab_size = len(vocab)

print("\nVocabulary Statistics:")
print(f"Vocabulary size: {vocab_size}")

plt.figure(figsize=(5, 4))
sns.boxplot(x="label", y="token_count", data=df)
plt.xticks([0, 1], ["Non-PCL", "PCL"])
plt.title("Token Length Distribution by Class")
plt.ylabel("Token Count")
plt.xlabel("Class")
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/tokencount_distribution.png")

p95 = df["token_count"].quantile(0.95)  #outliers
outliers = df[df["token_count"] > p95][
    ["par_id", "label", "token_count"]
].sort_values("token_count", ascending=False)
print(outliers.head(10))

#lexical analysis
df["tokens"] = df["text"].apply(clean_and_tokenize)
freq_pcl = get_ngram_freq(df[df["label"] == 1])
freq_non_pcl = get_ngram_freq(df[df["label"] == 0])

top_pcl = pd.DataFrame(freq_pcl.most_common(25), columns=["tokens", "frequency"])
top_non_pcl = pd.DataFrame(freq_non_pcl.most_common(25), columns=["tokens", "frequency"])
print(top_pcl)
print(top_non_pcl)

log_odds = compute_log_odds_ngrams(freq_pcl, freq_non_pcl)
log_odds_df = (
    pd.DataFrame(log_odds.items(), columns=["token", "log_odds"])
    .sort_values("log_odds", ascending=False)
)

print(log_odds_df.head(25))

plt.figure(figsize=(5, 4))
sns.barplot(
    x="log_odds",
    y="token",
    data=log_odds_df.head(25)
)
plt.title("Top Tokens Indicative of PCL (Log-Odds)")
plt.xlabel("Log-Odds (PCL vs Non-PCL)")
plt.ylabel("Token")
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/pcl_indicativeTokens_logodds_wtnoise.png")

#bigram
bigrams_pcl = get_ngram_freq(df[df["label"] == 1], n=2, min_freq=0)
bigrams_non_pcl = get_ngram_freq(df[df["label"] == 0], n=2, min_freq=0)

top_bigrams_pcl = pd.DataFrame(
    [(" ".join(k), v) for k, v in bigrams_pcl.most_common(25)],
    columns=["bigram", "frequency"]
)
top_bigrams_non_pcl = pd.DataFrame(
    [(" ".join(k), v) for k, v in bigrams_non_pcl.most_common(25)],
    columns=["bigram", "frequency"]
)
print(top_bigrams_pcl)
print(top_bigrams_non_pcl)

bigram_log_odds = compute_log_odds_ngrams(bigrams_pcl, bigrams_non_pcl, min_freq=10)
bigram_log_odds_df = (
    pd.DataFrame(bigram_log_odds.items(), columns=["bigram", "log_odds"])
    .sort_values("log_odds", ascending=False)
)
print(bigram_log_odds_df.head(25))

plt.figure(figsize=(5, 4))
sns.barplot(
    x="log_odds",
    y="bigram",
    data=bigram_log_odds_df.head(25)
)
plt.title("Top PCL-Indicative Bigrams (Log-Odds)")
plt.xlabel("Log-Odds (PCL vs Non-PCL)")
plt.ylabel("Bigram")
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/pcl_indicativeTokens_bigram_logodds_wtnoise.png")

#trigrams
trigrams_pcl = get_ngram_freq(df[df["label"] == 1], n=3, min_freq=10)
trigrams_non_pcl = get_ngram_freq(df[df["label"] == 0], n=3, min_freq=10)

trigram_log_odds = compute_log_odds_ngrams(trigrams_pcl, trigrams_non_pcl)

trigram_log_odds_df = (
    pd.DataFrame(trigram_log_odds.items(), columns=["trigram", "log_odds"])
    .sort_values("log_odds", ascending=False)
)

print(trigram_log_odds_df.head(25))
plt.figure(figsize=(5, 4))
sns.barplot(
    x="log_odds",
    y="trigram",
    data=trigram_log_odds_df.head(25)
)
plt.title("Top PCL-Indicative Trigram (Log-Odds)")
plt.xlabel("Log-Odds (PCL vs Non-PCL)")
plt.ylabel("Trigram")
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/pcl_indicativeTokens_trigram_logodds_wtnoise.png")

#embedding visualization
MAX_SAMPLES=1000

df_vis = df.sample(
    n=min(len(df), MAX_SAMPLES),
    random_state=42
).reset_index(drop=True)
df_vis["text_preprocessed"] = df_vis["text"].apply(preprocess_text)

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

embeddings = model.encode(
    df_vis["text_preprocessed"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)

umap_reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric="cosine",
    random_state=42
)
embeddings_2d = umap_reducer.fit_transform(embeddings)

plt.figure(figsize=(5, 4))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=df_vis["label"],
    palette={0:"steelblue", 1:"darkorange"},
    alpha=0.7,
    s=40
)
plt.title("UMAP Projection of Paragraph Embeddings (Train Set)")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend(title="Class", labels=["Non-PCL", "PCL"])
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/umap.png")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    random_state=42
)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(5, 4))
sns.scatterplot(
    x=embeddings_2d[:, 0],
    y=embeddings_2d[:, 1],
    hue=df_vis["label"],
    palette={0:"steelblue", 1:"darkorange"},
    alpha=0.7,
    s=40
)
plt.title("TSNE Projection of Paragraph Embeddings (Train Set)")
plt.xlabel("TSNE Dimension 1")
plt.ylabel("TSNE Dimension 2")
plt.legend(title="Class", labels=["Non-PCL", "PCL"])
plt.tight_layout()
plt.savefig("/vol/bitbucket/ns1324/coursework/nlp/dontpatronizeme/NLP_Coursework/results/stage_2/tsne.png")