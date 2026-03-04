import os
import ast
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer

import torch
from torch.utils.data import Dataset, DataLoader 

from utils.helper import lexical_features

MAX_LEN = 256

class DontPatronizeMe:

    def __init__(self, data_path, split="train", split_file=None):
        self.data_path = data_path
        self.split = split
        self.split_file = split_file

        if split in {"train", "val"} and split_file is None:
            raise ValueError("split_file must be provided for train/val splits")

    def load_task1(self):
        rows = []
        with open(os.path.join(self.data_path, "dontpatronizeme_pcl.tsv")) as f:
            for line in f.readlines()[4:]:
                fields = line.strip().split("\t")
                par_id, art_id, keyword, country, text, orig_label = (
                    fields[0], fields[1], fields[2],
                    fields[3], fields[4], fields[-1],
                )
                label = 0 if orig_label in {"0", "1"} else 1
                rows.append({
                    "par_id": par_id, "art_id": art_id,
                    "keyword": keyword, "country": country,
                    "text": text, "label": label, "orig_label": orig_label,
                })

        full_df = pd.DataFrame(rows)

        if self.split in {"train", "val"}:
            split_df = self._load_split_file(self.split_file)
            full_df  = full_df.merge(
                split_df[["par_id", "label"]], on="par_id",
                how="inner", suffixes=("", "_split"),
            )
            full_df["label"] = full_df["label_split"]
            full_df.drop(columns=["label_split"], inplace=True)

        self.df = full_df
        return self.df

    def _load_split_file(self, path):
        df = pd.read_csv(path) if path.endswith(".csv") else pd.read_csv(path, sep="\t")
        if isinstance(df["label"].iloc[0], str):
            df["label"] = df["label"].apply(lambda x: int(any(ast.literal_eval(x))))
        df["par_id"] = df["par_id"].astype(str)
        return df

    def load_test(self):
        rows = []
        with open(os.path.join(self.data_path, "task4_test.tsv")) as f:
            for line in f.readlines():
                fields = line.strip().split("\t")
                rows.append({
                    "par_id": fields[0], "art_id": fields[1],
                    "keyword": fields[2], "country": fields[3],
                    "text": fields[4],
                })
        self.df = pd.DataFrame(rows)
        return self.df

class PCLDataset(Dataset):
    def __init__(self, df, tokenizer, top_ngrams, log_odds_dict):
        self.texts = df["text"].tolist()
        self.preprocessed_texts = df["preprocessed_text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.top_ngrams = top_ngrams
        self.logs_odds_dict = log_odds_dict

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )

        lex_feats = lexical_features(self.preprocessed_texts[idx], self.top_ngrams, self.logs_odds_dict)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "lexical_feats": torch.tensor(lex_feats),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }
