import os
import re
import ast
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score,
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score, roc_curve, auc
)

from utils.helper import preprocess_text
from utils.load_data import DontPatronizeMe

sns.set(style="whitegrid")

EMBEDDING_MODELS = {
    "mxbai":  "mixedbread-ai/mxbai-embed-large-v1",
    "bge":    "BAAI/bge-large-en-v1.5",
    "e5":     "intfloat/e5-large-v2",
    "mpnet":  "sentence-transformers/all-mpnet-base-v2",
}

EMBEDDING_DIMS = {
    "mxbai": 1024,
    "bge":   1024,
    "e5":    1024,
    "mpnet": 768,
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#residual block
class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout = 0.3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        residual = self.proj(x)
        out = self.dropout(F.gelu(self.bn(self.fc(x))))
        return out + residual

#stong mlp
class StrongMLP(nn.Module):

    def __init__(self, input_dim, dropout= 0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.blocks = nn.Sequential(
            ResidualBlock(input_dim, 1024, dropout),
            ResidualBlock(1024,512, dropout),
            ResidualBlock(512, 256, dropout),
            ResidualBlock(256, 128, dropout),
        )
        self.classifier = nn.Linear(128, 2)

    def forward(self, x):
        x = self.input_bn(x)
        x = self.blocks(x)
        return self.classifier(x)


def load_embeddings(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Embedding cache not found: {cache_path}\n"f"Run train.py first to generate embeddings.")
    print(f"Loading embeddings from {cache_path}")
    return np.load(cache_path)

def find_cache(split_name, model_key, size):
    return f"embedding_cache/{split_name}_{model_key}_n{size}.npy"


def load_model_and_predict(checkpoint_path, X,device):

    if checkpoint_path.endswith(".joblib"):
        bundle = joblib.load(checkpoint_path)
        clf = bundle["clf"]
        thresh = bundle["threshold"]
        probs = clf.predict_proba(X)[:, 1]
        return probs, thresh

    elif checkpoint_path.endswith(".pt"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        input_dim = ckpt["input_dim"]
        mlp = StrongMLP(input_dim=input_dim).to(device)
        mlp.load_state_dict(ckpt["mlp_state"])
        mlp.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        with torch.no_grad():
            probs = F.softmax(mlp(X_t), dim=-1)[:, 1].cpu().numpy()
        return probs, ckpt["threshold"]

    else:
        raise ValueError(
            f"Unrecognised checkpoint format: {checkpoint_path}\n"
            "Expected .joblib (classical) or .pt (MLP)."
        )

def model_label(checkpoint_path):
    name = os.path.basename(checkpoint_path).replace(".joblib", "").replace(".pt", "")
    prefix = name.split("_")[0].upper()
    return prefix

#error analysis
def error_analysis(df, y_true, y_pred_mlp, out_dir):
    print("\n" + "="*55)
    print("Error Analysis")
    print("="*55)

    os.makedirs(out_dir, exist_ok=True)
    df = df.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_mlp"] = y_pred_mlp
    mlp_correct = df["y_mlp"] == df["y_true"]

    df["quadrant"] = "Wrong"
    df.loc[mlp_correct, "quadrant"] = "Correct"

    counts = df["quadrant"].value_counts().reindex(
        ["Correct", "Wrong"], fill_value=0)
    labels = {
        "Correct": "Correct",
        "Wrong": "Wrong",
    }

    print("\nPrediction breakdown:")
    for q, n in counts.items():
        print(f"{labels[q]:<12} {n}")

    #bar chart
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        [labels[q] for q in ["Correct", "Wrong"]],
        [counts[q] for q in ["Correct", "Wrong"]],
        color=["steelblue", "salmon"],
    )
    ax.set_ylabel("Number of samples")
    ax.set_title("Correct vs Incorrect Predictions")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_quadrants.png"), dpi=150)
    plt.close(fig)

    wrong_pcl = df[(df["quadrant"] == "Wrong") & (df["y_true"] == 1)]
    sample = wrong_pcl.sample(min(20, len(wrong_pcl)), random_state=42)
    sample[["text", "keyword", "y_true", "y_mlp"]].to_csv(os.path.join(out_dir, "wrong_pcl_samples.csv"), index=False)
    print(f"Saved wrong_pcl_samples.csv({len(sample)} rows)")

    fn = df[(df["y_true"] == 1) & (df["y_mlp"] == 0)]
    fp = df[(df["y_true"] == 0) & (df["y_mlp"] == 1)]
    print(f"\nFalse Negatives (missed PCL): {len(fn)}")
    print(f"False Positives (false alarm): {len(fp)}")

    fn.sample(min(10, len(fn)), random_state=42)[["text", "keyword"]].to_csv(os.path.join(out_dir, "mlp_false_negatives.csv"), index=False)
    fp.sample(min(10, len(fp)), random_state=42)[["text", "keyword"]].to_csv(os.path.join(out_dir, "mlp_false_positives.csv"), index=False)

    kw_errors = (
        df[df["y_mlp"] != df["y_true"]]
        .groupby("keyword").size()
        .reset_index(name="errors")
    )
    kw_total = df.groupby("keyword").size().reset_index(name="total")
    kw_stats = kw_errors.merge(kw_total, on="keyword")
    kw_stats["error_rate"] = kw_stats["errors"] / kw_stats["total"]
    kw_stats = kw_stats.sort_values("error_rate", ascending=False)

    print(kw_stats.to_string(index=False))
    kw_stats.to_csv(os.path.join(out_dir, "keyword_error_rates.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=kw_stats, x="keyword", y="error_rate",ax=ax, color="steelblue")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Error Rate by Keyword (Local Test Set)")
    ax.set_ylabel("Error Rate")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "keyword_error_rates.png"), dpi=150)
    plt.close(fig)
    print(f"Saved keyword_error_rates.png")

def ablation_studies(X_train, X_local_test, y_train, y_local_test,y_val, X_val,class_weights, device, args, out_dir):

    os.makedirs(out_dir, exist_ok=True)
    results = []
    input_dim = EMBEDDING_DIMS[args.embedding_model]

    def run_clf(X_tr, X_vl, X_test, y_tr, y_vl, y_test, cw,use_threshold=True):
        cw_dict = {0: cw[0], 1: cw[1]}
        clf = __import__("sklearn.svm", fromlist=["SVC"]).SVC(
            C=1.0, kernel="rbf", class_weight=cw_dict,
            probability=True, random_state=42,
        )
        clf.fit(X_tr, y_tr)

        val_probs = clf.predict_proba(X_vl)[:, 1]
        test_probs = clf.predict_proba(X_test)[:, 1]

        if use_threshold:
            best_t, best_f1_val = 0.5, 0.0
            for t in np.arange(0.1, 0.9, 0.01):
                f1v = f1_score(y_vl, (val_probs >= t).astype(int),pos_label=1, zero_division=0)
                if f1v > best_f1_val:
                    best_f1_val, best_t = f1v, t
        else:
            best_t = 0.5

        test_f1 = f1_score(y_test, (test_probs >= best_t).astype(int),pos_label=1, zero_division=0)
        return test_f1

    uniform_weights = np.array([1.0, 1.0])

    print("\n[1/4] No preprocessing...")
    raw_train_cache = find_cache("train", f"{args.embedding_model}",len(y_train))
    raw_test_cache = find_cache("local_test", f"{args.embedding_model}",len(y_local_test))
    raw_val_cache = find_cache("internal_val", f"{args.embedding_model}",len(y_val))
    if (os.path.exists(raw_train_cache) and os.path.exists(raw_test_cache)
            and os.path.exists(raw_val_cache)):
        X_train_raw = load_embeddings(raw_train_cache)
        X_test_raw = load_embeddings(raw_test_cache)
        X_val_raw = load_embeddings(raw_val_cache)
        f1_raw = run_clf(X_train_raw, X_val_raw, X_test_raw,y_train, y_val, y_local_test, class_weights)
        print(f"F1 (PCL): {f1_raw:.4f}")
    else:
        print("Raw embedding cache not found — skipping ablation 1.")
        f1_raw = None
    results.append({"ablation": "No Preprocessing", "test_f1_pcl": f1_raw})

    print("\n[2/4] No class reweighting...")
    f1_nocw = run_clf(X_train, X_val, X_local_test,y_train, y_val, y_local_test, uniform_weights)
    results.append({"ablation": "No Class Reweighting", "test_f1_pcl": f1_nocw})
    print(f"F1 (PCL): {f1_nocw:.4f}")

    print("\n[3/4] No threshold tuning (t=0.5)...")
    f1_nott = run_clf(X_train, X_val, X_local_test,y_train, y_val, y_local_test,class_weights, use_threshold=False)
    results.append({"ablation": "No Threshold Tuning", "test_f1_pcl": f1_nott})
    print(f"F1 (PCL): {f1_nott:.4f}")

    print("\n[4/4] Full system (from checkpoint)...")
    full_probs, full_thresh = load_model_and_predict(args.checkpoint, X_local_test, device)
    f1_full = f1_score(y_local_test, (full_probs >= full_thresh).astype(int),pos_label=1, zero_division=0)
    results.append({"ablation": "Full System", "test_f1_pcl": f1_full})
    print(f"F1 (PCL): {f1_full:.4f}")

    abl_df = pd.DataFrame(results).dropna()
    print("\nAblation Summary (local test set):")
    print(abl_df.to_string(index=False))
    abl_df.to_csv(os.path.join(out_dir, "ablation_results.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["salmon" if r["ablation"] != "Full System" else "steelblue" for _, r in abl_df.iterrows()]
    ax.barh(abl_df["ablation"], abl_df["test_f1_pcl"], color=colors)
    ax.axvline(f1_full, color="steelblue", linestyle="--", linewidth=1)
    ax.set_xlabel("Val F1 (PCL class)")
    ax.set_title("Ablation Study (Local Test Set)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(abl_df["test_f1_pcl"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation_results.png"), dpi=150)
    plt.close(fig)
    print(f"Saved ablation_results.png")


def custom_metrics(y_true, model_probs, model_thresh, out_dir):
    print("Custom Metrics")

    os.makedirs(out_dir, exist_ok=True)

    class_prior = y_true.mean()
    rng = np.random.default_rng(42)
    rand_probs = rng.uniform(0, 1, size=len(y_true))
    rand_probs = rand_probs * class_prior / 0.5
    rand_probs = np.clip(rand_probs, 0, 1)
    rand_thresh = class_prior  # natural threshold = prior
    rand_preds = (rand_probs >= rand_thresh).astype(int)

    model_preds = (model_probs >= model_thresh).astype(int)

    print(f"\nClass prior (PCL): {class_prior:.4f}")
    print(f"Random baseline threshold: {rand_thresh:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, preds, title in zip(axes,[model_preds,rand_preds],["Model","Random Baseline"],):
        cm = confusion_matrix(y_true, preds)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,xticklabels=["Non-PCL", "PCL"],yticklabels=["Non-PCL", "PCL"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix — {title}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "confusion_matrices.png"), dpi=150)
    plt.close(fig)
    print("Saved confusion_matrices.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    for probs, label, color in [(model_probs, "Model", "steelblue"),(rand_probs, "Random Baseline", "gray"),]:
        fpr, tpr, _ = roc_curve(y_true, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, label=f"{label} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close(fig)
    print("Saved roc_curves.png")

    thresholds = np.arange(0.1, 0.9, 0.01)
    fig, ax = plt.subplots(figsize=(6, 4))
    f1s = [f1_score(y_true, (model_probs >= t).astype(int),pos_label=1, zero_division=0) for t in thresholds]
    ax.plot(thresholds, f1s, color="steelblue", label="Method")
    ax.axvline(model_thresh, color="steelblue", linestyle="--", linewidth=0.8,label=f"Best t={model_thresh:.2f}")
    rand_f1 = f1_score(y_true, rand_preds, pos_label=1, zero_division=0)
    ax.axhline(rand_f1, color="gray", linestyle=":", linewidth=0.8,label=f"Random baseline F1={rand_f1:.3f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 (PCL class)")
    ax.set_title("F1 vs Decision Threshold")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "f1_vs_threshold.png"), dpi=150)
    plt.close(fig)
    print("Saved f1_vs_threshold.png")

    rows = []
    for t in [0.3, 0.4, model_thresh, 0.6, 0.7]:
        p = (model_probs >= t).astype(int)
        tp = int(((p == 1) & (y_true == 1)).sum())
        fp = int(((p == 1) & (y_true == 0)).sum())
        fn = int(((p == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec= tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = f1_score(y_true, p, pos_label=1, zero_division=0)
        rows.append({"threshold": round(t, 2),"precision": round(prec, 3), "recall": round(rec, 3),"f1_pcl": round(f1, 3), "TP": tp, "FP": fp, "FN": fn,})

    trade_df = pd.DataFrame(rows)
    print("\nPrecision / Recall trade-off:")
    print(trade_df.to_string(index=False))
    trade_df.to_csv(os.path.join(out_dir, "threshold_tradeoff.csv"), index=False)

    print(f"\nModel classification report (t={model_thresh:.2f}):")
    print(classification_report(y_true, model_preds,target_names=["Non-PCL", "PCL"]))
    print(f"Random Baseline classification report (t={rand_thresh:.2f}):")
    print(classification_report(y_true, rand_preds,target_names=["Non-PCL", "PCL"]))


PCL_CATEGORIES = [
    "Unbalanced_power_relations",
    "Shallow_solution",
    "Presupposition",
    "Authority_voice",
    "Compassion",
    "The_poorer_the_merrier",
    "Metaphors",
]

PCL_CATEGORY_SHORT = [
    "Unb. power rel.",
    "Shallow solu.",
    "Presupposition",
    "Authority voice",
    "Compassion",
    "The p., the mer.",
    "Metaphor",
]

def load_categories(categories_file, par_ids):
    
    cat_df = pd.read_csv(
        categories_file, sep="	",
        names=["par_id", "art_id", "text", "keyword", "country", "span_start",
               "span_end", "span_text", "pcl_category", "number_of_annotators"],
        header=None,
    )
    cat_df["par_id"] = cat_df["par_id"].astype(str)

    matched = cat_df["par_id"].isin([str(p) for p in par_ids])
    print(f"Categories file: {len(cat_df)} rows, "
          f"{cat_df['par_id'].nunique()} unique par_ids")
    print(f"Matched to local test set: {matched.sum()} rows "
          f"({cat_df[matched]['par_id'].nunique()} unique par_ids)")
    print(f"Category distribution in matched rows:")
    print(cat_df[matched]["pcl_category"].value_counts().to_string())

    index_df = pd.DataFrame({"par_id": [str(p) for p in par_ids]})
    for cat in PCL_CATEGORIES:
        has_cat = (
            cat_df[cat_df["pcl_category"] == cat]["par_id"]
            .unique()
        )
        index_df[cat] = index_df["par_id"].isin(has_cat).astype(int)

    print("\nSupport per category (paragraphs in local test set with that category):")
    for cat in PCL_CATEGORIES:
        n = index_df[cat].sum()
        print(f"{cat:<35} {n}")

    return index_df.set_index("par_id")


def category_and_keyword_analysis(local_test_df: pd.DataFrame,
                                   y_true: np.ndarray,
                                   y_pred: np.ndarray,
                                   categories_file: str,
                                   out_dir: str):
    
    print("  Category and Keyword Error Analysis")

    os.makedirs(out_dir, exist_ok=True)

    df = local_test_df.copy().reset_index(drop=True)
    df["y_true"] = y_true
    df["y_pred"] = y_pred

    print("\nLoading category annotations...")
    par_ids = df["par_id"].tolist()
    cat_matrix = load_categories(categories_file, par_ids)
    cat_matrix = cat_matrix.reindex([str(p) for p in par_ids])

    for cat in PCL_CATEGORIES:
        df[cat] = cat_matrix[cat].values

    fn_df = df[(df["y_true"] == 1) & (df["y_pred"] == 0)]
    fp_df = df[(df["y_true"] == 0) & (df["y_pred"] == 1)]

    print(f"\nFalse negatives (missed PCL): {len(fn_df)}")
    print(f"False positives (non-PCL predicted as PCL): {len(fp_df)}")

    cat_rows = []
    for cat, short in zip(PCL_CATEGORIES, PCL_CATEGORY_SHORT):
        total_with_cat = int(df[df["y_true"] == 1][cat].sum())  # all PCL with this cat
        fn_with_cat = int(fn_df[cat].sum())                   # missed ones with this cat
        miss_rate = fn_with_cat / total_with_cat if total_with_cat > 0 else 0.0
        cat_rows.append({
            "category":short,
            "total_pcl":total_with_cat,
            "false_negatives": fn_with_cat,
            "miss_rate_%":round(miss_rate * 100, 1),
        })

    cat_df = pd.DataFrame(cat_rows).sort_values("miss_rate_%", ascending=False)
    print("\nCategory error analysis (false negative rate per category):")
    print(cat_df.to_string(index=False))
    cat_df.to_csv(os.path.join(out_dir, "category_error_analysis.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["salmon" if r > cat_df["miss_rate_%"].mean() else "steelblue"
              for r in cat_df["miss_rate_%"]]
    bars = ax.barh(cat_df["category"], cat_df["miss_rate_%"], color=colors)
    ax.axvline(cat_df["miss_rate_%"].mean(), color="gray", linestyle="--",
               linewidth=0.8, label=f"Mean={cat_df['miss_rate_%'].mean():.1f}%")
    for bar, val in zip(bars, cat_df["miss_rate_%"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("False Negative Rate (%)")
    ax.set_title("PCL Miss Rate by Category (% of category instances missed)")
    ax.set_xlim(0, 105)
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "category_miss_rate.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    caught = cat_df["total_pcl"] - cat_df["false_negatives"]
    missed = cat_df["false_negatives"]
    ax.barh(cat_df["category"], caught, color="steelblue", label="Detected")
    ax.barh(cat_df["category"], missed, left=caught,       color="salmon",    label="Missed")
    ax.set_xlabel("Number of PCL instances")
    ax.set_title("Detected vs Missed PCL by Category")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "category_detected_vs_missed.png"), dpi=150)
    plt.close(fig)
    print("Saved category_miss_rate.png, category_detected_vs_missed.png")

    for cat, short in zip(PCL_CATEGORIES, PCL_CATEGORY_SHORT):
        subset = fn_df[fn_df[cat] == 1]
        if len(subset) > 0:
            subset.sample(min(5, len(subset)), random_state=42)[
                ["text", "keyword", cat]
            ].to_csv(
                os.path.join(out_dir,
                             f"fn_samples_{short.replace(' ', '_').replace('.','')}.csv"),
                index=False,
            )

    kw_rows = []
    for kw, kw_df in df.groupby("keyword"):
        yt = kw_df["y_true"].values
        yp = kw_df["y_pred"].values
        if yt.sum() == 0:
            continue
        fn_kw = int(((yp == 0) & (yt == 1)).sum())
        fp_kw = int(((yp == 1) & (yt == 0)).sum())
        errors     = fn_kw + fp_kw
        error_rate = errors / len(yt)
        miss_rate  = fn_kw / yt.sum() if yt.sum() > 0 else 0
        kw_rows.append({
            "keyword":    kw,
            "total":      len(yt),
            "pcl_count":  int(yt.sum()),
            "fn":         fn_kw,
            "fp":         fp_kw,
            "errors":     errors,
            "error_rate_%": round(error_rate * 100, 1),
            "miss_rate_%":  round(miss_rate  * 100, 1),
        })

    kw_df_out = pd.DataFrame(kw_rows).sort_values("error_rate_%", ascending=False)
    print("\nKeyword error analysis:")
    print(kw_df_out[["keyword", "total", "pcl_count", "fn", "fp",
                      "error_rate_%", "miss_rate_%"]].to_string(index=False))
    kw_df_out.to_csv(os.path.join(out_dir, "keyword_error_analysis.csv"), index=False)

    x = np.arange(len(kw_df_out))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, kw_df_out["fn"], width, label="False Negatives (missed PCL)",
           color="salmon")
    ax.bar(x + width/2, kw_df_out["fp"], width, label="False Positives",
           color="steelblue", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(kw_df_out["keyword"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("False Negatives and False Positives by Keyword")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "keyword_fn_fp.png"), dpi=150)
    plt.close(fig)
    print("Saved keyword_fn_fp.png")


def main(args):
    set_seed(42)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading local test set (val split)...")
    local_test_df = DontPatronizeMe(
        args.data_path, split="val", split_file=args.val_split
    ).load_task1()
    y_local_test  = local_test_df["label"].values
    print(f"Local test: {len(local_test_df)}  "
          f"(PCL={y_local_test.sum()}, "
          f"Non-PCL={(y_local_test==0).sum()})")

    print("Loading train data (for ablations)...")
    labelled_df = DontPatronizeMe(
        args.data_path, split="train", split_file=args.train_split
    ).load_task1()

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(labelled_df, labelled_df["label"]))
    train_df = labelled_df.iloc[train_idx].reset_index(drop=True)
    val_df = labelled_df.iloc[val_idx].reset_index(drop=True)
    y_train= train_df["label"].values
    y_val = val_df["label"].values

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)

    print("\nLoading cached embeddings...")
    m = args.embedding_model

    X_train = load_embeddings(find_cache("train",        m, len(y_train)))
    X_val= load_embeddings(find_cache("internal_val", m, len(y_val)))
    X_local_test = load_embeddings(find_cache("local_test",   m, len(y_local_test)))

    print("\nLoading saved predictions...")

    def load_preds(path):
        with open(path) as f:
            return np.array([int(l.strip()) for l in f.readlines()])

    y_pred_model = load_preds(args.preds)

    assert len(y_pred_model) == len(y_local_test), \
        f"Predictions length {len(y_pred_model)} != local test size {len(y_local_test)}"

    print("Loading checkpoint for probabilities...")
    model_probs, model_thresh = load_model_and_predict(args.checkpoint, X_local_test, args.device)
    
    error_analysis(local_test_df, y_local_test, y_pred_model,out_dir=os.path.join(args.out_dir, "error_analysis"))

    ablation_studies(
        X_train, X_local_test, y_train, y_local_test,
        y_val, X_val,
        class_weights, args.device, args,
        out_dir=os.path.join(args.out_dir, "ablation"),
    )

    custom_metrics(
        y_local_test, model_probs, model_thresh,
        out_dir=os.path.join(args.out_dir, "metrics"),
    )

    if args.categories_file:
        category_and_keyword_analysis(
            local_test_df, y_local_test, y_pred_model,
            categories_file=args.categories_file,
            out_dir=os.path.join(args.out_dir, "category_analysis"),
        )
    else:
        print("\n  Skipping category analysis (--categories_file not provided)")

    print(f"\nAll outputs saved to: {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCL Local Evaluation Suite")

    parser.add_argument("--data_path",required=True)
    parser.add_argument("--train_split", required=True)
    parser.add_argument("--val_split",required=True)

    parser.add_argument("--categories_file", default=None,help="Path to dontpatronizeme_categories.tsv (optional)")

    parser.add_argument("--preds",required=True, help="Path to local_test predictions txt")
    parser.add_argument("--checkpoint",  required=True,help="Path to .joblib (classical) or .pt (MLP) checkpoint")

    parser.add_argument("--embedding_model", default="mxbai",choices=list(EMBEDDING_MODELS.keys()))

    parser.add_argument("--out_dir",default="results/local_eval")
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(f"\nDevice: {args.device}")
    main(args)