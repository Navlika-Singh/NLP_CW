import os
import re
import ast
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_recall_curve
)

from utils.helper import preprocess_text
from utils.load_data import DontPatronizeMe

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from peft import LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

EMBEDDING_MODELS = {
    "mxbai":"mixedbread-ai/mxbai-embed-large-v1",      
    "bge": "BAAI/bge-large-en-v1.5",                  
    "e5": "intfloat/e5-large-v2",                    
    "mpnet":"sentence-transformers/all-mpnet-base-v2",
}

EMBEDDING_DIMS = {
    "mxbai":1024,
    "bge":1024,
    "e5":1024,
    "mpnet": 768,
}

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

#embedding extraction
def get_embeddings(texts, model, batch_size = 64, device= "cpu"):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i: i + batch_size]
        with torch.no_grad():
            embs = model.encode(
                batch, batch_size=batch_size, device=device,
                normalize_embeddings=True, show_progress_bar=False,
            )
        all_embeddings.append(embs)
    return np.vstack(all_embeddings)

def load_or_compute_embeddings(texts, cache_path, model,device):
    if os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)
    print(f"Computing embeddings {cache_path}")
    embs = get_embeddings(texts, model, device=device)
    np.save(cache_path, embs)
    return embs

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

#lora for encoder
def apply_lora(sentence_transformer):

    if not HAS_PEFT:
        raise ImportError("peft is not installed. Run: pip install peft")

    hf_model = sentence_transformer._first_module().auto_model

    #auto-detect attention layer names since different models use different naming
    all_module_names = [name for name, _ in hf_model.named_modules()]

    candidate_pairs = [
        ("q_proj", "v_proj"),   
        ("query","value"),    
        ("query_key_value","query_key_value"),  
        ("c_attn","c_attn"),   
    ]

    target_modules = None
    for q_name, v_name in candidate_pairs:
        if any(q_name in n for n in all_module_names):
            target_modules = list({q_name, v_name})
            break

    if target_modules is None:
        print("Could not auto-detect attention modules. All module names:")
        for n in all_module_names:
            print(f"{n}")
        raise ValueError(
            "Could not find attention modules for LoRA. " "Set target_modules manually in apply_lora()."
        )

    print(f"LoRA target modules: {target_modules}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
    )

    hf_model = get_peft_model(hf_model, lora_config)
    hf_model.print_trainable_parameters()

    sentence_transformer._first_module().auto_model = hf_model
    return sentence_transformer

#training loop for mlp
def train_mlp(sentence_transformer,train_texts,val_texts,y_train,y_val,class_weights,device,writer,args,X_train_frozen,X_val_frozen):
    lora_mode = args.lora
    input_dim = EMBEDDING_DIMS[args.embedding_model]

    mlp = StrongMLP(input_dim=input_dim, dropout=args.mlp_dropout).to(device)

    if lora_mode:
        optimizer = torch.optim.AdamW([
            {"params": sentence_transformer.parameters(), "lr": args.lora_lr},
            {"params": mlp.parameters(),"lr": args.mlp_lr},
        ], weight_decay=1e-2)
    else:
        optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.mlp_lr, weight_decay=1e-2)

    steps_per_epoch = int(np.ceil(len(y_train) / args.mlp_batch_size))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[args.lora_lr, args.mlp_lr] if lora_mode else args.mlp_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=args.mlp_epochs,
    )

    cw_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    if not lora_mode:
        X_tr = torch.tensor(X_train_frozen, dtype=torch.float32)
        X_vl = torch.tensor(X_val_frozen,dtype=torch.float32)
        y_tr = torch.tensor(y_train,dtype=torch.long)
        y_vl = torch.tensor(y_val,dtype=torch.long)

        train_loader = DataLoader(TensorDataset(X_tr, y_tr),batch_size=args.mlp_batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_vl, y_vl),batch_size=args.mlp_batch_size, shuffle=False)
    else:
        train_loader = DataLoader(list(zip(range(len(train_texts)), y_train.tolist())),batch_size=args.mlp_batch_size, shuffle=True)
        val_loader = DataLoader(list(zip(range(len(val_texts)), y_val.tolist())),batch_size=args.mlp_batch_size, shuffle=False)

    #train loop
    best_val_f1, best_state = 0.0, None

    for epoch in range(1, args.mlp_epochs + 1):
        mlp.train()
        if lora_mode:
            sentence_transformer.train()

        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()

            if not lora_mode:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
            else:
                idxs, yb = batch
                yb = torch.tensor(yb, dtype=torch.long).to(device)
                xb = torch.tensor(
                    sentence_transformer.encode(
                        [train_texts[i] for i in idxs],
                        normalize_embeddings=True,
                        show_progress_bar=False,
                        device=device,
                    ), dtype=torch.float32
                ).to(device)

            loss = criterion(mlp(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(mlp.parameters()) +
                (list(sentence_transformer.parameters()) if lora_mode else []),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        #val
        mlp.eval()
        if lora_mode:
            sentence_transformer.eval()

        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                if not lora_mode:
                    xb, yb = batch
                    xb = xb.to(device)
                else:
                    idxs, yb = batch
                    xb = torch.tensor(
                        sentence_transformer.encode(
                            [val_texts[i] for i in idxs],
                            normalize_embeddings=True,
                            show_progress_bar=False,
                            device=device,
                        ), dtype=torch.float32
                    ).to(device)

                probs = F.softmax(mlp(xb), dim=-1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(
                    yb.tolist() if not lora_mode else yb)

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Per-epoch threshold tuning
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (all_probs >= t).astype(int)
            f1 = f1_score(all_labels, preds, pos_label=1, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch:>3}/{args.mlp_epochs}"f"loss={avg_loss:.4f}  val_F1={best_f1:.4f}  t={best_t:.2f}")

        writer.add_scalar("MLP/train_loss",avg_loss,epoch)
        writer.add_scalar("MLP/val_F1_PCL",best_f1,epoch)
        writer.add_scalar("MLP/best_thresh",best_t,epoch)

        if best_f1 > best_val_f1:
            best_val_f1 = best_f1
            best_state = {
                "mlp": mlp.state_dict(),
                "threshold": best_t,
            }
            if lora_mode:
                best_state["encoder"] = {
                    k: v.cpu().clone()
                    for k, v in sentence_transformer.state_dict().items()
                }

    mlp.load_state_dict(best_state["mlp"])
    best_thresh = best_state["threshold"]
    print(f"\nBest Val F1 (PCL): {best_val_f1:.4f}  threshold: {best_thresh:.2f}")

    return mlp, best_thresh, best_val_f1, all_probs, all_labels

#simple classifiers
def build_classifier(name, class_weights, params=None):
    cw_dict = {0: class_weights[0], 1: class_weights[1]}
    params = params or {}

    if name == "lr":
        return LogisticRegression(
            C=params.get("C", 1.0),
            solver=params.get("solver", "lbfgs"),
            class_weight=cw_dict,
            max_iter=1000,
            random_state=42,
        )
    elif name == "svm":
        return SVC(
            C=params.get("C", 1.0),
            kernel=params.get("kernel", "rbf"),
            class_weight=cw_dict,
            probability=True,
            random_state=42,
        )
    elif name == "xgb":
        if not HAS_XGB:
            raise ImportError("xgboost not installed.")
        scale = class_weights[1] / class_weights[0]
        return XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            learning_rate=params.get("learning_rate", 0.05),
            max_depth=params.get("max_depth", 6),
            scale_pos_weight=scale,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
            use_label_encoder=False,
        )
    else:
        raise ValueError(f"Unknown classifier: {name}")

#eval
def evaluate(y_true, y_pred, y_prob, split_name,writer, step):
    f1_pos = f1_score(y_true, y_pred, pos_label=1)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"\n{'='*55}")
    print(f"{split_name} Evaluation")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred, target_names=["No PCL", "PCL"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    writer.add_scalar(f"{split_name}/F1_PCL",f1_pos,step)
    writer.add_scalar(f"{split_name}/F1_Macro", f1_macro, step)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots()
    ax.plot(recall, precision, marker=".", label=f"F1={f1_pos:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"PR Curve — {split_name}")
    ax.legend()
    writer.add_figure(f"{split_name}/PR_Curve", fig, step)
    plt.close(fig)

    return {"f1_pcl": f1_pos, "f1_macro": f1_macro}

#tune thresh
def tune_threshold(probs, y_val, writer, step):
    best_thresh, best_f1 = 0.5, 0.0
    thresholds, f1s = np.arange(0.1, 0.9, 0.01), []

    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_val, preds, pos_label=1, zero_division=0)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    print(f"Best threshold: {best_thresh:.2f}  Val F1 (PCL): {best_f1:.4f}")
    writer.add_scalar("Val/BestThreshold", best_thresh, step)
    writer.add_scalar("Val/F1_PCL_Tuned", best_f1,step)

    fig, ax = plt.subplots()
    ax.plot(thresholds, f1s)
    ax.axvline(best_thresh, color="r", linestyle="--", label=f"Best={best_thresh:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1 (PCL)")
    ax.set_title("Threshold Tuning")
    ax.legend()
    writer.add_figure("Val/ThresholdTuning", fig, step)
    plt.close(fig)

    return best_thresh, best_f1


def main(args):
    set_seed(42)

    run_name = (f"{args.classifier}"
                f"{'_lora' if args.lora else ''}"
                f"_{args.embedding_model}"
                f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"\nRun: {run_name}")
    print(f"TensorBoard {log_dir}\n")

    if args.lora and args.classifier != "mlp":
        raise ValueError("--lora is only supported with --classifier mlp")
    if args.lora and not HAS_PEFT:
        raise ImportError("peft not installed. Run: pip install peft")

    print("Loading datasets...")
    labelled_df = DontPatronizeMe(args.data_path, "train", args.train_split).load_task1()
    local_test_df  = DontPatronizeMe(args.data_path, "val", args.val_split).load_task1()
    marker_test_df = DontPatronizeMe(args.data_path, "test").load_test()
    print(len(marker_test_df))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(labelled_df, labelled_df["label"]))

    train_df = labelled_df.iloc[train_idx].reset_index(drop=True)
    val_df = labelled_df.iloc[val_idx].reset_index(drop=True)

    print("\nPreprocessing text...")
    train_texts = [preprocess_text(t) for t in train_df["text"].tolist()]
    val_texts = [preprocess_text(t) for t in val_df["text"].tolist()]
    local_test_texts = [preprocess_text(t) for t in local_test_df["text"].tolist()]
    marker_test_texts = [preprocess_text(t) for t in marker_test_df["text"].tolist()]

    y_train = train_df["label"].values
    y_val = val_df["label"].values
    y_local_test = local_test_df["label"].values

    class_weights = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    print(f"\nClass weights Non-PCL: {class_weights[0]:.3f} "f"PCL: {class_weights[1]:.3f}")

    model_name = EMBEDDING_MODELS[args.embedding_model]
    print(f"\nLoading embedding model: {model_name}")
    embed_model = SentenceTransformer(model_name, device=args.device)

    if args.lora:
        print("Applying LoRA to encoder attention layers...")
        embed_model = apply_lora(embed_model)

    os.makedirs("embedding_cache", exist_ok=True)
    cache_tag = args.embedding_model

    if not args.lora:
        embed_model.eval()
        X_train = load_or_compute_embeddings(train_texts,f"embedding_cache/train_{cache_tag}_n{len(train_texts)}.npy",embed_model, args.device)
        X_val = load_or_compute_embeddings(val_texts,f"embedding_cache/internal_val_{cache_tag}_n{len(val_texts)}.npy",embed_model, args.device)
        X_local_test = load_or_compute_embeddings(local_test_texts,f"embedding_cache/local_test_{cache_tag}_n{len(local_test_texts)}.npy",embed_model, args.device)
        X_marker_test = load_or_compute_embeddings(marker_test_texts,f"embedding_cache/marker_test_{cache_tag}_n{len(marker_test_texts)}.npy",embed_model, args.device)
        
        assert len(X_train) == len(y_train), \
            f"X_train size {len(X_train)} != y_train size {len(y_train)}. Delete embedding_cache and rerun."
        assert len(X_val) == len(y_val), \
            f"X_val size {len(X_val)} != y_val size {len(y_val)}. Delete embedding_cache and rerun."
        assert len(X_local_test) == len(y_local_test), \
            f"X_local_test size {len(X_local_test)} != y_local_test size {len(y_local_test)}. Delete embedding_cache and rerun."
    else:
        X_train = X_val = X_local_test = X_marker_test = None

    if args.classifier == "mlp":
        print(f"\nTraining StrongMLP "f"({'LoRA encoder' if args.lora else 'frozen encoder'})...")

        mlp, best_thresh, _, val_probs, val_labels = train_mlp(
            sentence_transformer=embed_model,
            train_texts=train_texts,
            val_texts=val_texts,
            y_train=y_train,
            y_val=y_val,
            class_weights=class_weights,
            device=args.device,
            writer=writer,
            args=args,
            X_train_frozen=X_train,
            X_val_frozen=X_val,
        )

        val_pred = (val_probs >= best_thresh).astype(int)
        val_metrics = evaluate(val_labels, val_pred, val_probs, "Val", writer, step=1)

        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = f"checkpoints/{run_name}.pt"
        torch.save({
            "mlp_state": mlp.state_dict(),
            "threshold": best_thresh,
            "input_dim": EMBEDDING_DIMS[args.embedding_model],
            "embedding": args.embedding_model,
            "lora": args.lora,
        }, ckpt_path)
        print(f"Checkpoint saved {ckpt_path}")

        mlp.eval()
        if args.lora:
            embed_model.eval()
            X_local_test_np = embed_model.encode(local_test_texts, normalize_embeddings=True,show_progress_bar=False, device=args.device)
            X_marker_test_np = embed_model.encode(marker_test_texts, normalize_embeddings=True,show_progress_bar=True, device=args.device)
        else:
            X_local_test_np  = X_local_test
            X_marker_test_np = X_marker_test

        with torch.no_grad():
            local_test_probs = F.softmax(mlp(torch.tensor(X_local_test_np, dtype=torch.float32).to(args.device)),dim=-1)[:, 1].cpu().numpy()
            marker_test_probs = F.softmax(mlp(torch.tensor(X_marker_test_np, dtype=torch.float32).to(args.device)),dim=-1)[:, 1].cpu().numpy()

    else:
        print(f"\nTraining: {args.classifier.upper()}")
        clf = build_classifier(args.classifier, class_weights)

        if args.classifier == "xgb":
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        else:
            clf.fit(X_train, y_train)

        val_probs = clf.predict_proba(X_val)[:, 1]
        val_pred = clf.predict(X_val)
        val_metrics = evaluate(y_val, val_pred, val_probs, "Val", writer, step=1)

        print("\nTuning decision threshold...")
        best_thresh, _ = tune_threshold(val_probs, y_val, writer, step=1)

        val_pred = (val_probs >= best_thresh).astype(int)
        print("\nVal results with tuned threshold:")
        print(classification_report(y_val, val_pred, target_names=["No PCL", "PCL"]))

        local_test_probs= clf.predict_proba(X_local_test)[:, 1]
        marker_test_probs = clf.predict_proba(X_marker_test)[:, 1]

        import joblib
        os.makedirs("checkpoints", exist_ok=True)
        clf_path = f"checkpoints/{run_name}.joblib"
        joblib.dump({"clf": clf, "threshold": best_thresh,"embedding": args.embedding_model}, clf_path)
        print(f"Classifier saved \u2192 {clf_path}")

        val_labels = y_val

    writer.add_hparams(
        {
            "classifier":args.classifier,
            "embedding_model": args.embedding_model,
            "lora":args.lora,
        },
        {
            "hparam/val_f1_pcl": val_metrics["f1_pcl"],
            "hparam/val_f1_macro": val_metrics["f1_macro"],
        },
    )

    os.makedirs("predictions", exist_ok=True)

    local_test_preds = (local_test_probs >= best_thresh).astype(int)

    print(f"\n{'='*55}")
    print(f"Local Test Evaluation (val set)")
    print(f"{'='*55}")
    print(classification_report(y_local_test, local_test_preds,target_names=["No PCL", "PCL"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_local_test, local_test_preds))

    local_test_f1 = f1_score(y_local_test, local_test_preds, pos_label=1)
    writer.add_scalar("LocalTest/F1_PCL", local_test_f1, 1)
    print(f"\nLocal Test F1 (PCL): {local_test_f1:.4f}")

    with open("predictions/local_test.txt", "w") as f:
        f.write("\n".join(map(str, local_test_preds.tolist())))
    print(f"Saved predictions/local_test.txt")

    marker_test_preds = (marker_test_probs >= best_thresh).astype(int)
    with open("predictions/test.txt", "w") as f:
        f.write("\n".join(map(str, marker_test_preds.tolist())))
    print(f"Saved predictions/test.txt  ({len(marker_test_preds)} lines marker submission)")

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PCL Classification")

    parser.add_argument("--data_path", required=True)
    parser.add_argument("--train_split",required=True)
    parser.add_argument("--val_split", required=True)

    parser.add_argument("--classifier", default="svm",choices=["lr", "svm", "xgb", "mlp"])
    parser.add_argument("--embedding_model", default="mxbai",choices=list(EMBEDDING_MODELS.keys()),help="mxbai,bge,e5,mpnet")
    parser.add_argument("--lora",action="store_true",help="Apply LoRA to encoder attention layers (mlp only)")

    parser.add_argument("--mlp_epochs", type=int,default=30)
    parser.add_argument("--mlp_batch_size", type=int, default=32)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_dropout",type=float, default=0.3)

    parser.add_argument("--lora_lr",type=float, default=2e-5,help="Learning rate for LoRA encoder params")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    print(f"\nDevice: {args.device}")
    main(args)