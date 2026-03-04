# PCL Detection - Don't Patronize Me!

Code for binary PCL (Patronizing and Condescending Language) detection on the
[Don't Patronize Me!](https://github.com/Perez-AlmendrosC/dontpatronizeme) dataset.

---

## Setup

```bash
python -m venv nlp_env
source nlp_env/bin/activate
pip install -r requirements.txt
```

---

## Data

Download the dataset from the [shared task page](https://github.com/Perez-AlmendrosC/dontpatronizeme)
and place it under `data/`:

```
data/
  dontpatronizeme_pcl.tsv
  dontpatronizeme_categories.tsv
  splits/
    train_semeval_parids-labels.csv
    dev_semeval_parids-labels.csv
```

---

## Training

```bash
python train.py \
    --data_path    ./data \
    --train_split  ./data/splits/train_semeval_parids-labels.csv \
    --val_split    ./data/splits/dev_semeval_parids-labels.csv \
    --classifier   svm \
    --embedding_model mxbai
```

Supported classifiers: `svm`, `lr`, `xgb`, `mlp`.  
Supported embedding models: `mxbai`, `bge`, `e5`, `mpnet`.

Embeddings are cached to `embedding_cache/` on first run.  
Checkpoints are saved to `checkpoints/`.  
Predictions are saved to `predictions/`.

---

## Evaluation

```bash
python evaluate.py \
    --data_path       ./data \
    --train_split     ./data/splits/train_semeval_parids-labels.csv \
    --val_split       ./data/splits/dev_semeval_parids-labels.csv \
    --preds           ./predictions/local_test.txt \
    --checkpoint      ./checkpoints/<run_name>.joblib \
    --categories_file ./data/dontpatronizeme_categories.tsv \
    --embedding_model mxbai
```

Results are saved to `results/local_eval/`.

---

## Output Structure

```
checkpoints/        saved model checkpoints (.joblib or .pt)
embedding_cache/    cached sentence embeddings (.npy)
predictions/
  local_test.txt    predictions on labelled local test set
  test.txt          predictions for marker submission
results/
  local_eval/
    error_analysis/
    ablation/
    metrics/
    category_analysis/
runs/               TensorBoard logs
```

---

## Data Splits

| Split | Source | Purpose |
|---|---|---|
| 80% of train | `train_semeval_parids-labels.csv` | Training |
| 20% of train | `train_semeval_parids-labels.csv` | Internal validation / threshold tuning |
| dev | `dev_semeval_parids-labels.csv` | Local test set (reported results) |
| test | unlabelled | Marker submission |

**Note:** As per the submission instructions, all files required for marking have been copied into the `BestModel/` directory. This includes the trained model checkpoint (`checkpoints/svm_mxbai_20260304_174317.joblib`), predictions (`predictions/dev.txt`, `predictions/test.txt`), training code (`train.py`, `train.sh`), and utility modules (`utils/helper.py`, `utils/load_data.py`). The checkpoint can be loaded directly using `joblib` to reproduce the reported results without retraining.