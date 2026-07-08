import os
import re
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from gensim.models import Word2Vec

DATA_PATH = r"C:\Users\HP\Desktop\YT Live Sentiment Analysis\data\processed\all_categories_comments_labeled.csv"
TEXT_COL = "text"
TARGET_COL = "sentiment"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MLFLOW_EXPERIMENT_NAME = "sentiment_classification_pipeline"
W2V_VECTOR_SIZE = 150
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_EPOCHS = 15
BOW_MAX_FEATURES = 20000
TFIDF_MAX_FEATURES = 20000
STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    return tokens


def load_data(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])
    df["tokens"] = df[TEXT_COL].apply(clean_text)
    df["clean_text"] = df["tokens"].apply(lambda x: " ".join(x))
    df = df[df["clean_text"].str.strip() != ""]
    return df.reset_index(drop=True)


def encode_labels(y_train, y_test):
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)
    return y_train_enc, y_test_enc, le


def build_bow(train_texts, test_texts):
    vectorizer = CountVectorizer(max_features=BOW_MAX_FEATURES, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def build_tfidf(train_texts, test_texts):
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer


def train_word2vec(train_tokens):
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        epochs=W2V_EPOCHS,
        workers=os.cpu_count() or 1,
        seed=RANDOM_STATE,
    )
    return model


def vectorize_w2v(tokens_list, w2v_model):
    vectors = []
    for tokens in tokens_list:
        word_vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
        if len(word_vecs) == 0:
            vectors.append(np.zeros(w2v_model.vector_size))
        else:
            vectors.append(np.mean(word_vecs, axis=0))
    return np.array(vectors)


def build_word2vec(train_tokens, test_tokens):
    w2v_model = train_word2vec(train_tokens)
    X_train = vectorize_w2v(train_tokens, w2v_model)
    X_test = vectorize_w2v(test_tokens, w2v_model)
    return X_train, X_test, w2v_model


def get_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            tree_method="hist",
        ),
        "lightgbm": LGBMClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }


def apply_imbalance_strategy(strategy, model_name, model, X_train, y_train):
    fit_kwargs = {}
    if strategy == "class_weight":
        if model_name in ("logistic_regression", "random_forest"):
            model.set_params(class_weight="balanced")
        elif model_name in ("xgboost", "lightgbm"):
            fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
        return model, X_train, y_train, fit_kwargs
    elif strategy == "smote":
        smote = SMOTE(random_state=RANDOM_STATE)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        return model, X_res, y_res, fit_kwargs
    else:
        return model, X_train, y_train, fit_kwargs


def compute_metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_accuracy": accuracy_score(y_true, y_pred),
        f"{prefix}_precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        f"{prefix}_f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def log_confusion_matrix(y_true, y_pred, labels, run_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(run_name)
    path = f"{run_name}_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path)
    os.remove(path)


def run_experiment(embedding_name, model_name, imbalance_strategy, X_train, X_test, y_train, y_test, label_encoder):
    run_name = f"{embedding_name}_{model_name}_{imbalance_strategy}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("embedding", embedding_name)
        mlflow.log_param("model", model_name)
        mlflow.log_param("imbalance_strategy", imbalance_strategy)

        models = get_models()
        model = models[model_name]
        model, X_train_used, y_train_used, fit_kwargs = apply_imbalance_strategy(
            imbalance_strategy, model_name, model, X_train, y_train
        )

        model.fit(X_train_used, y_train_used, **fit_kwargs)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_metrics = compute_metrics(y_train, y_train_pred, "train")
        test_metrics = compute_metrics(y_test, y_test_pred, "test")

        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(test_metrics)

        classes = label_encoder.classes_
        log_confusion_matrix(y_test, y_test_pred, classes, run_name)

        if model_name == "xgboost":
            mlflow.xgboost.log_model(model, "model")
        elif model_name == "lightgbm":
            mlflow.lightgbm.log_model(model, "model")
        else:
            mlflow.sklearn.log_model(model, "model")

        report = classification_report(y_test, y_test_pred, target_names=classes, zero_division=0)
        report_path = f"{run_name}_classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)
        os.remove(report_path)

        print(run_name, test_metrics)


def main():
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    df = load_data(DATA_PATH)

    X = df["clean_text"].values
    tokens = df["tokens"].values
    y = df[TARGET_COL].values

    X_train_text, X_test_text, tokens_train, tokens_test, y_train_raw, y_test_raw = train_test_split(
        X, tokens, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    y_train, y_test, label_encoder = encode_labels(y_train_raw, y_test_raw)

    embeddings = {}

    X_train_bow, X_test_bow, _ = build_bow(X_train_text, X_test_text)
    embeddings["bow"] = (X_train_bow, X_test_bow)

    X_train_tfidf, X_test_tfidf, _ = build_tfidf(X_train_text, X_test_text)
    embeddings["tfidf"] = (X_train_tfidf, X_test_tfidf)

    X_train_w2v, X_test_w2v, _ = build_word2vec(tokens_train, tokens_test)
    embeddings["word2vec"] = (X_train_w2v, X_test_w2v)

    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    imbalance_strategies = ["class_weight", "smote"]

    for embedding_name, (X_train_emb, X_test_emb) in embeddings.items():
        for model_name in model_names:
            for imbalance_strategy in imbalance_strategies:
                run_experiment(
                    embedding_name,
                    model_name,
                    imbalance_strategy,
                    X_train_emb,
                    X_test_emb,
                    y_train,
                    y_test,
                    label_encoder,
                )


if __name__ == "__main__":
    main()