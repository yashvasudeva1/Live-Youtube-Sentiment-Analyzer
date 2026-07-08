import time
import pandas as pd
import joblib
from scipy.sparse import hstack
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

print("--- Sentiment Model Accuracy Evaluation ---")

# 1. Load Models
print("Loading models...")
try:
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    local_lgbm = joblib.load(os.path.join(model_dir, "lgbm_sentiment_model.joblib"))
    tfidf_word = joblib.load(os.path.join(model_dir, "tfidf_word_vectorizer.joblib"))
    tfidf_char = joblib.load(os.path.join(model_dir, "tfidf_char_vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    print("Models loaded successfully.\n")
except Exception as e:
    print(f"Failed to load models: {e}")
    exit(1)

# 2. Load Dataset
data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "processed", "all_categories_comments_labeled.csv")
print(f"Loading dataset from: {data_path}")

try:
    # Read the data (using a sample for speed, but you can remove nrows to test on all ~90MB of data)
    # 50,000 rows is a good representative sample that will run quickly
    df = pd.read_csv(data_path, nrows=50000)
    
    # Drop rows with missing text or sentiment
    df = df.dropna(subset=['text', 'sentiment'])
    print(f"Dataset loaded. Testing on {len(df):,} comments.\n")
except Exception as e:
    print(f"Failed to load dataset: {e}")
    exit(1)

# 3. Prepare Ground Truth
true_labels = df['sentiment'].str.lower().tolist()

# 4. Predict
print("Running predictions (this may take a few seconds)...")
start_time = time.time()

word_features = tfidf_word.transform(df['text'])
char_features = tfidf_char.transform(df['text'])
combined_features = hstack([word_features, char_features])

pred_encoded = local_lgbm.predict(combined_features)
pred_labels_raw = label_encoder.inverse_transform(pred_encoded)

# Normalize predictions to lowercase strings matching ground truth
pred_labels = []
for label in pred_labels_raw:
    l_lower = str(label).lower()
    if "pos" in l_lower: pred_labels.append("positive")
    elif "neg" in l_lower: pred_labels.append("negative")
    else: pred_labels.append("neutral")

print(f"Predictions completed in {time.time() - start_time:.2f} seconds.\n")

# 5. Calculate Metrics
print("-" * 50)
print("ACCURACY SCORE:")
print(f"{accuracy_score(true_labels, pred_labels) * 100:.2f}%\n")

print("-" * 50)
print("CLASSIFICATION REPORT (Precision, Recall, F1-Score):")
print(classification_report(true_labels, pred_labels, target_names=['negative', 'neutral', 'positive']))

print("-" * 50)
print("CONFUSION MATRIX:")
cm = confusion_matrix(true_labels, pred_labels, labels=['negative', 'neutral', 'positive'])
cm_df = pd.DataFrame(cm, 
                     index=['True Neg', 'True Neu', 'True Pos'], 
                     columns=['Pred Neg', 'Pred Neu', 'Pred Pos'])
print(cm_df)
print("-" * 50)
