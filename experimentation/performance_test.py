import time
import pandas as pd
import joblib
from scipy.sparse import hstack
import os

print("--- Sentiment Model Performance Test ---")

# 1. Load the Models
print("Loading models...")
start_load = time.time()
try:
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    
    local_lgbm = joblib.load(os.path.join(model_dir, "lgbm_sentiment_model.joblib"))
    tfidf_word = joblib.load(os.path.join(model_dir, "tfidf_word_vectorizer.joblib"))
    tfidf_char = joblib.load(os.path.join(model_dir, "tfidf_char_vectorizer.joblib"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
    
    load_time = time.time() - start_load
    print(f"Models loaded successfully in {load_time:.3f} seconds.\n")
except Exception as e:
    print(f"Failed to load models: {e}")
    exit(1)


# 2. Generate Test Data
print("Generating test data...")
sample_texts = [
    "This video is absolutely amazing, loved every second!",
    "Not my favorite, felt a bit too long.",
    "Could be better honestly.",
    "One of the best videos I've ever watched.",
    "The content is decent, nothing too special.",
    "Really helpful, thank you for making this!",
    "I disagree with most of the points here.",
    "Great editing and production quality.",
    "A bit boring but informative.",
    "Would definitely recommend to others."
]

def run_performance_test(batch_size):
    # Duplicate sample texts to match batch size
    texts = (sample_texts * (batch_size // len(sample_texts) + 1))[:batch_size]
    
    print(f"Testing Batch Size: {batch_size:,} comments")
    
    # Measure Vectorization
    start_vec = time.time()
    word_features = tfidf_word.transform(texts)
    char_features = tfidf_char.transform(texts)
    combined_features = hstack([word_features, char_features])
    vec_time = time.time() - start_vec
    
    # Measure Prediction
    start_pred = time.time()
    pred_encoded = local_lgbm.predict(combined_features)
    labels = label_encoder.inverse_transform(pred_encoded)
    pred_time = time.time() - start_pred
    
    total_time = vec_time + pred_time
    throughput = batch_size / total_time
    avg_latency = (total_time / batch_size) * 1000 # in ms
    
    print(f"  Vectorization Time : {vec_time:.4f}s")
    print(f"  Prediction Time    : {pred_time:.4f}s")
    print(f"  Total Time         : {total_time:.4f}s")
    print(f"  Average Latency    : {avg_latency:.4f} ms / comment")
    print(f"  Throughput         : {throughput:,.0f} comments / sec")
    print("-" * 40)


# 3. Run tests with varying batch sizes
batch_sizes = [100, 1000, 5000, 10000, 50000]

for size in batch_sizes:
    run_performance_test(size)

print("Performance testing complete.")
