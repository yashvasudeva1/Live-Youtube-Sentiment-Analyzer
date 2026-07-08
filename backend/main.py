from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import random
from datetime import datetime, timedelta
import pandas as pd
import joblib
from scipy.sparse import hstack
from typing import Optional
import time

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    from googleapiclient.discovery import build
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False

app = FastAPI(title="YouTube Sentiment Analysis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mlflow_model = None
local_lgbm = None
tfidf_word = None
tfidf_char = None
label_encoder = None

@app.on_event("startup")
def load_models():
    global mlflow_model, local_lgbm, tfidf_word, tfidf_char, label_encoder

    RUN_ID = "ad3f15e56f454190abe50e638a0de871"
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri("sqlite:///mlflow.db")
            mlflow_model = mlflow.pyfunc.load_model(f"runs:/{RUN_ID}/model")
            print(f"Successfully loaded MLflow model from run {RUN_ID}")
            return
        except Exception as e:
            print(f"MLflow model load failed: {e}. Falling back to local joblib files...")

    try:
        model_dir = os.path.join(os.getcwd(), "models")
        local_lgbm = joblib.load(os.path.join(model_dir, "lgbm_sentiment_model.joblib"))
        tfidf_word = joblib.load(os.path.join(model_dir, "tfidf_word_vectorizer.joblib"))
        tfidf_char = joblib.load(os.path.join(model_dir, "tfidf_char_vectorizer.joblib"))
        label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        print("Successfully loaded local LightGBM model and vectorizers.")
    except Exception as e:
        print(f"Warning: Could not load local models: {e}. Will use mock predictions.")


# ---------- Pydantic Models ----------

class AnalyzeRequest(BaseModel):
    video_id: str

class SentimentData(BaseModel):
    month: str
    positive: int
    negative: int
    neutral: int
    total: int

class CommentItem(BaseModel):
    text: str
    sentiment: str

class AnalyzeResponse(BaseModel):
    video_id: str
    total_comments: int
    sentiments: dict
    sentiment_pct: dict
    monthly_data: list[SentimentData]
    top_positive: list[str]
    top_negative: list[str]
    most_active_month: Optional[str]
    dominant_sentiment: str
    avg_comments_per_month: float


# ---------- Comment Fetching ----------

def get_youtube_comments(video_id: str):
    api_key = os.environ.get("YOUTUBE_API_KEY")
    if not api_key or not GOOGLE_API_AVAILABLE:
        print("No YOUTUBE_API_KEY found. Generating mock comments.")
        return generate_mock_comments(video_id, 200)

    try:
        youtube = build("youtube", "v3", developerKey=api_key)
        comments = []
        next_page_token = None

        while True:
            retries = 0
            max_retries = 5
            success = False
            
            while retries < max_retries and not success:
                try:
                    params = dict(
                        part="snippet",
                        videoId=video_id,
                        maxResults=100,
                        textFormat="plainText",
                    )
                    if next_page_token:
                        params["pageToken"] = next_page_token

                    response = youtube.commentThreads().list(**params).execute()

                    for item in response.get("items", []):
                        snippet = item["snippet"]["topLevelComment"]["snippet"]
                        comments.append({
                            "text": snippet["textDisplay"],
                            "date": snippet["publishedAt"],
                            "likes": snippet.get("likeCount", 0),
                        })

                    next_page_token = response.get("nextPageToken")
                    success = True
                    
                except Exception as page_error:
                    retries += 1
                    print(f"Pagination error (fetched {len(comments)} so far, retry {retries}/{max_retries}): {page_error}")
                    if retries < max_retries:
                        time.sleep(2 ** retries) # Exponential backoff: 2s, 4s, 8s, 16s...
                    else:
                        print("Max retries reached. Stopping fetch.")
                        break
            
            if not success or not next_page_token:
                break

        print(f"Fetched {len(comments)} comments for video {video_id}")
        
        # If we successfully fetched real comments, return them
        if len(comments) > 0:
            return comments
            
    except Exception as e:
        print(f"Fatal error fetching YouTube comments: {e}")
        
    # Only fall back to mock data if absolutely necessary
    print("Falling back to generating mock comments.")
    return generate_mock_comments(video_id, 200)


def generate_mock_comments(video_id: str, count: int):
    texts = [
        "This video is absolutely amazing, loved every second!",
        "Not my favorite, felt a bit too long.",
        "Could be better honestly.",
        "One of the best videos I've ever watched.",
        "The content is decent, nothing too special.",
        "Really helpful, thank you for making this!",
        "I disagree with most of the points here.",
        "Great editing and production quality.",
        "A bit boring but informative.",
        "Would definitely recommend to others.",
    ]
    now = datetime.utcnow()
    return [
        {
            "text": random.choice(texts),
            "date": (now - timedelta(days=random.randint(0, 365))).isoformat(),
            "likes": random.randint(0, 500),
        }
        for _ in range(count)
    ]


# ---------- Prediction ----------

def predict_single_text(text: str) -> str:
    if mlflow_model is not None:
        try:
            pred = mlflow_model.predict(pd.DataFrame([{"text": text}]))
            return str(pred[0]).lower()
        except:
            pass

    if local_lgbm is not None:
        try:
            word_features = tfidf_word.transform([text])
            char_features = tfidf_char.transform([text])
            combined = hstack([word_features, char_features])
            encoded = local_lgbm.predict(combined)
            label = label_encoder.inverse_transform(encoded)[0]
            label_lower = str(label).lower()
            if "pos" in label_lower: return "positive"
            if "neg" in label_lower: return "negative"
            return "neutral"
        except Exception as e:
            print(f"Prediction error: {e}")

    return random.choice(["positive", "negative", "neutral"])


def analyze_sentiment(comments: list):
    return [
        {**c, "sentiment": predict_single_text(c["text"])}
        for c in comments
    ]


# ---------- Endpoint ----------

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_video(request: AnalyzeRequest):
    video_id = request.video_id
    if not video_id:
        raise HTTPException(status_code=400, detail="Video ID is required")

    raw_comments = get_youtube_comments(video_id)
    analyzed = analyze_sentiment(raw_comments)

    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    monthly_counts: dict = {}
    top_positive: list[dict] = []
    top_negative: list[dict] = []

    for c in analyzed:
        s = c["sentiment"]
        sentiment_counts[s] = sentiment_counts.get(s, 0) + 1

        try:
            dt = datetime.fromisoformat(c["date"].replace("Z", "+00:00"))
            month_key = dt.strftime("%Y-%m")
        except:
            month_key = "Unknown"

        monthly_counts.setdefault(month_key, {"positive": 0, "negative": 0, "neutral": 0, "total": 0})
        monthly_counts[month_key][s] = monthly_counts[month_key].get(s, 0) + 1
        monthly_counts[month_key]["total"] += 1

        if s == "positive" and len(top_positive) < 3:
            top_positive.append(c)
        if s == "negative" and len(top_negative) < 3:
            top_negative.append(c)

    total = len(analyzed)
    sentiment_pct = {
        k: round((v / total) * 100, 1) if total else 0
        for k, v in sentiment_counts.items()
    }

    monthly_data_list = [
        SentimentData(
            month=m,
            positive=monthly_counts[m].get("positive", 0),
            negative=monthly_counts[m].get("negative", 0),
            neutral=monthly_counts[m].get("neutral", 0),
            total=monthly_counts[m]["total"],
        )
        for m in sorted(monthly_counts.keys())
    ]

    most_active_month = (
        max(monthly_counts, key=lambda m: monthly_counts[m]["total"])
        if monthly_counts else None
    )
    dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
    avg_per_month = round(total / len(monthly_counts), 1) if monthly_counts else 0.0

    return AnalyzeResponse(
        video_id=video_id,
        total_comments=total,
        sentiments=sentiment_counts,
        sentiment_pct=sentiment_pct,
        monthly_data=monthly_data_list,
        top_positive=[c["text"] for c in top_positive],
        top_negative=[c["text"] for c in top_negative],
        most_active_month=most_active_month,
        dominant_sentiment=dominant_sentiment,
        avg_comments_per_month=avg_per_month,
    )
