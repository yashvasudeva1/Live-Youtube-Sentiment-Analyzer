# YouTube Comment Sentiment Analyzer

A full-stack application that analyzes the sentiment of YouTube video comments. It uses a custom-trained LightGBM model exposed via a FastAPI backend, paired with a Chrome Extension for on-the-fly analysis directly from your browser.

## Features

- **Pagination & Unlimited Fetching**: Retrieves all comments for a given video using the YouTube Data API with exponential backoff handling.
- **Sentiment Inference**: Utilizes a pre-trained LightGBM model with TF-IDF vectorizers to classify text as Positive, Negative, or Neutral.
- **Chrome Extension UI**: A dashboard-style popup providing visual breakdowns, month-by-month trends, and highlighted top comments.
- **Docker Support**: Containerized backend for easy deployment.

## Repository Structure

- `backend/`: FastAPI application code and Dockerfile.
- `frontend/`: Chrome Extension source files (Manifest v3, HTML, CSS, JS).
- `models/`: Exported `.joblib` model artifacts and vectorizers.
- `experimentation/`: Scripts for local performance and accuracy benchmarking.
- `data/`: Datasets for training and testing.

## Prerequisites

- Python 3.10+
- Docker (optional)
- YouTube Data API v3 Key (Get it from [Google Cloud Console](https://console.cloud.google.com/))

## Setup & Execution

### 1. Running the Backend Locally

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. Set your API key and run the server:
   ```bash
   # On Windows PowerShell
   $env:YOUTUBE_API_KEY="YOUR_API_KEY_HERE"
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   
   # On Linux/macOS
   export YOUTUBE_API_KEY="YOUR_API_KEY_HERE"
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

### 2. Running with Docker

Build and run the container:

```bash
docker build -f backend/Dockerfile -t yt-sentiment-api .
docker run -p 8000:8000 -e YOUTUBE_API_KEY="YOUR_API_KEY_HERE" yt-sentiment-api
```

### 3. Installing the Chrome Extension

1. Open Google Chrome and navigate to `chrome://extensions/`.
2. Toggle **Developer mode** in the top right corner.
3. Click **Load unpacked** and select the `frontend/` directory in this repository.
4. Open any YouTube video, click the extension icon, and select **Analyze Current Video**.

## Benchmarks

A script is provided in `experimentation/performance_test.py` to evaluate local model latency and throughput. On a standard machine, the batch prediction throughput is roughly 8,000+ comments per second with an average latency of ~0.11 ms per comment. Accuracy on a 50k validation split sits at roughly 80%.

## License

MIT License
