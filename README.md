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

The model was rigorously tested for both accuracy and performance on large comment datasets. 

### Accuracy (Tested on 49,985 labeled comments)
- **Overall Accuracy**: 79.96%
- **Macro F1-Score**: 0.80
- **Positive Precision**: 0.91
- **Negative Recall**: 0.80

### Performance & Throughput (Tested on 50,000 comments)
- **Throughput**: ~8,869 comments processed per second
- **Total Pipeline Time**: 5.63 seconds (for 50k comments)
- **Average Latency**: 0.11 ms per comment

These metrics prove the system is highly capable of scaling to viral videos with hundreds of thousands of comments efficiently.

## License

MIT License
