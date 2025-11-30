import os
import tensorflow as tf
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
import uvicorn
from data_loader import get_vectorization_layer

app = FastAPI()

# Configuration
VOCAB_FILE = os.path.join('aclImdb', 'imdb.vocab')
MAX_TOKENS = 10000
MAX_LENGTH = 150
MODEL_PATH = 'best_model.keras'

# Global variables
model = None
vectorize_layer = None

@app.on_event("startup")
async def startup_event():
    global model, vectorize_layer
    
    # Load Model
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: {MODEL_PATH} not found. Please run train.py first.")
        
    # Prepare Vectorization
    if os.path.exists(VOCAB_FILE):
        vectorize_layer = get_vectorization_layer(VOCAB_FILE, MAX_TOKENS, MAX_LENGTH)
        print("Vectorization layer prepared.")
    else:
        print(f"Warning: {VOCAB_FILE} not found.")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>IMDB Sentiment Analysis</title>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; background-color: #f5f5f5; }
            .container { background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1 { color: #333; text-align: center; }
            textarea { width: 100%; height: 150px; margin-bottom: 20px; padding: 10px; border: 1px solid #ddd; border-radius: 4px; resize: vertical; font-family: inherit; }
            button { background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; }
            button:hover { background-color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Movie Review Sentiment Analysis</h1>
            <form action="/predict" method="post">
                <label for="review"><strong>Enter your movie review:</strong></label><br><br>
                <textarea id="review" name="review" placeholder="Type your review here..." required></textarea><br>
                <button type="submit">Analyze Sentiment</button>
            </form>
        </div>
    </body>
    </html>
    """
    return html_content

@app.post("/predict", response_class=HTMLResponse)
async def predict(review: str = Form(...)):
    global model, vectorize_layer
    
    if model is None:
        return HTMLResponse("<h1>Error: Model not loaded. Please run train.py first to generate best_model.keras</h1>")
    
    if vectorize_layer is None:
        return HTMLResponse("<h1>Error: Vectorization layer not initialized. Check imdb.vocab file.</h1>")
        
    # Preprocess
    # Expand dims to make it a batch of 1
    text_tensor = tf.expand_dims(review, -1)
    vectorized_text = vectorize_layer(text_tensor)
    
    # Predict
    prediction_prob = model.predict(vectorized_text)[0][0]
    
    sentiment = "Positive" if prediction_prob > 0.5 else "Negative"
    confidence = prediction_prob if sentiment == "Positive" else 1 - prediction_prob
    
    css_class = "pos" if sentiment == "Positive" else "neg"
    bg_color = "#d4edda" if sentiment == "Positive" else "#f8d7da"
    text_color = "#155724" if sentiment == "Positive" else "#721c24"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analysis Result</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 0 auto; padding: 40px; background-color: #f5f5f5; }}
            .container {{ background-color: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .result {{ padding: 20px; border-radius: 5px; margin-top: 20px; text-align: center; background-color: {bg_color}; color: {text_color}; }}
            h1 {{ color: #333; text-align: center; }}
            .review-box {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #ccc; margin: 20px 0; font-style: italic; }}
            a {{ display: block; text-align: center; margin-top: 20px; color: #007bff; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Analysis Result</h1>
            <div class="review-box">"{review}"</div>
            <div class="result">
                <h2>Sentiment: {sentiment}</h2>
                <p>Confidence: {confidence:.2%}</p>
            </div>
            <a href="/">&larr; Analyze another review</a>
        </div>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
