from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

@app.route("/", methods=["GET", "POST"])
def home():
    sentiment = None
    confidence = None

    if request.method == "POST":
        text = request.form.get("text")

        # Run sentiment analysis on the input text
        sentiment_result = sentiment_pipeline(text)[0]
        sentiment = sentiment_result['label']
        confidence = sentiment_result['score']
    return render_template("index.html", sentiment=sentiment, confidence=confidence)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
