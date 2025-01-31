import os
from flask import Flask, request, jsonify

from src.recommender import GiftRecommender

app = Flask(__name__)
recommender = GiftRecommender()

# 🏠 Homepage Route (Fixes 404)
@app.route("/")
def home():
    return "<h2>Welcome to the NLP Gift Recommender API!</h2><p>Use the /recommend endpoint to get recommendations.</p>"

# 🎁 Gift Recommendation API
@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No input provided"}), 400

    results = recommender.get_gift_recommendations(text)
    return jsonify(results)

# ✅ Ensure Correct Port Binding for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to port 5000
    app.run(host="0.0.0.0", port=port)  # Binds to all network interfaces
