from flask import Flask, request, jsonify
from src.recommender import GiftRecommender

app = Flask(__name__)
recommender = GiftRecommender()

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No input provided"}), 400
    
    results = recommender.get_gift_recommendations(text)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
