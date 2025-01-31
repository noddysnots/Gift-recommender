# recommender.py

from transformers import pipeline

zero_shot = pipeline(
    "zero-shot-classification", 
    model="typeform/distilbert-base-uncased-mnli",  # Smaller model
    device=-1  # Ensures CPU usage without unnecessary overhead
)

from .utils.text_processors import (
    extract_age,
    extract_gender,
    extract_interests,
    extract_dislikes
)

class GiftRecommender:
    
    def __init__(self):
        self.zero_shot = pipeline("zero-shot-classification")
        self.sentiment = pipeline("sentiment-analysis")
        
        # List of possible interest categories
        self.interest_categories = [
            "art", "music", "sports", "technology", "reading",
            "travel", "cooking", "gaming", "fashion", "outdoor activities"
        ]
        
        # Pre-defined gift suggestions for each category
        self.gift_rules = {
            "art": ["art supplies set", "digital drawing tablet", "museum membership"],
            "music": ["wireless headphones", "concert tickets", "vinyl records"],
            "sports": ["fitness tracker", "sports equipment", "team merchandise"],
            "technology": ["smart devices", "electronics", "tech gadgets"],
            "gaming": ["gaming console", "gaming accessories", "game subscription"],
            "travel": ["travel gear", "language courses", "travel guides"],
            "reading": ["e-reader", "book subscription", "rare books"],
            "cooking": ["cooking classes", "kitchen gadgets", "recipe books"]
        }

    def get_gift_recommendations(self, text: str):
        # Build the user's profile from the text
        profile = {
            'age': extract_age(text),
            'gender': extract_gender(text),
            'interests': extract_interests(text, self.interest_categories),
            'dislikes': extract_dislikes(text)
        }
        
        # Match each extracted interest to possible gift ideas
        recommendations = []
        for interest in profile['interests']:
            cat = interest['category']
            if cat in self.gift_rules:
                for gift in self.gift_rules[cat]:
                    recommendations.append({
                        'gift': gift,
                        'category': cat,
                        'reason': f"Based on interest in {interest['phrase']}"
                    })
        
        # Limit to top 5 for demonstration
        return {'profile': profile, 'recommendations': recommendations[:5]}

    def format_recommendations(self, results: dict) -> str:
        output = []
        output.append("🎁 Gift Recommendations\n")
        
        profile = results['profile']
        output.append("Profile Summary:")
        output.append(f"Age: {profile['age'] or 'Unknown'}")
        output.append(f"Gender: {profile['gender'].title()}")
        
        if profile['interests']:
            output.append("Interests: " + ", ".join(i['phrase'] for i in profile['interests']))
        if profile['dislikes']:
            output.append("Dislikes: " + ", ".join(profile['dislikes']))
        
        if results['recommendations']:
            output.append("\nTop Recommendations:")
            for i, rec in enumerate(results['recommendations'], 1):
                output.append(f"{i}. {rec['gift']}")
                output.append(f"   • {rec['reason']}")
        
        return "\n".join(output)
