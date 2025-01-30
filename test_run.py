from src.recommender import GiftRecommender

recommender = GiftRecommender()
text = "I'm looking for a gift for my 25-year-old sister who loves painting."
results = recommender.get_gift_recommendations(text)
print(recommender.format_recommendations(results))