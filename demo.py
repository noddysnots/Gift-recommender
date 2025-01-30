from src.recommender import GiftRecommender

def main():
    recommender = GiftRecommender()
    print("\n🎁 Gift Recommendation System")
    print("-" * 30)
    print("Example: '25-year-old sister who loves painting and traveling'")
    print("Or simply enter an interest like 'games' or 'sports'.")

    while True:
        try:
            text = input("\nWho are you shopping for? ('quit' to exit): ").strip()
            if text.lower() == 'quit':
                break

            if len(text) < 3:
                print("⚠️ Please provide more details or a valid interest.")
                continue

            # Get recommendations
            results = recommender.get_gift_recommendations(text)
            
            # If no structured interests found, assume the input itself is an interest
            if not results['profile']['interests']:
                print(f"\n🔍 No specific interests found, assuming '{text}' is the interest.")
                results['profile']['interests'].append({'phrase': text, 'category': text, 'confidence': 1.0, 'sentiment': 'POSITIVE', 'sentiment_score': 1.0})

                # Fetch new recommendations based on this interest
                results['recommendations'] = recommender.generate_gift_ideas(text)

            print("\n" + recommender.format_recommendations(results))

            print("\n🎁 Suggested Gifts:")
            for i, rec in enumerate(results['recommendations'], 1):
                print(f"{i}. {rec['gift']}")

        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("🔄 Please try again with different wording.")

if __name__ == "__main__":
    main()
