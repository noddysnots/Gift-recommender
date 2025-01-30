# NLP Gift Recommender

An NLP-powered gift recommendation system using Hugging Face transformers.

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nlp-gift-recommender.git
cd nlp-gift-recommender
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

```python
from src.recommender import GiftRecommender

recommender = GiftRecommender()
text = "I'm looking for a gift for my 25-year-old sister who loves painting."
recommendations = recommender.get_gift_recommendations(text)
print(recommender.format_recommendations(recommendations))
```

## Features

- Natural language processing for gift preferences
- Sentiment analysis for interest weighting
- Customizable gift categories and rules
- Detailed recommendation explanations

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request