import pytest
from src.recommender import GiftRecommender
from src.utils.text_processors import (
    extract_age,
    extract_gender,
    extract_interests,
    extract_dislikes
)

@pytest.fixture
def recommender():
    """Create a recommender instance for testing."""
    return GiftRecommender()

@pytest.fixture
def sample_text():
    """Provide sample text for testing."""
    return """I'm looking for a gift for my 25-year-old sister. 
              She loves painting and traveling, especially in Japan. 
              She hates loud noises and doesn't like spicy food."""

class TestTextProcessors:
    """Tests for text processing utility functions."""

    def test_extract_age(self):
        """Test age extraction from text."""
        test_cases = [
            ("25-year-old sister", 25),
            ("my sister is 30", 30),
            ("she is 25 years old", 25),
            ("no age here", None),
            ("age is 150 years", None),  # Invalid age
        ]
        
        for text, expected in test_cases:
            assert extract_age(text) == expected

    def test_extract_gender(self):
        """Test gender extraction from text."""
        test_cases = [
            ("my sister likes", "female"),
            ("his brother wants", "male"),
            ("they like", "unknown"),
            ("my mother enjoys", "female"),
            ("dad loves", "male"),
        ]
        
        for text, expected in test_cases:
            assert extract_gender(text) == expected

    def test_extract_interests(self):
        """Test interest extraction and categorization."""
        text = "She loves painting and enjoys traveling"
        categories = ["art", "travel", "music"]
        
        interests = extract_interests(text, categories)
        
        assert len(interests) == 2
        assert any(i['phrase'] == 'painting' for i in interests)
        assert any(i['phrase'] == 'traveling' for i in interests)
        assert all('confidence' in i for i in interests)
        assert all('sentiment' in i for i in interests)

    def test_extract_dislikes(self):
        """Test dislike extraction from text."""
        text = "She hates loud noises and doesn't like spicy food"
        dislikes = extract_dislikes(text)
        
        assert len(dislikes) == 2
        assert "loud noises" in dislikes
        assert "spicy food" in dislikes

class TestGiftRecommender:
    """Tests for main GiftRecommender class."""

    def test_get_recommendations(self, recommender, sample_text):
        """Test the complete recommendation process."""
        results = recommender.get_gift_recommendations(sample_text)
        
        # Check if results contain all required fields
        assert 'profile' in results
        assert 'recommendations' in results
        
        # Check profile data
        profile = results['profile']
        assert profile['age'] == 25
        assert profile['gender'] == 'female'
        assert len(profile['interests']) > 0
        assert len(profile['dislikes']) > 0
        
        # Check recommendations
        recommendations = results['recommendations']
        assert len(recommendations) > 0
        assert all('gift' in r for r in recommendations)
        assert all('reason' in r for r in recommendations)

    def test_recommendation_relevance(self, recommender):
        """Test if recommendations are relevant to interests."""
        text = "My brother loves gaming and technology"
        results = recommender.get_gift_recommendations(text)
        
        recommendations = results['recommendations']
        assert any('gaming' in r['category'].lower() for r in recommendations)
        assert any('technology' in r['category'].lower() for r in recommendations)

    def test_format_recommendations(self, recommender, sample_text):
        """Test recommendation formatting."""
        results = recommender.get_gift_recommendations(sample_text)
        formatted = recommender.format_recommendations(results)
        
        assert isinstance(formatted, str)
        assert "Profile Summary" in formatted
        assert "Top Recommendations" in formatted
        assert "Age: 25" in formatted
        assert "Gender: Female" in formatted

    def test_empty_input(self, recommender):
        """Test handling of empty input."""
        results = recommender.get_gift_recommendations("")
        
        assert results['profile']['age'] is None
        assert results['profile']['gender'] == 'unknown'
        assert len(results['recommendations']) == 0

    @pytest.mark.parametrize("text,expected_count", [
        ("She loves art and music", 2),
        ("He likes gaming", 1),
        ("They enjoy reading, cooking, and traveling", 3),
    ])
    def test_interest_count(self, recommender, text, expected_count):
        """Test counting of extracted interests."""
        results = recommender.get_gift_recommendations(text)
        assert len(results['profile']['interests']) == expected_count

if __name__ == '__main__':
    pytest.main([__file__]) 
