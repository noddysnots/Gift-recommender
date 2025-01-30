from transformers import pipeline
import re
from typing import Dict, List, Optional

# Initialize the NLP pipelines
zero_shot = pipeline("zero-shot-classification")
sentiment = pipeline("sentiment-analysis")

def extract_age(text: str) -> Optional[int]:
    """
    Extract age from text using regex.
    
    Args:
        text (str): Input text containing age information
        
    Returns:
        Optional[int]: Extracted age or None if not found
    """
    age_pattern = r'\b(\d{1,2})\s*-?\s*years?\s*-?\s*old\b|\b(\d{1,2})\b'
    matches = re.findall(age_pattern, text)
    if matches:
        # Return the first number found
        age = next(int(num) for nums in matches for num in nums if num)
        return age if 0 < age < 120 else None
    return None

def extract_gender(text: str) -> str:
    """
    Extract gender from text using keywords.
    
    Args:
        text (str): Input text containing gender indicators
        
    Returns:
        str: Detected gender ('male', 'female', or 'unknown')
    """
    text = text.lower()
    gender_indicators = {
        'female': ['she', 'her', 'sister', 'girlfriend', 'wife', 'daughter', 'mom', 'mother'],
        'male': ['he', 'him', 'brother', 'boyfriend', 'husband', 'son', 'dad', 'father']
    }
    
    for gender, indicators in gender_indicators.items():
        if any(indicator in text for indicator in indicators):
            return gender
    return "unknown"

def extract_interests(text: str, categories: List[str]) -> List[Dict]:
    """
    Extract and categorize interests using zero-shot classification.
    
    Args:
        text (str): Input text containing interest information
        categories (List[str]): List of possible interest categories
        
    Returns:
        List[Dict]: List of detected interests with categories and sentiment scores
    """
    # Extract potential interest phrases
    interest_pattern = r'loves?\s+([^,.]+)|\blikes?\s+([^,.]+)'
    matches = re.findall(interest_pattern, text.lower())
    
    interests = []
    for match in matches:
        phrase = next(m for m in match if m)
        # Classify the interest into predefined categories
        result = zero_shot(
            phrase,
            candidate_labels=categories,
            multi_label=False
        )
        
        # Get sentiment score for the interest
        sentiment_score = sentiment(phrase)[0]
        
        interests.append({
            'phrase': phrase,
            'category': result['labels'][0],
            'confidence': result['scores'][0],
            'sentiment': sentiment_score['label'],
            'sentiment_score': sentiment_score['score']
        })
    
    return interests

def extract_dislikes(text: str) -> List[str]:
    """
    Extract dislikes from text.
    
    Args:
        text (str): Input text containing dislike information
        
    Returns:
        List[str]: List of extracted dislikes
    """
    dislike_pattern = r'hates?\s+([^,.]+)|dislikes?\s+([^,.]+)'
    matches = re.findall(dislike_pattern, text.lower())
    return [next(m for m in match if m) for match in matches]

def format_profile(profile: Dict) -> str:
    """
    Format the profile information into a readable string.
    
    Args:
        profile (Dict): Dictionary containing profile information
        
    Returns:
        str: Formatted profile string
    """
    output = []
    output.append("Profile Summary:")
    output.append(f"- Age: {profile['age'] or 'Unknown'}")
    output.append(f"- Gender: {profile['gender'].title()}")
    output.append("- Interests: " + ", ".join(i['phrase'] for i in profile['interests']))
    if profile['dislikes']:
        output.append("- Dislikes: " + ", ".join(profile['dislikes']))
    return "\n".join(output) 
