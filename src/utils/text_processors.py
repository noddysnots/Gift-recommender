import re
from typing import Dict, List, Optional

from transformers import pipeline

# Instantiate your pipelines just once
zero_shot = pipeline("zero-shot-classification")
sentiment = pipeline("sentiment-analysis")

def extract_age(text: str) -> Optional[int]:
    age_pattern = r'\b(\d{1,2})\s*-?\s*years?\s*-?\s*old\b|\b(\d{1,2})\b'
    matches = re.findall(age_pattern, text)
    if matches:
        age = next(int(num) for nums in matches for num in nums if num)
        return age if 0 < age < 120 else None
    return None

def extract_gender(text: str) -> str:
    text_lower = text.lower()
    gender_indicators = {
        'male': ['he', 'him', 'his', 'brother', 'boyfriend', 'husband', 'son', 'dad', 'father'],
        'female': ['she', 'her', 'hers', 'sister', 'girlfriend', 'wife', 'daughter', 'mom', 'mother']
    }
    
    for gender, indicators in gender_indicators.items():
        if any(f" {indicator} " in f" {text_lower} " for indicator in indicators):
            return gender
    return "unknown"

def extract_interests(text: str, categories: List[str]) -> List[Dict]:
    """
    Extracts all interests after verbs like "love(s)", "like(s)", or "enjoy(s)" until we hit
    another recognized verb or the end of the text. Then splits on "and"/commas as standalone words,
    preserving original casing (so "painting" is recognized properly).

    Example:
      "She loves painting and enjoys traveling" -> ["painting", "traveling"]
      "She loves art and music" -> ["art", "music"]
    """
    import re
    from transformers import pipeline

    # Fresh pipelines each call (or you can move these outside)
    zero_shot = pipeline("zero-shot-classification")
    sentiment = pipeline("sentiment-analysis")
    
    # Tokenize by any non-whitespace
    tokens = re.findall(r"\S+", text)  
    n = len(tokens)
    
    # Recognized verbs (compare lowercased)
    verb_set = {"love", "loves", "like", "likes", "enjoy", "enjoys"}
    
    interests_list = []
    seen = set()
    
    i = 0
    while i < n:
        word_lower = tokens[i].lower()
        
        if word_lower in verb_set:
            # Collect subsequent tokens until next verb or end
            j = i + 1
            while j < n and tokens[j].lower() not in verb_set:
                j += 1
            
            # Now tokens i+1..j-1 form the chunk
            chunk_tokens = tokens[i+1 : j]
            if chunk_tokens:
                # e.g. ["painting", "and"]
                chunk_str = " ".join(chunk_tokens)
                
                # Key fix: split on standalone "and" or commas, ignoring case
                sub_parts = re.split(r'\s*,\s*|\s*\band\b\s*', chunk_str, flags=re.IGNORECASE)
                
                for candidate in sub_parts:
                    candidate = candidate.strip()
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        
                        # Zero-shot + sentiment
                        z_result = zero_shot(candidate, categories, multi_label=False)
                        s_result = sentiment(candidate)[0]
                        
                        interests_list.append({
                            'phrase': candidate,  # preserve original
                            'category': z_result['labels'][0],
                            'confidence': z_result['scores'][0],
                            'sentiment': s_result['label'],
                            'sentiment_score': s_result['score']
                        })
            
            i = j  # skip forward
        else:
            i += 1
    
    return interests_list

def extract_dislikes(text: str) -> List[str]:
    text_lower = text.lower()
    dislike_pattern = r'(?:hates|dislikes|(?:doesn\'t|does\s+not)\s+like)\s+([^,.]+?)(?=\s+and\s+|$|,)'
    matches = re.findall(dislike_pattern, text_lower)

    dislikes = []
    for match in matches:
        parts = re.split(r'(?:,\s*|\s+and\s+)', match)
        for p in parts:
            cleaned = p.replace("doesn't like ", "").replace("does not like ", "").strip()
            if cleaned:
                dislikes.append(cleaned)

    return dislikes

def format_profile(profile: Dict) -> str:
    output = []
    output.append("Profile Summary:")
    output.append(f"- Age: {profile['age'] or 'Unknown'}")
    output.append(f"- Gender: {profile['gender'].title()}")
    output.append("- Interests: " + ", ".join(i['phrase'] for i in profile['interests']))
    if profile['dislikes']:
        output.append("- Dislikes: " + ", ".join(profile['dislikes']))
    return "\n".join(output)
