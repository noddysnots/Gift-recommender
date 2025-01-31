import gradio as gr
import requests
from transformers import pipeline

# Load NLP model
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 🎁 Web search for gift suggestions
def search_gifts(query):
    amazon_url = f"https://www.amazon.in/s?k={query.replace(' ', '+')}"
    igp_url = f"https://www.igp.com/search?q={query.replace(' ', '+')}"
    indiamart_url = f"https://dir.indiamart.com/search.mp?ss={query.replace(' ', '+')}"

    return {"Amazon": amazon_url, "IGP": igp_url, "IndiaMart": indiamart_url}

# 🎯 Main function for gift recommendation
def recommend_gifts(text):
    if not text:
        return "Please enter a description."

    # NLP Processing
    categories = ["art", "music", "tech", "travel", "books", "fashion", "fitness", "gaming"]
    results = zero_shot(text, categories)

    # Get top interest
    top_interest = results["labels"][0]

    # Search for gifts based on interest
    links = search_gifts(top_interest)

    return {
        "Predicted Interest": top_interest,
        "Gift Suggestions": links
    }

# 🎨 Gradio UI for easy interaction
demo = gr.Interface(
    fn=recommend_gifts, 
    inputs="text", 
    outputs="json",
    title="🎁 AI Gift Recommender",
    description="Enter details about the person you are buying a gift for, and get personalized suggestions with shopping links!",
)

# 🚀 Launch Gradio App
if __name__ == "__main__":
    demo.launch()
