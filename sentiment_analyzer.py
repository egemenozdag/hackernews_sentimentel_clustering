from transformers import pipeline
# Eğer başka kütüphaneler de uyarı veriyorsa, onları da bastırabilirsiniz

class SentimentAnalyzer:
    def __init__(self):
        self.model = pipeline('sentiment-analysis')

    def analyze_sentiment(self, text):
        """Verilen metnin sentiment analizini yapar."""
        result = self.model(text)
        return result[0]['label'], result[0]['score']
