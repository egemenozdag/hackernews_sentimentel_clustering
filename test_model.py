from sentiment_analyzer import SentimentAnalyzer
from clustering import Clustering


def test_model():
    # Sentiment analizi
    sentiment_analyzer = SentimentAnalyzer()
    test_text = "Hacker News is a great platform for tech enthusiasts!"
    sentiment_label, sentiment_score = sentiment_analyzer.analyze_sentiment(test_text)
    print(f"Sentiment: {sentiment_label}, Score: {sentiment_score}")

    # Clustering
    clustering = Clustering(n_clusters=3)
    texts = [
        "Hacker News is full of tech news.",
        "I love learning about machine learning.",
        "AI is the future of technology.",
        "I love reading about technology.",
        "Space exploration is fascinating."
    ]
    clusters = clustering.fit(texts)
    print(f"Clustering labels: {clusters}")


if __name__ == "__main__":
    test_model()
