from data_collector import HackerNewsCollector
from sentiment_analyzer import SentimentAnalyzer
from clustering import Clustering
import warnings

# TensorFlow ve Hugging Face uyarılarını bastırma
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import transformers
transformers.logging.set_verbosity_error()
import logging
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()


# Eğer başka kütüphaneler de uyarı veriyorsa, onları da bastırabilirsiniz


def main():
    # Hacker News verilerini çek
    collector = HackerNewsCollector()
    top_stories = collector.fetch_top_stories(limit=5)
    print(f"Top Stories: {top_stories}")

    # Sentiment analizi yap
    sentiment_analyzer = SentimentAnalyzer()
    for story in top_stories:
        title = story.get('title', 'No title')
        sentiment_label, sentiment_score = sentiment_analyzer.analyze_sentiment(title)
        print(f"Sentiment of '{title}': {sentiment_label} (Score: {sentiment_score})")

    # Clustering yap
    clustering = Clustering(n_clusters=3)
    titles = [story.get('title', 'No title') for story in top_stories]
    clusters = clustering.fit(titles)
    print(f"Clustering Results: {clusters}")


if __name__ == "__main__":
    main()
