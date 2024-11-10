from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

class Clustering:
    def __init__(self, n_clusters=5):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.model = KMeans(n_clusters=n_clusters)

    def fit(self, texts):
        """Metinleri vektörlere dönüştürüp KMeans ile kümeler."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X)
        return self.model.labels_

    def predict(self, texts):
        """Yeni metinleri kümelemek için tahmin yapar."""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)
