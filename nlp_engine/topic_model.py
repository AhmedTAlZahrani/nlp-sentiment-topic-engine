import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from .preprocessor import TextPreprocessor


class TopicModeler:
    """Topic modeling using Latent Dirichlet Allocation (LDA).

    Discovers latent topics from a collection of documents using
    bag-of-words representation and LDA decomposition.
    """

    def __init__(self, n_topics=5, max_iter=20, max_features=5000):
        self.n_topics = n_topics
        self.max_iter = max_iter
        self._preprocessor = TextPreprocessor()
        self._vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words="english",
        )
        self._model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=max_iter,
            random_state=42,
            learning_method="online",
        )
        self._is_fitted = False

    def fit(self, documents):
        """Fit the LDA model on a corpus of documents.

        :param documents: List of raw text strings.
        """
        cleaned = self._preprocessor.clean_batch(documents)
        dtm = self._vectorizer.fit_transform(cleaned)
        self._model.fit(dtm)
        self._is_fitted = True

    def get_topics(self, n_words=10):
        """Extract the top keywords for each discovered topic.

        :param n_words: Number of top words per topic.
        :returns: List of dicts with topic id and keywords.
        :raises RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        feature_names = self._vectorizer.get_feature_names_out()
        topics = []
        for idx, component in enumerate(self._model.components_):
            top_indices = component.argsort()[-n_words:][::-1]
            keywords = [feature_names[i] for i in top_indices]
            topics.append({"id": idx, "keywords": keywords})
        return topics

    def predict_topic(self, text):
        """Predict the dominant topic for a single text.

        :param text: Input text string.
        :returns: Dict with ``top_topic``, ``keywords``, and ``distribution``.
        :raises RuntimeError: If the model has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        cleaned = self._preprocessor.clean(text)
        dtm = self._vectorizer.transform([cleaned])
        distribution = self._model.transform(dtm)[0]

        top_idx = int(np.argmax(distribution))
        topics = self.get_topics()

        return {
            "top_topic": top_idx,
            "keywords": topics[top_idx]["keywords"],
            "distribution": [round(float(p), 4) for p in distribution],
        }
