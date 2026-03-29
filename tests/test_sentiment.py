"""Tests for sentiment analysis, topic extraction, and text preprocessing."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers — mock NLTK and transformers before any nlp_engine imports
# ---------------------------------------------------------------------------

def _install_nltk_mocks():
    """Patch nltk.corpus.stopwords and nltk.stem so imports succeed without data.

    :returns: Tuple of (mock_stopwords_module, mock_lemmatizer_class).
    """
    mock_stopwords = MagicMock()
    mock_stopwords.words.return_value = ["the", "a", "is", "in", "and", "of", "to", "it"]

    mock_lemmatizer_cls = MagicMock()
    inst = MagicMock()
    inst.lemmatize.side_effect = lambda w: w
    mock_lemmatizer_cls.return_value = inst

    # Ensure nltk sub-modules exist in sys.modules
    if "nltk" not in sys.modules:
        sys.modules["nltk"] = MagicMock()
    if "nltk.corpus" not in sys.modules:
        sys.modules["nltk.corpus"] = MagicMock()
    if "nltk.stem" not in sys.modules:
        sys.modules["nltk.stem"] = MagicMock()

    sys.modules["nltk.corpus"].stopwords = mock_stopwords
    sys.modules["nltk.stem"].WordNetLemmatizer = mock_lemmatizer_cls

    return mock_stopwords, mock_lemmatizer_cls


# Install NLTK mocks before importing nlp_engine modules
_install_nltk_mocks()

# Now safe to import preprocessor and topic_model
from nlp_engine.preprocessor import TextPreprocessor  # noqa: E402
from nlp_engine.topic_model import TopicModeler  # noqa: E402


# ---------------------------------------------------------------------------
# Sentiment Analyzer — mock the transformers pipeline
# ---------------------------------------------------------------------------

# Import sentiment module with transformers mocked
_mock_transformers = MagicMock()
sys.modules.setdefault("transformers", _mock_transformers)
from nlp_engine import sentiment as _sentiment_mod  # noqa: E402


def _mock_pipeline_factory(*args, **kwargs):
    """Build a fake HuggingFace pipeline callable.

    :returns: A callable that mimics ``transformers.pipeline`` output.
    """
    def _fake_pipeline(texts):
        single = isinstance(texts, str)
        if single:
            texts = [texts]
        batch = []
        for t in texts:
            if "bad" in t.lower() or "terrible" in t.lower():
                batch.append([
                    {"label": "NEGATIVE", "score": 0.9532},
                    {"label": "POSITIVE", "score": 0.0468},
                ])
            else:
                batch.append([
                    {"label": "NEGATIVE", "score": 0.0231},
                    {"label": "POSITIVE", "score": 0.9769},
                ])
        return batch if not single else batch
    return _fake_pipeline


@pytest.fixture()
def analyzer():
    """Create a ``SentimentAnalyzer`` with a mocked transformer pipeline.

    :returns: An instance of ``SentimentAnalyzer`` backed by a fake model.
    """
    with patch("nlp_engine.sentiment.pipeline", side_effect=_mock_pipeline_factory):
        from nlp_engine.sentiment import SentimentAnalyzer
        sa = SentimentAnalyzer()
    return sa


class TestSentimentPredict:
    """Tests for :meth:`SentimentAnalyzer.predict`."""

    @pytest.mark.parametrize("text,expected_label", [
        ("I love this product, it is amazing!", "POSITIVE"),
        ("This is a bad experience, terrible service", "NEGATIVE"),
        ("Great weather today", "POSITIVE"),
        ("The food was bad and cold", "NEGATIVE"),
    ])
    def test_predict_label(self, analyzer, text, expected_label):
        """Verify predicted label matches expected sentiment.

        :param analyzer: Mocked sentiment analyzer fixture.
        :param text: Input text to classify.
        :param expected_label: Expected sentiment label.
        """
        result = analyzer.predict(text)
        assert result["label"] == expected_label

    @pytest.mark.parametrize("text", [
        "Absolutely wonderful day",
        "This is terrible and awful",
    ])
    def test_predict_structure(self, analyzer, text):
        """Ensure predict returns the expected dict keys.

        :param analyzer: Mocked sentiment analyzer fixture.
        :param text: Input text to classify.
        """
        result = analyzer.predict(text)
        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert isinstance(result["scores"], dict)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_scores_contain_both_labels(self, analyzer):
        """Check that scores dict has both POSITIVE and NEGATIVE entries."""
        result = analyzer.predict("neutral text here")
        assert "POSITIVE" in result["scores"]
        assert "NEGATIVE" in result["scores"]

    def test_predict_confidence_matches_top_score(self, analyzer):
        """Confidence should equal the score of the winning label."""
        result = analyzer.predict("I love it")
        assert result["confidence"] == result["scores"][result["label"]]


class TestSentimentBatch:
    """Tests for :meth:`SentimentAnalyzer.predict_batch`."""

    @pytest.mark.parametrize("texts,expected_labels", [
        (
            ["good stuff", "bad stuff"],
            ["POSITIVE", "NEGATIVE"],
        ),
        (
            ["wonderful", "great", "terrible"],
            ["POSITIVE", "POSITIVE", "NEGATIVE"],
        ),
    ])
    def test_batch_labels(self, analyzer, texts, expected_labels):
        """Verify batch prediction returns correct labels per text.

        :param analyzer: Mocked sentiment analyzer fixture.
        :param texts: List of input texts.
        :param expected_labels: Expected labels in order.
        """
        results = analyzer.predict_batch(texts)
        assert len(results) == len(texts)
        for res, expected in zip(results, expected_labels):
            assert res["label"] == expected

    def test_batch_returns_list(self, analyzer):
        """Batch predict should always return a list."""
        results = analyzer.predict_batch(["hello"])
        assert isinstance(results, list)
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Topic Modeler
# ---------------------------------------------------------------------------

@pytest.fixture()
def topic_modeler():
    """Create a :class:`TopicModeler` fitted on a small synthetic corpus.

    :returns: A fitted ``TopicModeler`` instance with 2 topics.
    """
    tm = TopicModeler(n_topics=2, max_iter=5, max_features=500)
    corpus = [
        "machine learning algorithms improve prediction accuracy",
        "deep learning neural networks train on large datasets",
        "supervised learning classification regression models",
        "stock market trading prices rise fall economy",
        "financial portfolio investment returns quarterly",
        "economic growth inflation rates central bank policy",
    ] * 3
    tm.fit(corpus)
    return tm


class TestTopicModeler:
    """Tests for :class:`TopicModeler` fit, get_topics, and predict_topic."""

    def test_get_topics_returns_list(self, topic_modeler):
        """get_topics should return a list of topic dicts."""
        topics = topic_modeler.get_topics(n_words=5)
        assert isinstance(topics, list)
        assert len(topics) == 2

    def test_get_topics_structure(self, topic_modeler):
        """Each topic dict should have ``id`` and ``keywords`` keys."""
        topics = topic_modeler.get_topics(n_words=5)
        for t in topics:
            assert "id" in t
            assert "keywords" in t
            assert isinstance(t["keywords"], list)
            assert len(t["keywords"]) == 5

    def test_get_topics_not_fitted_raises(self):
        """Calling get_topics before fit should raise RuntimeError."""
        tm = TopicModeler(n_topics=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            tm.get_topics()

    @pytest.mark.parametrize("text", [
        "neural network deep learning model training",
        "stock market economy financial growth",
    ])
    def test_predict_topic_structure(self, topic_modeler, text):
        """predict_topic should return top_topic, keywords, distribution.

        :param topic_modeler: Fitted topic modeler fixture.
        :param text: Input text to assign a topic.
        """
        result = topic_modeler.predict_topic(text)
        assert "top_topic" in result
        assert "keywords" in result
        assert "distribution" in result
        assert isinstance(result["top_topic"], int)
        assert 0 <= result["top_topic"] < 2
        assert len(result["distribution"]) == 2

    def test_predict_topic_distribution_sums_near_one(self, topic_modeler):
        """Topic distribution should approximately sum to 1."""
        result = topic_modeler.predict_topic("learning algorithms data")
        total = sum(result["distribution"])
        assert abs(total - 1.0) < 0.05

    def test_predict_topic_not_fitted_raises(self):
        """Calling predict_topic before fit should raise RuntimeError."""
        tm = TopicModeler(n_topics=2)
        with pytest.raises(RuntimeError, match="not fitted"):
            tm.predict_topic("some text")


# ---------------------------------------------------------------------------
# Text Preprocessor
# ---------------------------------------------------------------------------

class TestPreprocessor:
    """Tests for :class:`TextPreprocessor`."""

    @pytest.fixture()
    def preprocessor(self):
        """Create a ``TextPreprocessor`` instance.

        :returns: A ``TextPreprocessor`` using mocked NLTK resources.
        """
        return TextPreprocessor()

    @pytest.mark.parametrize("raw,expected_substring", [
        ("Hello WORLD", "hello world"),
        ("Visit http://example.com now", "visit now"),
        ("Price: $100!!!", "price"),
    ])
    def test_clean_normalizes(self, preprocessor, raw, expected_substring):
        """Verify basic cleaning operations like lowering and URL removal.

        :param preprocessor: Preprocessor fixture.
        :param raw: Raw input text.
        :param expected_substring: Substring expected in cleaned output.
        """
        cleaned = preprocessor.clean(raw)
        assert expected_substring in cleaned

    def test_clean_removes_stopwords(self, preprocessor):
        """Stopwords configured in the mock should be removed."""
        cleaned = preprocessor.clean("the cat is in a box")
        assert "the" not in cleaned.split()
        assert "is" not in cleaned.split()
        assert "cat" in cleaned.split()

    def test_clean_batch(self, preprocessor):
        """clean_batch should return one result per input document."""
        results = preprocessor.clean_batch(["Hello!", "World!"])
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
