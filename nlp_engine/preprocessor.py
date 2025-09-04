import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    """Text cleaning and normalization pipeline.

    Applies lowercasing, URL removal, special character removal,
    stopword filtering, and lemmatization.
    """

    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self._stop_words = set(stopwords.words("english")) if remove_stopwords else set()
        self._lemmatizer = WordNetLemmatizer() if lemmatize else None

    def clean(self, text):
        """Clean and normalize a single text string.

        :param text: Raw input text.
        :returns: Cleaned text string.
        """
        text = text.lower()
        text = re.sub(r"http\S+|www\.\S+", "", text)
        text = re.sub(r"[^a-z\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]

        if self._lemmatizer:
            tokens = [self._lemmatizer.lemmatize(t) for t in tokens]

        return " ".join(tokens)

    def clean_batch(self, texts):
        """Clean a list of texts.

        :param texts: List of raw text strings.
        :returns: List of cleaned text strings.
        """
        return [self.clean(t) for t in texts]
