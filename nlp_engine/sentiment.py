import logging

from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analyzer using a pre-trained DistilBERT model.

    Uses the HuggingFace ``distilbert-base-uncased-finetuned-sst-2-english``
    checkpoint for binary sentiment classification.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        logger.info("Loading model: %s", self.MODEL_NAME)
        try:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.MODEL_NAME,
                return_all_scores=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load model '{self.MODEL_NAME}'. Ensure you have "
                f"an internet connection or the model is cached locally. "
                f"Original error: {exc}"
            ) from exc
        logger.info("Model loaded")

    def predict(self, text):
        """Analyze sentiment of a single text.

        :param text: Input text string.
        :returns: Dict with keys ``label``, ``confidence``, ``scores``.
        """
        results = self._pipeline(text)[0]
        scores = {r["label"]: round(r["score"], 4) for r in results}
        top = max(results, key=lambda r: r["score"])
        return {
            "label": top["label"],
            "confidence": round(top["score"], 4),
            "scores": scores,
        }

    def predict_batch(self, texts):
        """Analyze sentiment for a batch of texts.

        :param texts: List of input text strings.
        :returns: List of result dicts.
        """
        logger.debug("Batch prediction on %d texts", len(texts))
        batch_results = self._pipeline(texts)
        output = []
        for results in batch_results:
            scores = {r["label"]: round(r["score"], 4) for r in results}
            top = max(results, key=lambda r: r["score"])
            output.append({
                "label": top["label"],
                "confidence": round(top["score"], 4),
                "scores": scores,
            })
        return output
