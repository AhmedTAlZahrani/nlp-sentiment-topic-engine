from transformers import pipeline


class SentimentAnalyzer:
    """Sentiment analyzer using a pre-trained DistilBERT model.

    Uses the HuggingFace ``distilbert-base-uncased-finetuned-sst-2-english``
    checkpoint for binary sentiment classification.
    """

    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self):
        self._pipeline = pipeline(
            "sentiment-analysis",
            model=self.MODEL_NAME,
            return_all_scores=True,
        )

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
