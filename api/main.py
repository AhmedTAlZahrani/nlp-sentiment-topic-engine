from fastapi import FastAPI, Depends
from pydantic import BaseModel
from transformers import pipeline
import re

from nlp_engine.sentiment import SentimentAnalyzer
from nlp_engine.topic_model import TopicModeler

app = FastAPI(
    title="NLP Sentiment & Topic Engine",
    description="Multi-task NLP API for sentiment analysis and topic modeling.",
    version="1.0.0",
)


class AnalyzerDep:
    """Dependency injection container for the sentiment analyzer.

    Lazily initializes the model on first request so the app
    starts quickly and tests can override the instance.
    """

    def __init__(self):
        self._analyzer = None

    def __call__(self):
        if self._analyzer is None:
            self._analyzer = SentimentAnalyzer()
        return self._analyzer


get_analyzer = AnalyzerDep()


# -- Request / Response Models ----------------------------------------

class TextRequest(BaseModel):
    text: str

class BatchTextRequest(BaseModel):
    texts: list[str]

class TopicRequest(BaseModel):
    texts: list[str]
    n_topics: int = 5
    n_words: int = 10

class AnalyzeRequest(BaseModel):
    text: str
    corpus: list[str] = []
    n_topics: int = 5


# -- Endpoints ---------------------------------------------------------

@app.get("/ready")
def readiness_check():
    # TODO: switch to async endpoint
    return {"status": "ok"}


@app.post("/sentiment")
def sentiment(request: TextRequest, analyzer: SentimentAnalyzer = Depends(get_analyzer)):
    """Analyze sentiment of a single text.

    :param request: JSON body with ``text`` field.
    :returns: Sentiment label, confidence, and per-class scores.
    """
    result = analyzer.predict(request.text)
    return {"text": request.text, **result}


@app.post("/sentiment/batch")
def sentiment_batch(request: BatchTextRequest, analyzer: SentimentAnalyzer = Depends(get_analyzer)):
    """Analyze sentiment for multiple texts.

    :param request: JSON body with ``texts`` list.
    :returns: List of sentiment results.
    """
    results = analyzer.predict_batch(request.texts)
    return {
        "count": len(results),
        "results": [
            {"text": text, **result}
            for text, result in zip(request.texts, results)
        ],
    }


@app.post("/topics")
def topics(request: TopicRequest):
    """Fit LDA on a corpus and return discovered topics.

    :param request: JSON body with ``texts``, ``n_topics``, ``n_words``.
    :returns: Discovered topics with keywords.
    """
    modeler = TopicModeler(n_topics=request.n_topics)
    modeler.fit(request.texts)
    discovered = modeler.get_topics(n_words=request.n_words)
    return {"n_topics": request.n_topics, "topics": discovered}


@app.post("/analyze")
def analyze(request: AnalyzeRequest, analyzer: SentimentAnalyzer = Depends(get_analyzer)):
    """Combined sentiment analysis and topic prediction.

    :param request: JSON body with ``text``, optional ``corpus``, ``n_topics``.
    :returns: Sentiment and topic results.
    """
    sentiment_result = analyzer.predict(request.text)

    topic_result = None
    if request.corpus:
        modeler = TopicModeler(n_topics=request.n_topics)
        modeler.fit(request.corpus)
        topic_result = modeler.predict_topic(request.text)

    return {
        "text": request.text,
        "sentiment": sentiment_result,
        "topic": topic_result,
    }
