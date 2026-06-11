# NLP Sentiment & Topic Engine

![CI](https://github.com/AhmedTAlZahrani/nlp-sentiment-topic-engine/actions/workflows/ci.yml/badge.svg)

Sentiment analysis (DistilBERT) and topic modeling (LDA) exposed via FastAPI.

## Install

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

Or run `./setup.sh` which does both steps.

## Run

```bash
uvicorn api.main:app --reload --port 8000
```

Docs at `http://localhost:8000/docs`.

## Endpoints

- `POST /sentiment` -- single text sentiment
- `POST /sentiment/batch` -- batch sentiment
- `POST /topics` -- fit LDA and return topics
- `POST /analyze` -- combined sentiment + topic
- `GET /ready` -- readiness check

## Library usage

```python
from nlp_engine.sentiment import SentimentAnalyzer
from nlp_engine.topic_model import TopicModeler

analyzer = SentimentAnalyzer()
print(analyzer.predict("Revenue exceeded expectations"))

modeler = TopicModeler(n_topics=3)
modeler.fit(corpus)
print(modeler.get_topics())
```

## Tests

```bash
pytest tests/ -v
```

## Structure

```
nlp_engine/
  preprocessor.py
  sentiment.py
  topic_model.py
api/
  main.py
notebooks/
  demo.py
```

MIT License
