# CLAUDE.md

Guidance for Claude Code (and other AI assistants) working in this repository.

## What this is

`nlp-sentiment-topic-engine` — a small Python project that pairs two NLP tasks
behind one FastAPI service:

- **Sentiment analysis** via the pre-trained HuggingFace checkpoint
  `distilbert-base-uncased-finetuned-sst-2-english` (inference only, no training).
- **Topic modeling** via scikit-learn `LatentDirichletAllocation` over a
  `CountVectorizer` bag-of-words matrix (fitted per request, nothing persisted).

There is no training pipeline, no dataset in the repo, and no model artifact
checked in. The DistilBERT weights are downloaded from the HuggingFace Hub on
first use and cached by `transformers` in the usual `~/.cache/huggingface`
location.

## Layout

```
nlp_engine/            # the library — no FastAPI/web imports belong here
  preprocessor.py      # TextPreprocessor: clean() / clean_batch()
  sentiment.py         # SentimentAnalyzer: predict() / predict_batch()
  topic_model.py       # TopicModeler: fit() / get_topics() / predict_topic()
api/
  main.py              # FastAPI app: all endpoints and Pydantic models
tests/
  test_sentiment.py    # the entire suite — sentiment, topics, preprocessing
notebooks/
  demo.py              # runnable library demo (a .py, not a notebook)
.github/workflows/ci.yml
setup.sh               # pip install + NLTK corpora download
requirements.txt
```

The dependency direction is one-way: `api/` imports `nlp_engine/`, never the
reverse. `nlp_engine/topic_model.py` imports `preprocessor` relatively
(`from .preprocessor import TextPreprocessor`); `api/main.py` imports the
library absolutely (`from nlp_engine.sentiment import SentimentAnalyzer`).
Keep it that way — the library must stay usable without FastAPI installed.

## Setup

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

`./setup.sh` does both. The NLTK download is **not optional** for real use:
`TextPreprocessor.__init__` calls `stopwords.words("english")` eagerly, so
importing `nlp_engine.topic_model` and constructing anything will fail with an
NLTK `LookupError` if the corpora are missing.

Run the API:

```bash
uvicorn api.main:app --reload --port 8000   # docs at /docs
```

Run the library demo:

```bash
python -m notebooks.demo
```

## Tests

```bash
PYTHONPATH=. python -m pytest tests/ -v
```

Things to know before touching the suite:

- **There is no `pyproject.toml`, `setup.py`, or `conftest.py`.** The package is
  never installed; imports resolve purely from the repo root being on
  `sys.path`. That is why CI sets `PYTHONPATH: .` and why `python -m pytest`
  (which prepends the cwd) works while a bare `pytest` from elsewhere does not.
  If you add packaging or a `conftest.py`, the `PYTHONPATH` in `ci.yml` becomes
  redundant — remove it in the same change rather than leaving both.
- **The suite runs with `transformers`, `torch`, and `nltk` absent.**
  `tests/test_sentiment.py` installs `MagicMock`s into `sys.modules` for
  `nltk`, `nltk.corpus`, and `nltk.stem` *before* importing `nlp_engine`, and
  `sys.modules.setdefault("transformers", ...)` for the model library. The
  imports at lines 45–56 are deliberately below that setup and carry `# noqa:
  E402`. Do not "clean up" the import order — it will break collection.
- The `analyzer` fixture patches `nlp_engine.sentiment.pipeline` with a fake
  that returns `POSITIVE` unless the text contains `"bad"` or `"terrible"`.
  New sentiment assertions must respect that rule, not real model behavior.
- `TopicModeler` is exercised for real (scikit-learn is a genuine dependency in
  tests) on a small synthetic corpus, with `random_state=42` making it
  deterministic.
- **`api/main.py` has no test coverage.** If you change endpoint behavior, add
  tests using `fastapi.testclient.TestClient` and override the analyzer with
  `app.dependency_overrides[get_analyzer] = lambda: fake` rather than loading
  the real model.

## CI

`.github/workflows/ci.yml` runs on every push and pull request: Python 3.11,
`pip install -r requirements.txt`, `pip install pytest`, then `pytest tests/ -x`
with `PYTHONPATH: .`. Note that CI installs the full requirements (including
`torch`), so it is slow; the tests themselves do not need those packages.
`pytest` is not in `requirements.txt` — it is installed separately in the
workflow. Keep it that way, or add a dev-requirements file rather than putting
test tooling into the runtime deps.

## API surface

| Method | Path                | Loads DistilBERT | Fits LDA |
|--------|---------------------|------------------|----------|
| GET    | `/ready`            | no               | no       |
| POST   | `/sentiment`        | yes (lazy)       | no       |
| POST   | `/sentiment/batch`  | yes (lazy)       | no       |
| POST   | `/topics`           | no               | yes      |
| POST   | `/analyze`          | yes (lazy)       | only if `corpus` is non-empty |

Request/response models are the four Pydantic classes at the top of
`api/main.py` (`TextRequest`, `BatchTextRequest`, `TopicRequest`,
`AnalyzeRequest`). Add new endpoints with an explicit model rather than raw
dicts, and keep the request/response block separated by the existing
`# -- Section ---` comment banners.

### Model loading

`AnalyzerDep` in `api/main.py` is a callable class used as a FastAPI
dependency. It holds one `SentimentAnalyzer` and constructs it on the first
request that needs it, so app startup stays fast and tests can override it.
The module-level `get_analyzer = AnalyzerDep()` instance is the singleton —
process-wide, not per-request. Do not move model loading into a startup event
or import-time constructor; that breaks both the fast-start and the
test-override properties.

`TopicModeler`, by contrast, is **constructed and fitted fresh on every
`/topics` and `/analyze` request**. This is intentional (LDA is fitted on the
caller's corpus, not a global one), but it means those endpoints are O(corpus)
per call and stateless across calls. If you add caching, make the cache key
include the corpus *and* `n_topics`.

## Conventions

- **Docstrings** use reStructuredText field syntax (`:param x:`, `:returns:`,
  `:raises:`) on every public method and endpoint. Match it.
- **Private state** is single-underscore-prefixed instance attributes
  (`self._pipeline`, `self._model`, `self._vectorizer`, `self._is_fitted`).
  Constructor arguments that are part of the public contract stay public
  (`self.n_topics`, `self.remove_stopwords`).
- **Not-fitted guards**: `TopicModeler.get_topics` and `predict_topic` raise
  `RuntimeError("Model not fitted. Call fit() first.")`. Tests match on
  `"not fitted"` — keep that substring if you reword the message.
- **Logging**: `sentiment.py` uses a module `logger` (`logging.getLogger(__name__)`)
  with `logging.basicConfig(level=logging.INFO)` at import. Use `logger`, not
  `print()`, in library code. `notebooks/demo.py` prints because it is a script.
- **Rounding**: scores and distributions are rounded to 4 decimal places at the
  boundary (`round(x, 4)`). Keep full precision internally.
- **Imports** are grouped stdlib / third-party / local with blank lines between.
- No formatter or linter is configured (no black/ruff/flake8 config, no
  pre-commit). Match the surrounding style by hand; do not reformat files
  wholesale as a side effect of another change.

## Behavioral gotchas

These are real properties of the current code, not bugs to fix drive-by — know
them before you debug something surprising:

- **`TextPreprocessor.clean` deletes digits.** The regex `[^a-z\s]` runs after
  lowercasing, so `"Q3 2024 revenue up 15%"` becomes `"q revenue up"`. Any
  feature that needs numbers must preprocess differently.
- **Stopwords are removed twice** in the topic path: once by
  `TextPreprocessor` (NLTK list) and again by `CountVectorizer(stop_words="english")`
  (scikit-learn list). Harmless but redundant; if you tune one, check the other.
- **`SentimentAnalyzer` passes `return_all_scores=True`** to
  `transformers.pipeline`. That argument is deprecated in modern `transformers`
  in favor of `top_k=None` and emits a warning. If you migrate it, the return
  shape must stay a list-of-lists-of-dicts or `predict`/`predict_batch` and the
  test fakes both break.
- **`predict` indexes `self._pipeline(text)[0]`** — it relies on the pipeline
  wrapping a single string in a one-element batch. `predict_batch` iterates
  instead. Don't unify them without checking both shapes.
- **`api/main.py` has unused imports**: `pipeline` from `transformers` and `re`
  are imported but never referenced. `/ready` also carries a `# TODO: switch to
  async endpoint`. Safe to clean up if you are already editing the file.
- **No input validation or size limits.** `/sentiment/batch` and `/topics`
  accept arbitrarily long lists, and `/topics` will happily try to fit LDA with
  more topics than documents. Add validation deliberately, with tests.

## Git workflow

- Default branch: `main`. Remote: `https://github.com/AhmedTAlZahrani/nlp-sentiment-topic-engine`.
- Commit messages in this repo are short, lowercase-ish, imperative summaries
  ("fix readme nltk downloads and document running tests", "Add sentiment tests
  with mocked models, CI workflow, and setup script"). No Conventional Commits
  prefix convention is in force.
- Work on a feature branch and push with `git push -u origin <branch>`.
- Do not commit model weights, `model_cache/`, `.pt` files, or virtualenvs —
  `.gitignore` already covers them.
- Only open a pull request when explicitly asked.
