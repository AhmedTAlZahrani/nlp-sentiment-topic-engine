"""
Demo script showing how to use the NLP Sentiment & Topic Engine as a library.

Run: python -m notebooks.demo
"""

from nlp_engine.sentiment import SentimentAnalyzer
from nlp_engine.topic_model import TopicModeler


def sentiment_demo():
    print("=" * 60)
    print("SENTIMENT ANALYSIS DEMO")
    print("=" * 60)

    analyzer = SentimentAnalyzer()

    texts = [
        "Apple reported record quarterly revenue driven by strong iPhone sales.",
        "The company laid off 10,000 employees amid declining profits.",
        "Markets remained flat as investors await the Fed decision.",
        "Tesla stock surged 15% after exceeding delivery expectations.",
        "Global supply chain disruptions continue to weigh on growth.",
    ]

    for text in texts:
        result = analyzer.predict(text)
        print(f"\n  Text: {text}")
        print(f"  Label: {result['label']} (confidence: {result['confidence']})")


def topic_demo():
    print("\n" + "=" * 60)
    print("TOPIC MODELING DEMO")
    print("=" * 60)

    documents = [
        "Machine learning models are improving healthcare diagnostics.",
        "Deep learning enables accurate medical image classification.",
        "Neural networks can detect cancer in radiology scans.",
        "Electric vehicles are gaining market share globally.",
        "Tesla and Rivian compete for the EV truck market.",
        "Battery technology advances reduce EV charging times.",
        "Federal reserve raised interest rates by 25 basis points.",
        "Inflation remains above target despite monetary tightening.",
        "Bond yields rose sharply following the rate decision.",
    ]

    modeler = TopicModeler(n_topics=3)
    modeler.fit(documents)

    print("\nDiscovered Topics:")
    for topic in modeler.get_topics(n_words=5):
        print(f"  Topic {topic['id']}: {', '.join(topic['keywords'])}")

    test_text = "Self-driving cars use computer vision for navigation."
    prediction = modeler.predict_topic(test_text)
    print(f"\nTest: '{test_text}'")
    print(f"  Assigned to Topic {prediction['top_topic']}: {', '.join(prediction['keywords'][:5])}")


if __name__ == "__main__":
    sentiment_demo()
    topic_demo()
