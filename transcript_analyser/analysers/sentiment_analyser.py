from typing import List, Optional
from transformers import pipeline
from nltk.tokenize import sent_tokenize

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline(task='text-classification', model=checkpoint)


def get_sentiments(text: str, dimension: Optional[str]) -> List[dict]:
    sentences = sent_tokenize(text)
    results = classifier(sentences)

    if dimension is None:
        return [{'content': sent, **result} for sent, result in list(zip(sentences, results))]
    else:
        dimensions = dimension.split(',')
