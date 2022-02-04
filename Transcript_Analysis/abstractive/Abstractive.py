from typing import Tuple
from keybert import KeyBERT
from Transcript_Analysis.utterances import Utterance
from transformers import pipeline


class Abstractive:

    keybert_model = None

    classifier = None

    @classmethod
    def get_keybert_keywords(
        cls,
        text: str,
        keyphrase_ngram_range: Tuple,
        n_keyphrases: int
    ):
        print(text)
        if cls.keybert_model == None:
            cls.keybert_model = KeyBERT()
        keyphrase_ngram_range = keyphrase_ngram_range if keyphrase_ngram_range != (
            0, 0) else (1, 3)
        top_n = n_keyphrases if n_keyphrases != None else 3
        keywords = [entity for entity in
                    cls.keybert_model.extract_keywords(
                        text,
                        keyphrase_ngram_range=keyphrase_ngram_range,
                        top_n=top_n
                    )]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        return [keyword[0] for keyword in keywords]

    @classmethod
    def get_bart_summary(
        cls,
        text: str
    ) -> str:
        return Utterance.get_bart_keywords_openai(text)
