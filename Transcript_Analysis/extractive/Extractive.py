from numpy.linalg import norm
from typing import Any, List
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from Transcript_Analysis.data_types.Transcript import Transcript
from Transcript_Analysis.don_unsupervised_sentence_weighing import *
from fastapi.responses import HTMLResponse
import yake
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize

from Transcript_Analysis.utils.Autocomplete import Meeting_Autocomplete


class Extractive:

    @staticmethod
    def get_related_words(
        transcript: Transcript,
        target_word: str,
        n_keywords: int
    ) -> List:

        n_keywords = 5 if n_keywords == 0 else n_keywords

        autocomplete_obj = Meeting_Autocomplete(transcript=transcript)
        autocomplete_results = autocomplete_obj.search(
            query=target_word, size_of_results=n_keywords)
        print('**' * 10)
        print(autocomplete_results)
        if len(autocomplete_results) == 1:
            target_word = autocomplete_results[0][0]
            print(target_word)
            print('*' * 10)
            return Extractive.get_related_words_cooc_matrix(
                transcript=transcript,
                target_word=target_word,
                n_keywords=n_keywords
            )
        elif len(autocomplete_results) > 1:
            results = [
                word[0] for word in autocomplete_results
            ]
            return results
        else:
            return []

    @staticmethod
    def get_related_words_cooc_matrix(
        transcript: Transcript,
        target_word: str,
        n_keywords: int
    ) -> np.array:
        """
        Get the co-occurrence matrix
        """

        def process_sentence(sentence, window_size=5):
            words_in_sentence = Utils.sentence_to_wordlist(sentence)
            list_of_indeces = [list_of_words.index(
                word) for word in words_in_sentence]
            for i, index1 in enumerate(list_of_indeces):
                for j, index2 in enumerate(list_of_indeces):
                    if index1 != index2 and abs(i - j) <= window_size:
                        cooc[index1, index2] += 1

        text = '.'.join([turn.text for turn in transcript.turns])
        stop_words = Utils.load_stop_words()
        text = ' '.join([word for word in text.split()
                        if word not in stop_words])

        list_of_words = list(set(Utils.sentence_to_wordlist(text)))

        list_of_sentences = sent_tokenize(text)

        cooc = np.zeros([len(list_of_words), len(list_of_words)], np.float64)
        for sentence in list_of_sentences:
            process_sentence(sentence)

        def cos_dis(u, v):
            dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
            return dist

        sorted_list = sorted(list_of_words, key=lambda word: cos_dis(
            cooc[list_of_words.index(target_word), :], cooc[list_of_words.index(word), :]))
        return sorted_list[0:n_keywords + 1]

    @staticmethod
    def get_rake_keywords(
        text: str,
        top_n: int = 10
    ) -> List[Any]:
        """
        get the keywords using the Rake algorithm
        """
        r = Rake()

        r.extract_keywords_from_text(text)
        return r.get_ranked_phrases()[:top_n]

    @staticmethod
    def get_yake_keywords(
        text: str,
        language="en",
        max_ngram_size=3,
        deduplication_thresold=0.9,
        deduplication_algo='seqm',
        windowSize=1,
        numOfKeywords=10,
    ) -> List[Any]:
        """
        Get the YAKE keywords based on https://github.com/LIAAD/yake
        """
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_thresold,
                                                    dedupFunc=deduplication_algo, windowsSize=windowSize, top=numOfKeywords, features=None)
        keywords = custom_kw_extractor.extract_keywords(text)
        keywords = [keyword[0] for keyword in keywords]
        return keywords

    @staticmethod
    def get_tf_idf_keywords(
        documents: List[str],
        ngram_range: Tuple,
        num_keywords: int
    ) -> np.array:
        ngram_range = ngram_range if ngram_range != (0, 0) else (1, 3)
        num_keywords = num_keywords if num_keywords != None else 2

        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range)
        X = vectorizer.fit_transform(documents)
        df = pd.DataFrame(X.A, columns=vectorizer.get_feature_names())
        keywords = []
        columns = df.columns
        for _, row in df.iterrows():
            indices = np.argsort(row)
            keywords.append(list(np.take(columns, indices[:num_keywords])))
        return keywords

    @staticmethod
    def get_sentence_weights(
        transcript: Transcript
    ) -> List[dict]:
        dt = DialogueTranscript(
            csv_file=None,
            source_dataframe=transcript.df
        )
        dt.filter_backchannels()
        dt.count_keywords()
        dt.weigh_sentences()
        output_json = []
        for sentence, weight in zip(dt.sentences, dt.sentence_weights):
            output_json.append({
                'content': sentence,
                'weight': weight
            })
        return output_json

    @staticmethod
    def get_frequent_keywords(
        transcript: Transcript
    ) -> List[dict]:
        dt = DialogueTranscript(
            csv_file=None,
            source_dataframe=transcript.df
        )
        dt.filter_backchannels()
        dt.count_keywords()
        sorted_keywords = dict(
            sorted(dt.keyword_counts.items(), key=lambda x: x[1], reverse=True))
        output_json = []
        for keyword, score in sorted_keywords.items():
            output_json.append({
                'content': keyword,
                'weight': score
            })
        return output_json

    @staticmethod
    def get_louvain_topics_sentences(
        transcript: Transcript
    ) -> HTMLResponse:

        dt = DialogueTranscript(
            csv_file=None,
            source_dataframe=transcript.df
        )

        dt.filter_backchannels()
        dt.count_keywords()
        dt.weigh_sentences()
        dt.vectorize_sentences_with_keywords()
        dt.calculate_sentence_similarity()
        dt.simple_entailment()
        dt.construct_sentences_graph()

        dt.simple_entailment()
        dt.remove_nodes_based_on_entailment_matrix_and_similarity_graph()

        dt.sentences_graph = Utils.get_graph_backbone(
            dt.sentences_graph)
        dt.load_sentence_weights_to_graph()
        dt.cluster_sentences_by_louvain(dt.sentences_graph)
        # whether to do community detection before entailment process and removing the nodes or after it

        html_output = dt.write_dataframe_with_weight_community_html()
        return HTMLResponse(content=html_output, status_code=200)

    @staticmethod
    def get_statistics(transcript: Transcript) -> Any:
        """
        Get certain statistics regarding the speakers and the whole utterances
        """
        num_speakers = len(set([turn.speaker_id for turn in transcript.turns]))
        num_utterances = len(transcript)
        return {
            'num_speakers': num_speakers,
            'num_utterances': num_utterances
        }


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    print(Extractive.get_tf_idf_keywords(corpus))
