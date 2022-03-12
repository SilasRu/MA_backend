from numpy.linalg import norm
from typing import Any, DefaultDict, List
import numpy as np

from Transcript_Analysis.data_types.Transcript import Transcript
from Transcript_Analysis.custom_unsupervised_summarizer import *
import yake
from rake_nltk import Rake
from nltk.tokenize import sent_tokenize
from Transcript_Analysis.extractive.lsa.lsa_summarizer import LsaSummarizer

from Transcript_Analysis.utils.Autocomplete import Meeting_Autocomplete


class Extractive:

    @staticmethod
    def get_lsa_sentences(
        text: str,
        n_keyphrases: int
    ) -> str:

        summarizer = LsaSummarizer()

        stopwords = Utils.load_stop_words()
        summarizer.stop_words = stopwords
        summary = summarizer(text, n_keyphrases)

        return " ".join(summary)

    @staticmethod
    def get_related_words(
        text: str,
        target_word: str,
        n_keyphrases: int
    ) -> List:
        autocomplete_obj = Meeting_Autocomplete(text=text)
        autocomplete_results = autocomplete_obj.search(
            query=target_word, size_of_results=n_keyphrases)

        if len(autocomplete_results) == 1:
            target_word = autocomplete_results[0][0]

            return Extractive.get_related_words_cooc_matrix(
                text=text,
                target_word=target_word,
                n_keyphrases=n_keyphrases
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
        text: str,
        target_word: str,
        n_keyphrases: int
    ) -> np.array:
        """
        Get the co-occurrence matrix
        """

        def process_sentence(sentence, window_size=5):
            words_in_sentence = Utils.sentence_to_wordlist(sentence)
            list_of_indeces = [vocab.index(
                word) for word in words_in_sentence]
            for i, index1 in enumerate(list_of_indeces):
                for j, index2 in enumerate(list_of_indeces):
                    if index1 != index2 and abs(i - j) <= window_size:
                        cooc[index1, index2] += 1

        stop_words = Utils.load_stop_words()
        text = ' '.join([word for word in text.split()
                        if word not in stop_words])

        splited_text = Utils.sentence_to_wordlist(text)
        vocab = list(set(splited_text))
        text = ' '.join(splited_text)

        list_of_sentences = sent_tokenize(text)

        cooc = np.zeros([len(vocab), len(vocab)], np.float64)
        for sentence in list_of_sentences:
            process_sentence(sentence)

        def cos_dis(u, v):
            dist = 1.0 - np.dot(u, v) / (norm(u) * norm(v))
            return dist

        sorted_list = sorted(vocab, key=lambda word: cos_dis(
            cooc[vocab.index(target_word), :], cooc[vocab.index(word), :]))
        return sorted_list[0:n_keyphrases + 1]

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
    def get_sentence_properties(
        transcript: Transcript or str,
        output_type: Output_type,
        filter_backchannels: bool = True,
        remove_entailed_sentences: bool = True,
        get_graph_backbone: bool = True,
        do_cluster: bool = True,
        clustering_algorithm: str = 'louvain',
        per_cluster_results: bool = True,
    ) -> List[dict]:
        if isinstance(transcript, Transcript):
            source_dataframe = transcript.df
        elif isinstance(transcript, str):
            source_dataframe = Utils.text2df(transcript)
        else:
            return 'Instance provided by user is not supported!'
        dt = Unsupervised_Summarizer(
            csv_file=None,
            source_dataframe=source_dataframe
        )
        results = dt(
            output=output_type,
            filter_backchannels=filter_backchannels,
            remove_entailed_sentences=remove_entailed_sentences,
            get_graph_backbone=get_graph_backbone,
            do_cluster=do_cluster,
            clustering_algorithm=clustering_algorithm,
            per_cluster_results=per_cluster_results,
        )
        if output_type == Output_type.WORD:
            results = [
                {
                    'content': word,
                    'weight': cnt,
                }
                for word, cnt in results.items()
            ]
        elif output_type == Output_type.SENTENCE:
            if per_cluster_results:
                return results
            results = [
                {
                    'content': sentence,
                    'weight': properties['weight'],
                    'properties': {
                        key: value
                        for key, value in properties.items() if key != 'weight'
                    }
                }
                for sentence, properties in results.items()
            ]
        else:
            raise Exception(
                'The output type you requested is not being supported!')
        return Utils.sort_json_by_property(results, "weight")

    @staticmethod
    def get_statistics(transcript: Transcript) -> Any:
        """
        Get certain statistics regarding the speakers and the whole utterances
        """

        def get_time_for_all_speakers() -> float:
            speakers_total_times = DefaultDict(lambda: 0)
            for turn in transcript.turns:
                speakers_total_times[turn.speaker_id] += turn.end_time - \
                    turn.start_time
            return speakers_total_times

        num_speakers = len(set([turn.speaker_id for turn in transcript.turns]))
        num_utterances = len(transcript)
        speakers_total_times = get_time_for_all_speakers()
        return {
            'num_speakers': num_speakers,
            'num_utterances': num_utterances,
            'speaker_times': speakers_total_times
        }


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    print(Extractive.get_tf_idf_keywords(corpus))
