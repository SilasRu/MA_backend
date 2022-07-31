from collections import OrderedDict

from numpy.linalg import norm
from typing import Any, DefaultDict, List
import numpy as np
from transcript_analyser.consts import TYPE_NOT_SUPPORTED
from transformers import pipeline
import nltk
from transcript_analyser.data_types.transcript import Transcript
from transcript_analyser.custom_unsupervised_summarizer import *
import yake
from rake_nltk import Rake
from Levenshtein import ratio as similarity_ratio
from nltk.tokenize import sent_tokenize
from transcript_analyser.extractive.lsa.lsa_summarizer import LsaSummarizer

from transcript_analyser.utils.autocomplete import Meeting_Autocomplete


class Extractive:

    @staticmethod
    def get_entities(transcript, section_length) -> Dict[str, Dict[str, Any]]:
        ner = pipeline("ner", aggregation_strategy='average')
        nltk_tokenizer = nltk.RegexpTokenizer(r"\w+")
        turns_segmented_by_time = [
            (f' ', text)
            for speaker_id, text in
            zip([turn.speaker_id for turn in transcript.turns], [turn.text for turn in transcript.turns])
        ]
        turns_segmented_by_speaker = OrderedDict()
        for speaker_id in transcript.speaker_info.keys():
            turns_segmented_by_speaker[speaker_id] = [
                (f' ', turn.text) for turn in transcript.turns if
                turn.speaker_id == speaker_id
            ]
        time_sections_to_process = Utils.get_sections_from_texts(turns_segmented_by_time, section_length)
        speaker_sections_to_process = [''.join(Utils.get_sections_from_texts(speaker, section_length)) for speaker in
                                       turns_segmented_by_speaker.values()]

        def get_entities_for_dimension(sections):
            dimension = {}
            speakers = [word.lower() for word in transcript.speaker_info.values()]
            for i, utterance in enumerate(sections):
                dimension[i] = []
                entities = ner(utterance)
                tokenized_utterance = nltk_tokenizer.tokenize(utterance)

                for entity in entities:
                    in_speakers = entity['word'].lower() in speakers
                    occurrence = len([word for word in tokenized_utterance if word == entity['word']])
                    dimension[i].append(
                        {'in_speakers': in_speakers, 'word': entity['word'], 'entity_group': entity['entity_group'],
                         'occurrence': occurrence})
                dimension[i] = list({v['word']: v for v in dimension[i]}.values())
            return dimension

        dimensions = {
            'time': get_entities_for_dimension(time_sections_to_process),
            'speaker': get_entities_for_dimension(speaker_sections_to_process)
        }
        flat_dimension = [item for sublist in dimensions['time'].values() for item in sublist]
        entities = list({v['word']: v for v in flat_dimension}.values())

        return {'entities': entities, 'dimensions': dimensions}

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
            return [
                {
                    'content': word,
                    'weight': similarity_ratio(target_word, word)
                }
                for word in results
            ]
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

        def cos_sim(u, v):
            sim = np.dot(u, v) / (norm(u) * norm(v))
            return sim

        vocab_weighted = list(zip(
            vocab,
            list(map(lambda word: cos_sim(
                cooc[vocab.index(target_word), :], cooc[vocab.index(word), :]), vocab))
        ))
        sorted_list = sorted(vocab_weighted, key=lambda x: x[1], reverse=True)
        return [
            {
                'content': content,
                'weight': weight
            }
            for content, weight in sorted_list[0:n_keyphrases + 1]
        ]

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
                                                    dedupFunc=deduplication_algo, windowsSize=windowSize,
                                                    top=numOfKeywords, features=None)
        keywords = custom_kw_extractor.extract_keywords(text)
        keywords = [keyword[0] for keyword in keywords]
        return keywords

    @staticmethod
    def get_sentence_properties(
            transcript: Transcript or str,
            **kwargs
    ) -> List[dict]:
        if isinstance(transcript, Transcript):
            source_dataframe = transcript.df
        elif isinstance(transcript, str):
            source_dataframe = Utils.text2df(transcript)
        else:
            return TYPE_NOT_SUPPORTED
        dt = UnsupervisedSummarizer(
            csv_file=None,
            source_dataframe=source_dataframe
        )
        results = dt(
            **kwargs
        )
        if Output_type[kwargs.get("output_type")] == Output_type.WORD:
            results = [
                {
                    'content': word,
                    'weight': cnt,
                }
                for word, cnt in results.items()
            ]
        elif Output_type[kwargs.get("output_type")] == Output_type.SENTENCE:
            if kwargs.get("per_cluster_results"):
                return results

        else:
            raise Exception(
                'The output type you requested is not being supported!')
        return Utils.sort_json_by_property(results, "weight")

    @staticmethod
    def get_statistics(transcript: Transcript, speaker_id: str) -> dict:
        num_utterances = 0
        time_spoken = 0
        for turn in transcript.turns:
            if turn.speaker_id == speaker_id:
                num_utterances += 1
                time_spoken += turn.end_time - turn.start_time
        return {
            'num_utterances': num_utterances,
            'time_spoken': time_spoken
        }


if __name__ == '__main__':
    corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
    ]
    print(Extractive.get_tf_idf_keywords(corpus))
