import json
from fastapi import BackgroundTasks
from regex import F
from transcript_analyser.abstractive.abstractive import Abstractive
from transcript_analyser.consts import CACHING_TIME_TOLERANCE, N_GRAMS_MAX, N_GRAMS_MIN, N_KEYPHRASES
from transcript_analyser.data_types.general import TranscriptInputObj
from transcript_analyser.extractive.extractive import Extractive
from transcript_analyser.searchers.whoosh_searcher import add_many_documents, get_index, search
from transcript_analyser.analysers import sentiment_analyser
from typing import Any, Dict, List, Union
from path import Path
import os
import time

from transcript_analyser.data_types.transcript import Transcript
from transcript_analyser.utils.utils import Utils


JOBS_DIRECTORY = os.getenv('JOBS_DIR')
if not os.path.exists(JOBS_DIRECTORY):
    os.mkdir(JOBS_DIRECTORY)


class CustomError(Exception):
    pass


class JobManager:

    def do_job(
        self,
        background_tasks: BackgroundTasks,
        json_obj: TranscriptInputObj,
        task: str,
        **kwargs
    ) -> str:
        transcript = self.__preprocess(json_obj=json_obj)

        job_id = self.__get_job_id(transcript, task, **kwargs)
        cached_results = self.__get_job(job_id=job_id)
        if cached_results:
            return cached_results

        start_time = time.time()
        results = getattr(self, task)(transcript, **kwargs)
        end_time = time.time()

        if end_time - start_time > CACHING_TIME_TOLERANCE:
            self.__store_job(job_id=job_id, output=results)

        return results

    def __get_job_id(
        self,
        transcript: Transcript,
        task: str,
        **kwargs
    ) -> str:
        return Utils.dict_hash({
            "transcript": transcript.json,
            "task": task,
            **kwargs
        })


# TODO, store the data companied with the status of the job being ["in_progress", "completed"]


    def __store_job(
        self,
        job_id: str,
        output: dict
    ) -> None:
        path = os.path.join(JOBS_DIRECTORY, f'{job_id}.json')
        with open(path, 'w') as f:
            json.dump(output, f)

    def __get_job(
        self,
        job_id: str
    ) -> Union[None, Dict]:
        path = os.path.join(JOBS_DIRECTORY, f'{job_id}.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None

    def __preprocess(self, json_obj: TranscriptInputObj):
        json_obj = json_obj.dict()
        transcript = Transcript(json_obj['transcript'])
        transcript = self.__apply_conditions(
            transcript=transcript,
            start_times=json_obj['start_times'],
            end_times=json_obj['end_times'],
            speaker_ids=json_obj['speaker_ids'],
            speaker_info=json_obj['speaker_info']
        )
        return transcript

    def __apply_conditions(
        self,
        transcript: Transcript,
        start_times: List[float],
        end_times: List[float],
        speaker_ids: List[int],
        speaker_info: List[Dict]
    ) -> Transcript:
        """
        Apply the filters on the speakers and the start times and end times imposed
        """
        transcript = self.__lower_case(transcript)
        if len(speaker_ids) != 0:
            transcript = self.__filter_speaker(transcript, speaker_ids)
        if len(start_times) != len(end_times):
            raise CustomError(
                'The length of the start times and end times should be the same')
        if len(start_times) != 0:
            transcript = self.__filter_time(
                transcript, start_times, end_times)
        transcript.speaker_info = {i['id']: i['name'] for i in speaker_info}
        return transcript

    def get_keyphrases(
        self,
        transcript: Transcript,
        **kwargs
    ) -> List[str or dict] or str:
        """
        Get the key phrases or the generated summaries
        """

        if kwargs.get('algorithm') == "keybert":
            return Abstractive.get_keybert_keywords(
                text=transcript.text,
                keyphrase_ngram_range=(kwargs.get(
                    'n_grams_min'), kwargs.get('n_grams_max')),
                n_keyphrases=kwargs.get('n_keyphrases')
            )

        elif kwargs.get('algorithm') == "rake":
            return Extractive.get_rake_keywords(
                text=transcript.text,
                top_n=kwargs.get('n_keyphrases')
            )

        elif kwargs.get('algorithm') == "yake":
            return Extractive.get_yake_keywords(
                text=transcript.text
            )

        elif kwargs.get('algorithm') == "bart":
            return Abstractive.get_bart_summary(
                turns=transcript.turns,
                speaker_info=transcript.speaker_info,
                model=kwargs.get('model')
            )
        elif kwargs.get('algorithm') == "lsa":
            return Extractive.get_lsa_sentences(
                text=transcript.text,
                n_keyphrases=kwargs.get('n_keyphrases')
            )
        else:
            raise NotImplementedError

    def get_statistics(
        self,
        transcript: Transcript,
    ) -> Any:
        """
        Get some descriptive statistics about the utterances being fed
        """

        topics = Abstractive.get_keybert_keywords(
            text=transcript.text, keyphrase_ngram_range=(N_GRAMS_MIN, N_GRAMS_MAX), n_keyphrases=N_KEYPHRASES)

        speaker_ids = set([turn.speaker_id for turn in transcript.turns])
        speaker_stats = {}
        for speaker_id in speaker_ids:
            speaker_stats[speaker_id] = Extractive.get_statistics(
                transcript=transcript, speaker_id=speaker_id)
            speaker_turns = [
                turn for turn in transcript.turns if turn.speaker_id == speaker_id]
            speaker_stats[speaker_id]['topics'] = Abstractive.get_keybert_keywords(
                text=' '.join([turn.text for turn in speaker_turns]),
                keyphrase_ngram_range=(N_GRAMS_MIN, N_GRAMS_MAX),
                n_keyphrases=N_KEYPHRASES
            )
        num_utterances = len(transcript.turns)
        meeting_duration = sum(
            [speaker_stats[speaker_id]['time_spoken']
                for speaker_id in speaker_ids]
        )

        return {
            'topics': topics,
            'num_utterances': num_utterances,
            'meeting_duration': meeting_duration,
            'speaker_stats': speaker_stats
        }

    def get_important_text_blocks(
        self,
        transcript: Transcript,
        **kwargs
    ) -> List[dict or str]:
        """
        Get the important_text_blocks of the meeting based on different algorithms such as Louvain community detection or sentence weights
        """

        return Extractive.get_sentence_properties(
            transcript,
            **kwargs
        )

    def get_related_words(
        self,
        transcript: Transcript,
        **kwargs
    ) -> List[str]:
        """
        Get the list of related words to the target word
        """

        return Extractive.get_related_words(
            text=transcript.text,
            target_word=kwargs.get("target_word"),
            n_keyphrases=kwargs.get("n_keyphrases")
        )

    def get_sentiments(
        self,
        transcript: Transcript,
        dimension: str
    ) -> List[dict]:
        """
        Get the sentiment for each sentence in the meeting
        """
        return sentiment_analyser.get_sentiments(
            text=transcript.text,
            dimension=dimension
        )

    def __lower_case(self, transcript: Transcript):
        return transcript.lower()

    def __filter_speaker(
        self,
        transcript: Transcript,
        speaker_ids: List[int]
    ) -> Transcript:
        transcript.turns = [
            turn for turn in transcript.turns if turn.speaker_id in speaker_ids]
        return transcript

    def __filter_time(
        self,
        transcript: Transcript,
        start_times: List[float],
        end_times: List[float]
    ) -> Transcript:
        for turn in transcript.turns:
            turn.words = [word for word in turn.words if Utils.contains(
                (word.start_time, word.end_time), start_times, end_times)]
        transcript.turns = [
            turn for turn in transcript.turns if len(turn.words) != 0]
        return transcript

    def search(
        self,
        transcript: Transcript,
        **kwargs
    ) -> Any:
        ix, index_exists = get_index(transcript=transcript)
        if not index_exists:
            add_many_documents(ix, transcript=transcript)
        return search(ix, target_word=kwargs.get("target_word"))
