from .abstractive.Abstractive import Abstractive
from .data_types.Transcript import *
from .extractive.Extractive import *
from fastapi.responses import HTMLResponse

from .utils.Autocomplete import Meeting_Autocomplete


class CustomError(Exception):
    pass


class Interface:
    @staticmethod
    def apply_conditions(transcript: Transcript, start_times: List[float], end_times: List[float], speaker_ids: List[int]) -> Transcript:
        """
        Apply the filters on the speakers and the start times and end times imposed
        """
        if len(speaker_ids) != 0:
            transcript = Interface.filter_speaker(transcript, speaker_ids)
        if len(start_times) != len(end_times):
            raise CustomError(
                'The length of the start times and end times should be the same')
        if len(start_times) != 0:
            transcript = Interface.filter_time(
                transcript, start_times, end_times)
        return transcript

    @staticmethod
    def get_keyphrases(
        json_obj: dict,
        algorithm: str,
        n_keywords: int,
        n_grams_min: int,
        n_grams_max: int
    ) -> List[str or dict] or str:
        """
        Get the key phrases or the generated summaries
        """
        transcript = Transcript(json_obj['transcript'])
        transcript = Interface.apply_conditions(
            transcript=transcript, start_times=json_obj['start_times'],
            end_times=json_obj['end_times'], speaker_ids=json_obj['speaker_ids']
        )
        if algorithm == "keybert":
            return Abstractive.get_keybert_keywords(
                text=transcript.text,
                keyphrase_ngram_range=(n_grams_min, n_grams_max),
                n_keywords=n_keywords
            )

        elif algorithm == "rake":
            return Extractive.get_rake_keywords(
                text=transcript.text
            )

        elif algorithm == "yake":
            return Extractive.get_yake_keywords(
                text=transcript.text
            )

        elif algorithm == "frequency":
            return Extractive.get_frequent_keywords(
                transcript=transcript
            )
        elif algorithm == "bart":
            return Abstractive.get_bart_summary(
                text=transcript.text
            )
        else:
            raise NotImplementedError

    @staticmethod
    def get_statistics(
        json_obj: dict
    ) -> Any:
        """
        Get some descriptive statistics about the utterances being fed
        """
        transcript = Transcript(json_obj['transcript'])
        transcript = Interface.apply_conditions(
            transcript=transcript, start_times=json_obj['start_times'],
            end_times=json_obj['end_times'], speaker_ids=json_obj['speaker_ids']
        )
        topics = Abstractive.get_keybert_keywords(
            text=transcript.text, keyphrase_ngram_range=(0, 0), n_keywords=3)
        statistics = Extractive.get_statistics(transcript=transcript)
        return {
            'topics': topics,
            'statistics': statistics
        }

    @staticmethod
    def get_highlights(
        json_obj: dict,
        highlight_type: str,
    ) -> List[dict] or HTMLResponse:
        """
        Get the highlights of the meeting based on different algorithms such as Louvain community detection or sentence weights
        """
        transcript = Transcript(json_obj['transcript'])
        transcript = Interface.apply_conditions(
            transcript=transcript, start_times=json_obj['start_times'],
            end_times=json_obj['end_times'], speaker_ids=json_obj['speaker_ids']
        )

        if highlight_type == "sentence_weights":
            return Extractive.get_sentence_weights(
                transcript=transcript
            )
        elif highlight_type == "topics_by_louvain":
            return Extractive.get_louvain_topics_sentences(
                transcript=transcript
            )
        else:
            raise NotImplementedError

    @staticmethod
    def get_related_words(
        json_obj: dict,
        target_word: str,
        n_keywords: int
    ) -> List[str]:
        """
        Get the list of related words to the target word
        """
        transcript = Transcript(json_obj['transcript'])
        transcript = Interface.apply_conditions(
            transcript=transcript, start_times=json_obj['start_times'],
            end_times=json_obj['end_times'], speaker_ids=json_obj['speaker_ids']
        )
        return Extractive.get_related_words(
            transcript=transcript,
            target_word=target_word,
            n_keywords=n_keywords
        )

    @ staticmethod
    def filter_speaker(
        transcript: Transcript,
        speaker_ids: List[int]
    ) -> Transcript:
        transcript.turns = [
            turn for turn in transcript.turns if turn.speaker_id in speaker_ids]
        return transcript

    @ staticmethod
    def filter_time(
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


if __name__ == '__main__':
    print(Interface.tmp_function())
