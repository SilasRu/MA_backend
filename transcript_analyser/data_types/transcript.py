import json
import pandas as pd
from transcript_analyser.utils.utils import *


class Word:
    def __init__(self, word_obj: json) -> None:
        self._read_word_object(word_obj)

    def _read_word_object(self, word_obj: json) -> None:
        self.confidence = word_obj['attrs']['confidence']
        self.start_time = word_obj['attrs']['startTime']
        self.end_time = word_obj['attrs']['endTime']
        self.text = word_obj['content'][0]['text']

    @property
    def json(self):
        return {
            'confidence': self.confidence,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'text': self.text
        }


class Turn:
    def __init__(self, turn_obj: json) -> None:
        self.words: List[Word] = []
        self._read_turn_object(turn_obj)

    def _read_turn_object(self, turn_obj: json) -> None:
        self.speaker_id = turn_obj['attrs']['speakerId']
        self.start_time = turn_obj['attrs']['startTime']
        self.end_time = turn_obj['attrs']['endTime']
        for word_obj in turn_obj['content']:
            if word_obj['attrs']['type'] == 'WORD':
                self.words.append(Word(word_obj))

    @property
    def text(self):
        return "".join([word.text for word in self.words])

    @property
    def json(self):
        output_json = {}
        output_json['speaker_id'] = self.speaker_id
        output_json['start_time'] = self.start_time
        output_json['end_time'] = self.end_time
        output_json['words'] = []
        for word in self.words:
            output_json['words'].append(word.json)
        return output_json


class Transcript:
    def __init__(self, json_obj: dict) -> None:
        self.turns: List[Turn] = []
        self._read_json_file(json_obj)

    def __len__(self) -> int:
        return len(self.turns)

    def _read_json_file(self, json_obj: dict) -> None:
        for turn_obj in json_obj['content'][0]['content']:
            self.turns.append(Turn(turn_obj))

    @property
    def text(self):
        return "".join([turn.text for turn in self.turns])

    @property
    def df(self):
        df = pd.DataFrame()
        for turn in self.turns:
            df = df.append(pd.Series({
                'Speaker': turn.speaker_id,
                'Start time': turn.start_time,
                'End time': turn.end_time,
                'Duration (in sec)': turn.end_time - turn.start_time,
                'Utterance': turn.text
            }), ignore_index=True)
        return df

    @property
    def json(self):
        output_json = []
        for turn in self.turns:
            output_json.append(turn.json)
        return output_json


def main():
    with open('transcriptsforkeyphraseextraction/transcript_document.json') as f:
        transcript = Transcript(json.load(f))
        f.close()
    Utils.pretty_print_json(transcript.json)


if __name__ == "__main__":
    main()
