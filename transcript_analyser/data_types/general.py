import numpy as np
import json
from enum import Enum, auto, unique
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


@unique
class Output_type(Enum):
    # HTML = auto()
    SENTENCE = auto()
    WORD = auto()


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class TranscriptInputObj(BaseModel):
    speaker_ids: List[int] = Field(
        [], description='The list of the speakers regarding them, the transcript would be filtered!', example=[1, 2])
    speaker_info: List[Dict] = Field(
        [], description='The list of the speakers info', example=[{"id": 1, "name": "Bob"}])
    start_times: List[float or int] = Field(
        [], description='The list of the start times using them you want to filter the transcript to be analyzed.',
        example=[0])
    end_times: List[float or int] = Field(
        [], description='The list of end times that with them you want to filter the transcript to be analyzed.',
        example=[200])
    transcript: Dict = Field(
        ..., description='The json object of the transcript which has to be in the Interscriber Format.')


class KeywordsResponseObj(BaseModel):
    dimensions: Dict[str, Dict[str, Any]] = Field(
        {}, description='Dimensions that keywords are returned for',
        example={'time': {'0': 'Foo'}, 'speaker': {'0': 'Foo'}}
    )


class EntitiesResponseObj(BaseModel):
    dimensions: Dict[str, Dict[str, Any]] = Field(
        {}, description='Dimensions that keywords are returned for',
        example={'time': {'0': 'Foo'}, 'speaker': {'0': 'Foo'}}
    ),
    entities: List[Dict[str, Any]] = Field({}, description='All entities occurring in the transcript',
                                           example={
                                               'word': 'Peter',
                                               'entity_group': 'PER',
                                               'occurrence': 5,
                                               'in_speakers': False
                                           })


class StatisticsResponseObj(BaseModel):
    topics: List[str] = Field(
        ..., description='The most important topics discussed in the meeting.',
        example=['Topic A', 'Topic B', 'Topic C']
    )

    num_utterances: int = Field(
        ..., description='The number of utterances in the meeting.', example=10)

    meeting_duration: float = Field(...,
                                    description='The duration of the meeting.', example=10.0)
    speaker_stats: Dict[str, Dict[str, Any]] = Field(
        ..., description='The statistics of the speakers.',
        example={'speaker_1': {'num_utterances': 10, 'duration': 10.0, 'topics': ['Topic A', 'Topic B', 'Topic C']},
                 'speaker_2': {'num_utterances': 10, 'duration': 10.0, 'topics': ['Topic A', 'Topic B', 'Topic C']}}
    )


class SentimentsResponseObj(BaseModel):
    content: str = Field(..., description="Sentence.",
                         example="Uh huh mary hi hello, I'm Susan Thompson Resource manager.")
    label: str = Field(..., description="The overal sentiment for the sentence.",
                       example="POSITIVE")
    score: float = Field(..., description='Confidence score of the sentiment that is being assigned to this sentence.',
                         example=0.9716441035270691)


class RelatedWordsResponseObj(BaseModel):
    content: str = Field(..., example="kitchen")
    weight: float = Field(..., example=0.9999999999999998)


class SearchResponseObj(BaseModel):
    guessed_word: str = Field(
        None,
        description='If the specific words that is being searched for is not found in the text, different candidate words would be used to search which are specified in this field.')
    body: str = Field(..., description='The document which is returned in response to the search of word.')
    speaker: str = Field(...,
                         description="The speaker that has uttered the document being returned as response of the search.")
    start_time: float = Field(
        ..., description='The start time of the document or the utterance that is being returned for search.')


class JobResponseObj(BaseModel):
    status: str = Field(..., description='The status of the job.')
    output: Dict or SearchResponseObj or RelatedWordsResponseObj or SentimentsResponseObj or StatisticsResponseObj

# TODO change this later to omit Dict as a general json object from it
