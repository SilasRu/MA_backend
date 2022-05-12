import numpy as np
import json
from enum import Enum, auto, unique
from typing import Any, Dict, List
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
    start_times: List[float or int] = Field(
        [], description='The list of the start times using them you want to filter the transcript to be analyzed.', example=[0])
    end_times: List[float or int] = Field(
        [], description='The list of end times that with them you want to filter the transcript to be analyzed.', example=[200])
    transcript: Dict = Field(
        ..., description='The json object of the transcript which has to be in the Interscriber Format.')


class StatisticsResponseObj(BaseModel):
    topics: List[str] = Field(
        ..., description='The most important topics discussed in the meeting.', example=['Topic A', 'Topic B', 'Topic C']
    )
    statistics: Dict[str, Any] = Field(..., description='Some simple statistics about the meeting like the number of speakers or the proportion of the meeting that they are speaking in.', example={
        "num_speakers": 2,
        "num_utterances": 18,
        "speaker_times": {
            "2": 36.78999999999999,
            "1": 66.77000000000001
        }
    })
