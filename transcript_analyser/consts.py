from http.client import FORBIDDEN
from typing import List

from pydantic import BaseModel, Field, Json


VALIDATION_ERR = "Could not validate credentials!"
FORBIDDEN_STATUS_CODE = 403


class TranscriptInputObj(BaseModel):
    speaker_ids: List[int] = Field(
        default=[], description='The list of the speakers regarding them, the transcript would be filtered!', example=[1, 2])
    start_times: List[float or int] = Field(
        default=[], description='The list of the start times using them you want to filter the transcript to be analyzed.', example=[0])
    end_times: List[float or int] = Field(
        default=[], description='The list of end times that with them you want to filter the transcript to be analyzed.', example=[200])
    transcript: Json = Field(
        ..., description='The json object of the transcript which has to be in the Interscriber Format.')
