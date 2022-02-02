from fastapi.middleware.cors import CORSMiddleware
from typing import Any, List, Optional, Tuple
import warnings
from Transcript_Analysis.Output import *
from Transcript_Analysis.interface import Interface
from fastapi import FastAPI
import json

from fastapi.responses import HTMLResponse

warnings.filterwarnings("ignore")

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/TranscriptAnalysis/keyphrases/')
async def get_keywords(
    json_obj: dict,
    algorithm: str,
    n_keywords: Optional[int] = None,
    n_grams_min: Optional[int] = 0,
    n_grams_max: Optional[int] = 0
) -> List[Any]:
    return Interface.get_keyphrases(
        json_obj=json_obj,
        algorithm=algorithm,
        n_keywords=n_keywords,
        n_grams_min=n_grams_min,
        n_grams_max=n_grams_max
    )


@app.post('/TranscriptAnalysis/statistics/')
async def get_statistics(
    json_obj: dict,
) -> json:
    return Interface.get_statistics(json_obj=json_obj)


@app.post('/TranscriptAnalysis/highlights/')
async def get_highlights(
    json_obj: dict,
    highlight_type: str,
) -> List[Keyword] or HTMLResponse:
    return Interface.get_highlights(
        json_obj=json_obj,
        highlight_type=highlight_type
    )


@app.post('/TranscriptAnalysis/relatedWords/')
async def get_related_words(
    json_obj: dict,
    target_word: str,
    n_keywords: Optional[int] = 0
) -> List[Keyword]:
    return Interface.get_related_words(
        json_obj=json_obj,
        target_word=target_word,
        n_keywords=n_keywords
    )
