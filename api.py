from fastapi.middleware.cors import CORSMiddleware
from typing import Any, List, Optional, Tuple
import warnings
from Transcript_Analysis.Output import *
from Transcript_Analysis.interface import Interface
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyQuery, APIKeyCookie, APIKeyHeader, APIKey
import json

from fastapi.responses import HTMLResponse

warnings.filterwarnings("ignore")

API_KEY = "bcqoieyqp98DAHJBABJBy3498ypiuqhriuqy984"
API_KEY_NAME = "access_token"


api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(

    api_key_header: str = Security(api_key_header),

):

    if api_key_header == API_KEY:
        return api_key_header

    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials"
        )


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/TranscriptAnalysis/keyphrases/')
async def get_keyphrases(
    json_obj: dict,
    algorithm: str,
    api_key: APIKey = Depends(get_api_key),
    n_keyphrases: Optional[int] = None,
    n_grams_min: Optional[int] = 0,
    n_grams_max: Optional[int] = 0
) -> List[Any]:
    return Interface.get_keyphrases(
        json_obj=json_obj,
        algorithm=algorithm,
        n_keyphrases=n_keyphrases,
        n_grams_min=n_grams_min,
        n_grams_max=n_grams_max
    )


@app.post('/TranscriptAnalysis/statistics/')
async def get_statistics(
    json_obj: dict,
    api_key: APIKey = Depends(get_api_key),
) -> json:
    return Interface.get_statistics(json_obj=json_obj)


@app.post('/TranscriptAnalysis/importantTextBlocks/')
async def get_important_text_blocks(
    json_obj: dict,
    type: str,
    api_key: APIKey = Depends(get_api_key),
) -> List[Keyword] or HTMLResponse:
    return Interface.get_important_text_blocks(
        json_obj=json_obj,
        type=type
    )


@app.post('/TranscriptAnalysis/relatedWords/')
async def get_related_words(
    json_obj: dict,
    target_word: str,
    api_key: APIKey = Depends(get_api_key),
    n_keyphrases: Optional[int] = 0
) -> List[Keyword]:
    return Interface.get_related_words(
        json_obj=json_obj,
        target_word=target_word,
        n_keyphrases=n_keyphrases
    )
