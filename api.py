from inspect import trace
from typing import Any, List
from fastapi.middleware.cors import CORSMiddleware
import warnings

from transcript_analyser.interface import Interface
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
import json
import os


warnings.filterwarnings("ignore")

API_KEY = os.getenv('API_KEY')
API_KEY_NAME = os.getenv('API_KEY_NAME')


api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def get_api_key(
    api_key_header: str = Security(api_key_header),
):

    if api_key_header == API_KEY:
        return api_key_header

    else:
        raise HTTPException(
            status_code=403, detail="Could not validate credentials!"
        )


app = FastAPI(
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    dependencies=[Depends(get_api_key)]
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post('/transcript-analysis/keyphrases/')
def get_keyphrases(
    json_obj: dict,
    algorithm: str,
    n_keyphrases: int = 3,
    n_grams_min: int = 1,
    n_grams_max: int = 3
) -> List[Any]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_keyphrases(
        transcript=transcript,
        algorithm=algorithm,
        n_keyphrases=n_keyphrases,
        n_grams_min=n_grams_min,
        n_grams_max=n_grams_max
    )


@app.post('/transcript-analysis/statistics/')
def get_statistics(
    json_obj: dict,
) -> json:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_statistics(
        transcript=transcript
    )


@app.post('/transcript-analysis/important-text-blocks/')
def get_important_text_blocks(
    json_obj: dict,
    output_type: str = "WORD",
    filter_backchannels: bool = True,
    remove_entailed_sentences: bool = True,
    get_graph_backbone: bool = True,
    do_cluster: bool = True,
    clustering_algorithm: str = 'louvain',
    per_cluster_results: bool = False,
) -> List[dict or str]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_important_text_blocks(
        transcript=transcript,
        output_type=output_type,
        filter_backchannels=filter_backchannels,
        remove_entailed_sentences=remove_entailed_sentences,
        get_graph_backbone=get_graph_backbone,
        do_cluster=do_cluster,
        clustering_algorithm=clustering_algorithm,
        per_cluster_results=per_cluster_results
    )


@app.post('/transcript-analysis/related-words/')
def get_related_words(
    json_obj: dict,
    target_word: str,
    n_keyphrases: int = 5
) -> List[str]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_related_words(
        transcript=transcript,
        target_word=target_word,
        n_keyphrases=n_keyphrases
    )


@app.post('/transcript-analysis/sentiments/')
def get_sentiments(
    json_obj: dict,
) -> List[str]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_sentiments(
        transcript=transcript
    )
