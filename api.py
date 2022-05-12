from typing import Any, List
from fastapi.middleware.cors import CORSMiddleware
import warnings

from transcript_analyser.consts import *
from transcript_analyser.data_types.general import StatisticsResponseObj, TranscriptInputObj

from transcript_analyser.interface import Interface
from fastapi import Depends, FastAPI, HTTPException, Query, Security
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
            status_code=FORBIDDEN_STATUS_CODE, detail=VALIDATION_ERR
        )


app = FastAPI(
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


@app.post('/transcript-analyser/keyphrases/')
def get_keyphrases(
    json_obj: TranscriptInputObj,
    algorithm: str,
    n_keyphrases: int = Query(N_KEYPHRASES,
                              description=N_KEYPHRASES_DESC),
    n_grams_min: int = Query(
        N_GRAMS_MIN, description=N_GRAMS_MIN_DESC),
    n_grams_max: int = Query(N_GRAMS_MAX, description=N_GRAMS_MAX_DESC)
) -> List[Any]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_keyphrases(
        transcript=transcript,
        algorithm=algorithm,
        n_keyphrases=n_keyphrases,
        n_grams_min=n_grams_min,
        n_grams_max=n_grams_max
    )


@app.post('/transcript-analyser/statistics/', response_model=StatisticsResponseObj)
def get_statistics(
    json_obj: TranscriptInputObj,
) -> json:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_statistics(
        transcript=transcript
    )


@app.post('/transcript-analyser/important-text-blocks/')
def get_important_text_blocks(
    json_obj: TranscriptInputObj,
    output_type: str = Query("WORD", description=OUTPUT_TYPE_DESC),
    filter_backchannels: bool = Query(
        True, description=FILTER_BACKCHANNELS_DESC),
    remove_entailed_sentences: bool = Query(
        True, description=REMOVE_ENTAILED_SENTENCES_DESC),
    get_graph_backbone: bool = Query(
        True, description=GET_GRAPH_BACKBONE_DESC),
    do_cluster: bool = Query(True, description=DO_CLUSTER_DESC),
    clustering_algorithm: str = Query(
        'louvain', description=CLUSTERING_ALGORITHM_DESC),
    per_cluster_results: bool = Query(
        False, description=PER_CLUSTER_RESULTS_DESC),
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


@app.post('/transcript-analyser/related-words/')
def get_related_words(
    json_obj: TranscriptInputObj,
    target_word: str,
    n_keyphrases: int = Query(N_KEYPHRASES,
                              description=N_KEYPHRASES_DESC)
) -> List[str]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_related_words(
        transcript=transcript,
        target_word=target_word,
        n_keyphrases=n_keyphrases
    )


@app.post('/transcript-analyser/sentiments/')
def get_sentiments(
    json_obj: TranscriptInputObj,
) -> List[str]:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.get_sentiments(
        transcript=transcript
    )


@app.post('/transcript-analyser/search/')
def search(
    json_obj: TranscriptInputObj,
    target_word: str
) -> Any:
    transcript = Interface.preprocess(json_obj=json_obj)
    return Interface.search(
        transcript=transcript,
        target_word=target_word
    )
