from typing import List, Union, Dict, Optional, Any
from fastapi.middleware.cors import CORSMiddleware
import warnings

from transcript_analyser.consts import *
from transcript_analyser.data_types.general import RelatedWordsResponseObj, SearchResponseObj, SentimentsResponseObj, \
    StatisticsResponseObj, TranscriptInputObj, KeywordsResponseObj, EntitiesResponseObj

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Security
from fastapi.security.api_key import APIKeyHeader
import os

from transcript_analyser.job_manager import JobManager

job_manager = JobManager()

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
    # dependencies=[Depends(get_api_key)]
)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/hello")
def root():
    return {"message": "Hello World"}

@app.post('/transcript-analyser/keywords/', response_model=KeywordsResponseObj)
def get_keywords(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        section_length: Optional[str] = 175,
        model: Optional[str] = None
):
    return job_manager.do_job(
        json_obj=json_obj,
        task='get_keywords',
        model=model,
        background_tasks=background_tasks,
        section_length=section_length
    )


@app.post('/transcript-analyser/entities/', response_model=EntitiesResponseObj)
def get_keywords(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        section_length: Optional[str] = 175
):
    return job_manager.do_job(
        json_obj=json_obj,
        task='get_entities',
        background_tasks=background_tasks,
        section_length=section_length
    )

@app.post('/transcript-analyser/keyphrases/', response_model=Union[List[str], str, Dict])
def get_keyphrases(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        algorithm: str,
        model: Optional[str] = None,
        section_length: Optional[str] = 175,
        n_keyphrases: int = Query(default=N_KEYPHRASES, description=N_KEYPHRASES_DESC),
        n_grams_min: int = Query(default=N_GRAMS_MIN, description=N_GRAMS_MIN_DESC),
        n_grams_max: int = Query(default=N_GRAMS_MAX, description=N_GRAMS_MAX_DESC)
):
    return job_manager.do_job(
        json_obj=json_obj,
        background_tasks=background_tasks,
        task='get_keyphrases',
        algorithm=algorithm,
        model=model,
        section_length=section_length,
        n_keyphrases=n_keyphrases,
        n_grams_min=n_grams_min,
        n_grams_max=n_grams_max
    )


@app.post('/transcript-analyser/statistics/', response_model=StatisticsResponseObj)
def get_statistics(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks
):
    return job_manager.do_job(
        json_obj=json_obj,
        background_tasks=background_tasks,
        task='get_statistics'
    )


# TODO write the documentation for the repsonse model
@app.post('/transcript-analyser/important-text-blocks/')
def get_important_text_blocks(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        output_type: str = Query(default="WORD", description=OUTPUT_TYPE_DESC),
        filter_backchannels: bool = Query(
            default=True, description=FILTER_BACKCHANNELS_DESC),
        remove_entailed_sentences: bool = Query(
            default=True, description=REMOVE_ENTAILED_SENTENCES_DESC),
        get_graph_backbone: bool = Query(
            default=True, description=GET_GRAPH_BACKBONE_DESC),
        do_cluster: bool = Query(default=True, description=DO_CLUSTER_DESC),
        clustering_algorithm: str = Query(
            default='louvain', description=CLUSTERING_ALGORITHM_DESC),
        per_cluster_results: bool = Query(
            default=False, description=PER_CLUSTER_RESULTS_DESC),
):
    return job_manager.do_job(
        json_obj=json_obj,
        background_tasks=background_tasks,
        task='get_important_text_blocks',
        output_type=output_type,
        filter_backchannels=filter_backchannels,
        remove_entailed_sentences=remove_entailed_sentences,
        get_graph_backbone=get_graph_backbone,
        do_cluster=do_cluster,
        clustering_algorithm=clustering_algorithm,
        per_cluster_results=per_cluster_results
    )


@app.post('/transcript-analyser/related-words/', response_model=List[RelatedWordsResponseObj])
def get_related_words(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        target_word: str = Query(default=None, min_length=MIN_WORD_LEN),
        n_keyphrases: int = Query(default=N_KEYPHRASES,
                                  description=N_KEYPHRASES_DESC)
):
    return job_manager.do_job(
        json_obj=json_obj,
        task='get_related_words',
        background_tasks=background_tasks,
        target_word=target_word,
        n_keyphrases=n_keyphrases
    )


@app.post('/transcript-analyser/sentiments/',
          response_model=Union[List[SentimentsResponseObj], Dict])
def get_sentiments(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        dimensions: Optional[bool] = False,
        section_length: Optional[str] = 175
):
    return job_manager.do_job(
        json_obj=json_obj,
        task='get_sentiments',
        background_tasks=background_tasks,
        section_length=section_length,
        dimensions=dimensions
    )


@app.post('/transcript-analyser/search/', response_model=List[SearchResponseObj])
def search(
        json_obj: TranscriptInputObj,
        background_tasks: BackgroundTasks,
        target_word: str = Query(default=None, min_length=MIN_WORD_LEN),
):
    return job_manager.do_job(
        json_obj=json_obj,
        task='search',
        background_tasks=background_tasks,
        target_word=target_word
    )
