from typing import Dict, List

from numpy import indices
from transcript_analyser.data_types.transcript import Transcript, Turn
from whoosh.index import create_in
from whoosh.fields import *
from whoosh import index
from tqdm.auto import tqdm
from whoosh.qparser import QueryParser

import os

from transcript_analyser.utils.utils import Utils


if not os.path.exists('indices'):
    os.mkdir('indices')


class TranscriptSchema(SchemaClass):
    speaker = KEYWORD(stored=True)
    body = TEXT(stored=True)
    start_time = STORED()


def get_index(transcript: Transcript, schema: SchemaClass = TranscriptSchema):
    index_name = str(Utils.dict_hash(transcript.json))
    indices_directory = os.path.join("indices", index_name)
    if not os.path.exists(
        indices_directory
    ):
        os.mkdir(indices_directory)
        ix = index.create_in(indices_directory, schema, indexname=index_name)
        return ix, 0
    else:
        ix = index.open_dir(indices_directory, indexname=index_name)
        return ix, 1


def add_document(ix: index.Index, turn: Turn):
    writer = ix.writer()
    writer.add_document(
        speaker=str(turn.speaker_id),
        body=str(turn.text),
        start_time=turn.start_time
    )
    writer.commit()
    print('success')


def add_many_documents(ix: index.Index, transcript: Transcript):
    writer = ix.writer()
    for turn in tqdm(transcript.turns, leave=False):
        writer.add_document(
            speaker=str(turn.speaker_id),
            body=str(turn.text),
            start_time=turn.start_time
        )

    writer.commit()
    print('success')


def search(ix: index.Index, target_word: str) -> List[Dict]:
    """Search in the given index for the target word

    Args:
        ix (index.Index): the index which the document is stored in
        target_word (str): the word to search for in the index

    Returns:
        List[Dict]: The sentences that contain the target word
    """
    qp = QueryParser("body", schema=ix.schema)
    q = qp.parse(str(target_word))
    with ix.searcher() as searcher:
        results = searcher.search(q)
        all_results = [
            dict(result)
            for result
            in results
        ]
        if len(all_results) == 0:
            corrector = searcher.corrector("body")
            suggested_words = corrector.suggest(target_word)
            for suggested_word in suggested_words:
                q = qp.parse(str(suggested_word))
                results = searcher.search(q)
                all_results.extend([
                    {
                        "guessed_word": suggested_word,
                        **dict(result)
                    }
                    for result
                    in results
                ])
    return all_results
