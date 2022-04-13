import json
import hashlib
from typing import Dict, Any
from hmac import trans_5C
from transcript_analyser.data_types.transcript import Transcript, Turn
from whoosh.index import create_in
from whoosh.fields import *
from whoosh import index
from tqdm.auto import tqdm
from whoosh.qparser import QueryParser

import os


class TranscriptSchema(SchemaClass):
    speaker = KEYWORD(stored=True)
    body = TEXT(stored=True)


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()

    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


def get_index(transcript: Transcript, schema: SchemaClass = TranscriptSchema):
    index_name = str(dict_hash(transcript.json))
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
        body=str(turn.text)
    )
    writer.commit()
    print('success')


def add_many_documents(ix: index.Index, transcript: Transcript):
    writer = ix.writer()
    for turn in tqdm(transcript.turns, leave=False):
        writer.add_document(
            speaker=str(turn.speaker_id),
            body=str(turn.text)
        )
    writer.commit()
    print('success')


def search(ix: index.Index, target_word: str):
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
