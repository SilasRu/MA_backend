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


def get_index(schema: SchemaClass = TranscriptSchema, index_name: str = 'tmp'):
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
        results = [
            dict(result)
            for result
            in results
        ]

    return results


def autocorrect_query():
    raise NotImplementedError()
