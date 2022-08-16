from collections import OrderedDict
from typing import Union
from transcript_analyser.utils.utils import *
from typing import List, Optional
from transformers import pipeline
from nltk.tokenize import sent_tokenize

from transcript_analyser import Turn

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline(task='text-classification', model=checkpoint)


def tokenize_and_classify(text: str):
    sentences = sent_tokenize(text)
    results = classifier(sentences)
    return results


def aggregate_scores(segments: object) -> object:
    res = {}
    for segment, scores in segments.items():
        scores_for_segment = []
        for i in scores:
            if i['label'] == 'POSITIVE':
                scores_for_segment.append(1)
            else:
                scores_for_segment.append(0)
        res[segment] = np.round(np.mean(scores_for_segment), 2)
    return res


def get_sentiments(text: str, turns: List[Turn], speaker_info: {}, dimensions: bool, section_length: int) -> \
        Union[dict[str, dict[str, object]], List[dict]]:
    if dimensions:
        turns_segmented_by_speaker = OrderedDict()
        for speaker_id in speaker_info.keys():
            turns_segmented_by_speaker[speaker_id] = [turn.text for turn in turns if turn.speaker_id == speaker_id]

        turns_segmented_by_time = [
            (f'{speaker_info[speaker_id]}', text)
            for speaker_id, text in zip([turn.speaker_id for turn in turns], [turn.text for turn in turns])
        ]
        turns_segmented_by_time = Utils.get_sections_from_texts(turns_segmented_by_time, section_length)

        scores_per_speaker = {i: classifier(''.join(utterances).split('.')) for i, utterances in
                              turns_segmented_by_speaker.items()}
        scores_per_time_segment = {i: classifier(''.join(utterances).split('.')) for i, utterances in
                                   enumerate(turns_segmented_by_time)}

        sentences = sent_tokenize(text)
        results = classifier(sentences)

        time_counter = []
        for turn in turns:
            turn_sents = sent_tokenize(turn.text)
            word_counter = 0
            for sent in turn_sents:
                fragment_len = len([word for word in sent.split(' ') if word != ''])
                start_time = turn.words[word_counter].start_time
                end_time = turn.words[word_counter + fragment_len - 1].end_time
                avg_time = (end_time - start_time) + start_time
                word_counter += fragment_len
                time_counter.append(round(avg_time, 2))

        return {
            'dimensions': {
                'time': aggregate_scores(scores_per_speaker),
                'speaker': aggregate_scores(scores_per_time_segment)
            },
            'sentiments': [{'content': sent, **result, 'timestamp': timestamp} for sent, result, timestamp in
                           list(zip(sentences, results, time_counter))]
        }

    else:
        sentences = sent_tokenize(text)
        results = classifier(sentences)
        return [{'content': sent, **result} for sent, result in list(zip(sentences, results))]
