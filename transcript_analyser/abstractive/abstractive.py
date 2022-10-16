from typing import Tuple, Optional, Union
from collections import OrderedDict
from keybert import KeyBERT
from transformers import BartForConditionalGeneration, BartTokenizer, AutoModelForSeq2SeqLM, AutoTokenizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from transcript_analyser.data_types.transcript import Turn
from transcript_analyser.utils.utils import *
from tqdm.auto import tqdm
import nltk


class Abstractive:
    keybert_model = None
    classifier = None

    @classmethod
    def get_attention_keywords(
            cls,
            turns: List,
            speaker_info: {},
            section_length: int,
            keyphrase_dimensions: {}) -> Dict[str, Dict[str, Any]]:

        stop_words = set(nltk.corpus.stopwords.words('english'))
        nltk_tokenizer = nltk.RegexpTokenizer(r"\w+")
        for i in speaker_info.values():
            stop_words.add(i.lower())

        turns_segmented_by_time = [
            (f'{speaker_info[speaker_id]}', text)
            for speaker_id, text in zip([turn.speaker_id for turn in turns], [turn.text for turn in turns])
        ]
        turns_segmented_by_speaker = OrderedDict()
        for speaker_id in speaker_info.keys():
            turns_segmented_by_speaker[speaker_id] = [
                (f'{speaker_info[speaker_id]}', turn.text) for turn in turns if turn.speaker_id == speaker_id
            ]
        time_sections_to_process = Utils.get_sections_from_texts(turns_segmented_by_time, section_length)
        speaker_sections_to_process = [''.join(Utils.get_sections_from_texts(speaker, section_length)) for speaker in
                                       turns_segmented_by_speaker.values()]

        def tokenize(transcript, summary):
            input_tokens = nltk_tokenizer.tokenize(transcript)
            input_tokens = [w for w in input_tokens if not w.lower() in stop_words]
            output_tokens = nltk_tokenizer.tokenize(summary)
            output_tokens = [w for w in output_tokens if not w.lower() in stop_words]
            return input_tokens, output_tokens

        def compute_matches(input_tokens, output_tokens):
            matches = OrderedDict()
            for in_index in range(len(input_tokens)):
                i, o, = in_index, 0
                cur_index_range = []
                curr_str = []

                while i < len(input_tokens) and o < len(output_tokens):
                    if input_tokens[i].lower() == output_tokens[o].lower():
                        '''
                            Append token if it appears at the same spot in both input and output
                            input_tokens=     ['this', 'is', 'the', 'input']
                                                            i
                            output_tokens=    ['the', 'input']
                                                o
                            curr_str=['the']
                            curr_index_range=[2] 
                        '''
                        cur_index_range.append(i)
                        curr_str.append(input_tokens[i])
                        i, o = i + 1, o + 1
                    else:
                        if i - 1 in cur_index_range:
                            '''
                                Check if last token was added in the index range - used for softening the constraint 
                                so one token can be skipped and still account for the n-gram succession
                                input_tokens=     ['this', 'is', 'the', 'someword' 'input']
                                                                            i
                                output_tokens=    ['the', 'input']
                                                            o
                                curr_str=['the']
                                curr_index_range=[2]
                                i += 1 
                            '''
                            i += 1
                        elif len(curr_str) > 0:
                            matches['_'.join([str(i) for i in cur_index_range])] = curr_str
                            cur_index_range = []
                            curr_str = []
                            o += 1
                        else:
                            o += 1
            return matches

        def compute_max_n_gram(matches_dict):
            indexes_with_length = OrderedDict()

            for index_group in matches_dict.keys():
                indexes = index_group.split('_')
                for idx in indexes:
                    if idx in indexes_with_length:
                        if indexes_with_length[idx] < len(indexes):
                            indexes_with_length[idx] = len(indexes)
                    else:
                        indexes_with_length[idx] = len(indexes)

            return indexes_with_length

        def deduplicate_matches(matches_dict, max_n_grams_per_idx):
            # Deduplicate based on index
            deduplicated_matches = []
            for key in matches_dict.keys():
                indexes = key.split('_')
                is_longest = True
                for idx in indexes:
                    if max_n_grams_per_idx[idx] > len(indexes):
                        is_longest = False

                if is_longest:
                    deduplicated_matches.append(matches_dict[key])

            # Deduplicate 1-grams based on word occurence
            seen = set()
            deduplicated_words = []
            for n_gram in deduplicated_matches:
                if len(n_gram) == 1:
                    if n_gram[0].lower() in seen:
                        continue
                    else:
                        seen.add(n_gram[0].lower())
                        deduplicated_words.append(n_gram)
                else:
                    for word in n_gram:
                        if word.lower() not in seen:
                            seen.add(word.lower())
                    deduplicated_words.append(n_gram)

            return [' '.join(i) for i in deduplicated_words]

        def process_sections(raw_sections, summary_dimension):
            dimension_kws = []
            for section, summary in zip(raw_sections, keyphrase_dimensions['dimensions'][summary_dimension].values()):
                input_tokens, output_tokens = tokenize(section, ''.join(summary))
                matches_dict = compute_matches(input_tokens, output_tokens)
                max_n_grams_per_idx = compute_max_n_gram(matches_dict)
                keywords_for_section = deduplicate_matches(matches_dict, max_n_grams_per_idx)
                dimension_kws.append(keywords_for_section)
            return dimension_kws

        speaker_sections = process_sections(speaker_sections_to_process, 'speaker')
        time_sections = process_sections(time_sections_to_process, 'time')
        dimensions = {
            'time': {i: section for i, section in enumerate(time_sections)},
            'speaker': {i: section for i, section in enumerate(speaker_sections)}
        }
        return {'dimensions': dimensions}

    @classmethod
    def get_keybert_keywords(
            cls,
            text: str,
            keyphrase_ngram_range: Tuple,
            n_keyphrases: int
    ) -> List[str]:
        if cls.keybert_model == None:
            cls.keybert_model = KeyBERT()
        vectorizer = KeyphraseCountVectorizer()
        keywords = [entity for entity in
                    cls.keybert_model.extract_keywords(
                        text,
                        vectorizer=vectorizer,
                        keyphrase_ngram_range=keyphrase_ngram_range,
                        top_n=n_keyphrases
                    )]
        keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
        return [keyword[0] for keyword in keywords]

    @staticmethod
    def get_bart_keyphrases_finetuned(utterances: List[Tuple[str, str]],
                                      utterances_by_speaker: OrderedDict[int: Tuple[str, str]],
                                      model_name: str, section_length: int) -> Dict:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        time_sections_to_process = Utils.get_sections_from_texts(utterances, section_length)
        speaker_sections_to_process = [Utils.get_sections_from_texts(speaker, section_length) for speaker in
                                       utterances_by_speaker.values()]

        def generate_summaries(summary_section: [str]) -> list:
            inputs = tokenizer(summary_section, max_length=1024, return_tensors='pt', truncation=True, padding=True)
            summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=0, max_length=200)
            decoded = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return [i.lstrip() for i in decoded[0].split('.') if (i and i != ' ')]

        dimensions = {
            'time': {i: generate_summaries([section]) for i, section in enumerate(time_sections_to_process)},
            'source_time_section': {i: section for i, section in enumerate(time_sections_to_process)},
            'speaker': {i: generate_summaries(section) for i, section in enumerate(speaker_sections_to_process)}
        }
        return {'dimensions': dimensions}

    @staticmethod
    def get_bart_keywords_openai(utterances: List[Tuple[str, str]], debug: bool = False):
        model = BartForConditionalGeneration.from_pretrained(
            "philschmid/bart-large-cnn-samsum")
        tokenizer = BartTokenizer.from_pretrained(
            "philschmid/bart-large-cnn-samsum"
        )
        cnt = 0
        values = list(tokenizer.decoder.values())

        nltk_tokenizer = nltk.RegexpTokenizer(r"\w+")
        words_in_utterances = nltk_tokenizer.tokenize(
            ' '.join([text for _, text in utterances])
        )
        speaker_in_utterances = nltk_tokenizer.tokenize(
            ' '.join([speaker for speaker, _ in utterances])
        )
        filtered_values = [
            word for word in values if (word not in words_in_utterances) and (word not in speaker_in_utterances)
        ]
        bad_words_ids = [tokenizer.encode(
            bad_word, add_prefix_space=True) for bad_word in filtered_values]

        output = ''
        while True:
            if cnt == 1:
                model = BartForConditionalGeneration.from_pretrained(
                    'sshleifer/distilbart-cnn-12-6')
                tokenizer = BartTokenizer.from_pretrained(
                    'sshleifer/distilbart-cnn-12-6')
            results = []
            sections_to_process = Utils.get_sections_from_texts(
                utterances, 900)
            for section in tqdm(sections_to_process, leave=False):
                # if debug:
                # print(section)
                inputs = tokenizer(
                    [section], max_length=1024,
                    return_tensors='pt', truncation=True
                )
                summary_ids = model.generate(
                    inputs['input_ids'],
                    bad_words_ids=bad_words_ids
                )
                results.append(
                    tokenizer.decode(
                        summary_ids[0],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )
                )
            output += 'Round {}'.format(cnt) + '\n'
            output += '. '.join(results)
            output += '\n'
            if len(sections_to_process) == 1:
                # if debug:
                # print(output)
                return output if debug else '. '.join(results)
            else:
                utterances = '. '.join(results)
            cnt += 1

    @classmethod
    def get_bart_summary(
            cls,
            turns: List[Turn],
            speaker_info: {},
            section_length: int,
            model: Optional[str] = None,
    ) -> Union[dict, str]:
        turns_segmented_by_time = [
            (f'{speaker_info[speaker_id]}', text)
            for speaker_id, text in zip([turn.speaker_id for turn in turns], [turn.text for turn in turns])
        ]
        turns_segmented_by_speaker = OrderedDict()
        for speaker_id in speaker_info.keys():
            turns_segmented_by_speaker[speaker_id] = [
                (f'{speaker_info[speaker_id]}', turn.text) for turn in turns if turn.speaker_id == speaker_id
            ]

        if model:
            return cls.get_bart_keyphrases_finetuned(
                turns_segmented_by_time, turns_segmented_by_speaker, model, section_length
            )
        else:
            return cls.get_bart_keywords_openai(turns_segmented_by_time)
