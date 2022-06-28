from typing import Tuple
from keybert import KeyBERT
from transformers import BartForConditionalGeneration, BartTokenizer
from keyphrase_vectorizers import KeyphraseCountVectorizer
from transcript_analyser.data_types.transcript import Turn
from transcript_analyser.utils.utils import *
from tqdm.auto import tqdm
import nltk


class Abstractive:

    keybert_model = None

    classifier = None

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
    ) -> str:
        turns_with_speakers_agg = [
            (f'Speaker {speaker}', text)
            for speaker, text in zip([turn.speaker_id for turn in turns], [turn.text for turn in turns])
        ]
        return cls.get_bart_keywords_openai(turns_with_speakers_agg)
