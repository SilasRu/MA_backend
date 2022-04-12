from numpy.random.mtrand import sample
from pandas.core.construction import extract_array
from numpy.linalg import norm
from tqdm import tqdm
import tensorflow_hub as hub
import pandas as pd
import numpy as np
import re
from transcript_analyser.utils.utils import *
from termcolor import colored
from nltk.tokenize import WhitespaceTokenizer
from keybert import KeyBERT
from yake import KeywordExtractor
from transformers import pipeline
import json
from pathlib import Path
from nltk.tokenize import sent_tokenize
from textsplit.tools import get_segments
from textsplit.algorithm import split_optimal
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import NMF
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from transformers import LongformerTokenizer, EncoderDecoderModel
import os
from sklearn.cluster import KMeans
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


base_path = Path(__file__).parent


def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.Module(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})


class Utterance:
    def __init__(self, transcript_df, transformers_mode=False):
        self.transcript_df = transcript_df
        self.utterances = np.array(transcript_df['Utterance'].tolist())
        self.speakers = np.array(transcript_df['Speaker'].tolist())
        self.starting_times = np.array(transcript_df['Start time'].tolist())
        self.end_times = np.array(transcript_df['End time'].tolist())
        self.durations = np.array(transcript_df['Duration (in sec)'].tolist())
        self.tokenizer = WhitespaceTokenizer()
        self.stop_words = Utils.load_stop_words()
        self.keybert_model = KeyBERT('all-MiniLM-L6-v2')
        if transformers_mode:
            use_path = (base_path / "topic_clustering").resolve()
            self.embed_fn = embed_useT(str(use_path))

    def get_utterances_by_speaker(self, speaker):
        return '. '.join(list(np.array(self.utterances)[np.where(self.speakers == speaker)[0]]))

    def get_utterances_separated_by_speaker(self):
        results = []
        for speaker in set(self.speakers):
            results.append(self.get_utterances_by_speaker(speaker))
        return results

    def get_utterances_with_least_tokens(self, tokens):
        results = []
        for utterance in self.utterances:
            if len(utterance.split()) > tokens:
                results.append(utterance)
        return results

    def get_whole_as_one(self):
        return ' . '.join(self.utterances)

    def get_yake_keywords(self, utterances, **kwargs):
        kw_extractor = KeywordExtractor(
            lan=kwargs['language'],
            n=kwargs['max_ngram_size'],
            top=kwargs['n_keyphrases'],
            dedupLim=kwargs['deduplication_threshold']
        )
        summary = kw_extractor.extract_keywords(utterances)
        return [pair[0] for pair in summary]

    def get_bart_keywords_without_limit(self, utterances):
        utterances = utterances.split()
        max_tokens = 900
        model = BartForConditionalGeneration.from_pretrained(
            "philschmid/bart-large-cnn-samsum"
        )
        tokenizer = BartTokenizer.from_pretrained(
            "philschmid/bart-large-cnn-samsum"
        )
        result = ''
        while len(utterances) != 0:
            print('length of the left utterances',
                  colored(len(utterances), 'green'))
            print('length of the summary', colored(len(result.split()), 'red'))
            summary_length = len(result.split())
            free_space = min(max_tokens - summary_length, len(utterances))
            text = result + ' '.join(utterances[:free_space])
            utterances = utterances[free_space:]
            inputs = tokenizer([text], max_length=1024,
                               return_tensors='pt', truncation=True)
            summary_ids = model.generate(inputs['input_ids'])
            result = tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
        return result

    def get_bart_keywords(self, utterances, summary_min_length, summary_max_length, permutation=False):
        model = BartForConditionalGeneration.from_pretrained(
            "philschmid/bart-large-cnn-samsum"
        )
        tokenizer = BartTokenizer.from_pretrained(
            "philschmid/bart-large-cnn-samsum"
        )
        values = list(tokenizer.decoder.values())
        import nltk
        nltk_tokenizer = nltk.RegexpTokenizer(r"\w+")
        words_in_utterances = nltk_tokenizer.tokenize(utterances)
        filtered_values = [
            word for word in values if word not in words_in_utterances]
        bad_words_ids = [tokenizer.encode(
            bad_word, add_prefix_space=True) for bad_word in filtered_values]
        result = ''
        sections_to_process = Utils.get_sections(
            utterances, 900 - summary_max_length, permutation=permutation)
        for section in tqdm(sections_to_process, leave=False):
            text = result + section
            inputs = tokenizer([text], max_length=1024,
                               return_tensors='pt', truncation=True)
            summary_ids = model.generate(
                inputs['input_ids'],
                min_length=summary_min_length,
                max_length=summary_max_length,
                bad_words_ids=bad_words_ids
            )
            result = tokenizer.decode(
                summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return result

    def get_pegasus_keywords(self, utterances):
        try:
            from transformers import PegasusForConditionalGeneration, PegasusTokenizer
            import nltk

            model_name = 'google/pegasus-xsum'
            tokenizer = PegasusTokenizer.from_pretrained(model_name)
            model = PegasusForConditionalGeneration.from_pretrained(model_name)

            cnt = 0
            # keys = list(tokenizer.get_vocab().keys())
            # nltk_tokenizer = nltk.RegexpTokenizer(r"\w+")
            # words_in_utterances = nltk_tokenizer.tokenize(utterances)
            # filtered_values = [
            # word for word in keys if word not in words_in_utterances]

            # bad_words_ids = [tokenizer.encode(bad_word, add_special_tokens=False)
            #                  for bad_word in filtered_values]
            # bad_words_ids = [
            #     entity for entity in bad_words_ids if len(entity) != 0]
            # print(len(bad_words_ids))
            # breakpoint()
            output = ''
            while True:
                results = []
                sections_to_process = Utils.get_sections(utterances, 900)
                for section in tqdm(sections_to_process, leave=False):

                    batch = tokenizer(section, truncation=True,
                                      padding='longest', return_tensors="pt")

                    outputs = model.generate(
                        batch['input_ids'],
                        #  bad_words_ids=bad_words_ids
                    )
                    results.append(tokenizer.batch_decode(
                        outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
                    # breakpoint()

                # print(results)
                output += 'Round {}'.format(cnt) + '\n'
                output += '. '.join(results)
                output += '\n'
                if len(sections_to_process) == 1:
                    return output
                else:
                    utterances = '. '.join(results)
                cnt += 1
        except Exception as e:
            print(e)
            breakpoint()

    def get_t5_keywords(self, utterances):
        from transformers import T5Tokenizer, T5ForConditionalGeneration

        tokenizer = T5Tokenizer.from_pretrained(
            "csebuetnlp/mT5_multilingual_XLSum")

        model = T5ForConditionalGeneration.from_pretrained(
            "csebuetnlp/mT5_multilingual_XLSum")

        cnt = 0
        keys = list(tokenizer.get_vocab().keys())
        import nltk
        nltk_tokenizer = nltk.RegexpTokenizer(r"\w+")
        words_in_utterances = nltk_tokenizer.tokenize(utterances)
        filtered_values = [
            word for word in keys if word not in words_in_utterances]
        bad_words_ids = [tokenizer.encode(
            bad_word, add_special_tokens=False) for bad_word in filtered_values]
        bad_words_ids = [
            entity for entity in bad_words_ids if len(entity) != 0]
        print(len(bad_words_ids))
        output = ''
        while True:
            # print(len(utterances))
            results = []
            sections_to_process = Utils.get_sections(utterances, 900)
            for section in tqdm(sections_to_process, leave=False):
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
            # print(results)
            output += 'Round {}'.format(cnt) + '\n'
            output += '. '.join(results)
            output += '\n'
            if len(sections_to_process) == 1:
                return output
            else:
                utterances = '. '.join(results)
            cnt += 1

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, index):
        return self.utterances[index]

    def get_all(self):
        return self.utterances

    def remove_stop_words(self):
        self.utterances = [' '.join([word for word in utterance.split(
        ) if word not in self.stop_words]) for utterance in self.utterances]

    def remove_punctuations(self):
        for i in range(len(self.utterances)):
            self.utterances[i] = re.sub(r'[^\w\s-]', ' ', self.utterances[i])

    def lower_words(self):
        self.utterances = [utterance.lower() for utterance in self.utterances]

    def tokenize_sentences(self):
        tokenized_utterances = self.tokenizer.tokenize_sents(self.utterances)
        self.utterances = [' '.join(utterance)
                           for utterance in tokenized_utterances]

    def preprocess(self, lowering, remove_stop_words, remove_punctuations):
        if lowering:
            self.lower_words()
        self.tokenize_sentences()
        if remove_stop_words:
            self.remove_stop_words()
        if remove_punctuations:
            self.remove_punctuations()

    def get_utterances_segmented_by_kmeans(self, model_name, n_components):

        if model_name == 'use':
            encoding_matrix = self.embed_fn(self.utterances)
        elif model_name == 'tfidf':
            tfidf_vectorizer = TfidfVectorizer()
            encoding_matrix = tfidf_vectorizer.fit_transform(self.utterances)
        else:
            print('error: please provide the model name to extract the encoding matrices')
            return
        kmeans = KMeans(n_clusters=n_components,
                        random_state=0).fit(encoding_matrix)
        result = []
        for topic in range(n_components):
            result.append('. '.join(np.array(self.utterances)[
                          np.where(kmeans.labels_ == topic)]))

        return result

    def get_utterances_segmented_by_lda_or_nmf(self, model_name, n_components):

        if model_name == 'lda':
            count_vec = CountVectorizer(
                max_df=.8, min_df=2, stop_words=self.stop_words)
        elif model_name == 'nmf':
            count_vec = TfidfVectorizer(
                max_df=.8, min_df=2, stop_words=self.stop_words)
        doc_term_matrix = count_vec.fit_transform(self.utterances)

        if model_name == 'lda':
            model = LatentDirichletAllocation(
                n_components=n_components, random_state=41)
        elif model_name == 'nmf':
            model = NMF(n_components=n_components, random_state=41)

        model.fit(doc_term_matrix)

        topic_values = model.transform(doc_term_matrix)

        utterances_topics = topic_values.argmax(axis=1)
        result = []
        for topic in list(set(utterances_topics)):
            result.append('. '.join(np.array(self.utterances)[
                          np.where(utterances_topics == topic)]))

        return result

    def textsplit(self, transcript, language, seg_limit=None, penalty=None):
        text = ""
        utterance_breaks = [0]
        for utterance in transcript["utterance"].str.lower():
            sentenced_utterance = sent_tokenize(utterance)
            utterance_breaks.append(
                utterance_breaks[-1] + len(sentenced_utterance))
            text += utterance + " "
        del utterance_breaks[0]
        word2vec_path = (
            base_path / 'googlenews-vectors-negative300.bin').resolve()
        model = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
        wrdvecs = pd.dataframe(model.vectors, index=model.index_to_key)
        del model
        sentenced_text = sent_tokenize(text)
        vecr = CountVectorizer(vocabulary=wrdvecs.index)
        sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
        optimal_segments = split_optimal(
            sentence_vectors, penalty=penalty, seg_limit=seg_limit)
        segmented_text = get_segments(sentenced_text, optimal_segments)
        lengths_optimal = [len(segment)
                           for segment in segmented_text for sentence in segment]

        normalized_splits = [0]
        for split in optimal_segments.splits:
            diff = list(map(lambda ub: split - ub, utterance_breaks))
            smallest_positive_value_index = max(
                [i for i in range(len(diff)) if diff[i] > 0])
            normalized_splits.append(smallest_positive_value_index+1)
        normalized_splits.append(len(transcript)-1)
        normalized_splits = list(set(normalized_splits))
        normalized_splits.sort()

        return normalized_splits, optimal_segments.splits, lengths_optimal

    def get_longformer_summaries(self, utterances, summary_min_length, summary_max_length, permutation=False):
        model = EncoderDecoderModel.from_pretrained(
            "patrickvonplaten/longformer2roberta-cnn_dailymail-fp16")
        tokenizer = LongformerTokenizer.from_pretrained(
            "allenai/longformer-base-4096")
        result = ''
        sections_to_process = Utils.get_sections(
            utterances, 3900 - summary_max_length, permutation=permutation)

        for section in tqdm(sections_to_process, leave=False):
            text = result + section
            inputs = tokenizer([text], max_length=4096,
                               return_tensors='pt', truncation=True)
            summary_ids = model.generate(
                inputs['input_ids'], num_beams=4, min_length=summary_min_length, max_length=summary_max_length, early_stopping=True)
            result = tokenizer.decode(
                summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return result

    def get_utterances_segments(self):
        results_len = 0
        penalty = 1
        while results_len < 5:
            print(f'penalty: {penalty}')
            try:
                normalized_splits, _, _ = self.textsplit(
                    self.transcript_df, 'english', penalty=penalty)
                results_len = len(normalized_splits)
            except Exception as e:
                print(e)
                continue
            finally:
                penalty = penalty + 1
        return normalized_splits

    def get_utterances_segmented_by_context(self):
        splits = self.get_utterances_segments()
        result = []
        for i in range(len(splits) - 1):
            result.append(' '.join(self.utterances[splits[i]:min(
                splits[i + 1] + 1, len(self.utterances))]))
        return result

    def get_agg_results_with_bart(self, text, num_runs, minimum_length, maximum_length):
        results = []
        for i in tqdm(range(num_runs), leave=False):
            results.append(self.get_bart_keywords(
                text, summary_min_length=minimum_length, summary_max_length=maximum_length, permutation=True))
        final_result = self.get_bart_keywords(' .'.join(
            results), summary_min_length=maximum_length, summary_max_length=2*maximum_length)
        return final_result

    def get_words_frequency(self, text):
        from collections import Counter
        counter = Counter(text)
        return counter.most_common(10)


def encode_sentences(text: str) -> np.array:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    # Sentences are encoded by calling model.encode()
    embeddings = model.encode(text)
    return embeddings


def normalize(vec: np.array) -> np.array:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm


def cluster_sentences(sentences: str, sent_encodings: np.array, dist_threshold: int = .6) -> np.array:
    from scipy.spatial.distance import pdist, squareform
    import networkx as nx
    import community as community_louvain
    sent_dists = squareform(pdist(sent_encodings, metric='cosine'))
    indices = np.where(sent_dists < dist_threshold)
    indices = list(zip(indices[0], indices[1]))
    indices = [index for index in indices if index[0] != index[1]]

    G = nx.Graph()
    for i, j in indices:
        G.add_edge(sentences[i], sentences[j])
    nx.write_graphml(G, 'sent_graph.graphml')
    communities = community_louvain.best_partition(G)
    output = ''
    segmented_sentences = []
    for i in range(len(set(communities.values()))):
        output += '-----' + '\n'
        output += str(i) + '\n'
        segment = '; '.join(np.array(list(communities.keys()))[
            np.where(np.array(list(communities.values())) == i)[0]])
        segmented_sentences.append(segment)
        output += segment + '\n'
        output += '-----' + '\n'
    Utils.write_keywords(output, 'louvain_community_segments.txt')
    return segmented_sentences


def check_if_contains_keywords():
    raise NotImplementedError


def use_sentence_clustering_before_summarization(data: Utterance, df: pd.DataFrame, output_file_name: str, version: int = 1):
    text = df['Utterance'].tolist()
    sent_encodings = encode_sentences(text=text)

    segmented_sentences = cluster_sentences(
        sentences=text,
        sent_encodings=sent_encodings
    )
    all_keywords = ''
    for i, segment in tqdm(enumerate(segmented_sentences), leave=False):
        all_keywords += str(i) + '\n'
        all_keywords += Utterance.get_bart_keywords_openai(segment)
        all_keywords += '------\n'
    Utils.write_keywords(
        str(all_keywords), f'{output_file_name}_V-{version}.txt')


def extract_topics_from_icsi(file_name: str) -> list:
    results = []
    id = 'description'

    def _decode_dict(a_dict):
        try:
            results.append(a_dict[id])
        except KeyError:
            pass
        return a_dict

    base_path = '/Users/user/Documents/dialogue_summarization/datasets/icsi/data'
    with open(os.path.join(base_path, 'topics', file_name + '.topic.xml.json')) as f:
        json.load(f, object_hook=_decode_dict)
        f.close()
    return results


def get_icsi_datapoints(file_name: str) -> dict:
    base_path = '/Users/user/Documents/dialogue_summarization/datasets/icsi/data'
    data_point = {}
    with open(os.path.join(base_path, 'cleaned/train', file_name + '.txt')) as f:
        # breakpoint()
        text = f.read()
        text = [line.split('\t')[1] for line in text.split('\n') if line != '']
        data_point['text'] = text
        f.close()

    data_point['topics'] = extract_topics_from_icsi(file_name)
    return data_point


if __name__ == '__main__':
    file_name = 'Bed002'
    data = get_icsi_datapoints(file_name=file_name)
    text = data['text']
    text = '; '.join(text)
    topics = data['topics']
    df = pd.read_csv(
        'transcriptsforkeyphraseextraction/2021-07-12 14.35.38 interscriber wrapup.m4a.csv')
    data = Utterance(df)
    first_pass_keywords = data.get_keybert_keywords(text)
    print(colored('First pass', 'red'))
    print(first_pass_keywords)
    second_pass_keywords = data.get_keybert_keywords(
        first_pass_keywords, keyphrase_ngram_range=(1, 3), top_n=1)
    print(colored('Second pass', 'red'))
    print(second_pass_keywords)
    print(colored('Actual topics', 'green'))
    print(topics)
