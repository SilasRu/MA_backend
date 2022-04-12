from typing import List, Tuple, Set
from sentence_transformers.SentenceTransformer import SentenceTransformer
from transformers import pipeline
import pandas as pd
import numpy as np
import re
import os
import json
import string
from transcript_analyser.utils.consts import backchannel_consts
from sklearn.metrics.pairwise import cosine_similarity
from transcript_analyser.utils.network_backbone import *
import networkx as nx
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize


class Utils:
    classifier = None
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def __init__(self) -> None:
        pass

    @staticmethod
    def sort_json_by_property(json_obj: List[dict], property: str) -> List:
        return sorted(json_obj, key=lambda x: x[property], reverse=True)

    @staticmethod
    def text2df(text: str) -> pd.DataFrame:
        df = pd.DataFrame()
        sentences = sent_tokenize(text)
        for sent in sentences:
            df = df.append({
                'Utterance': sent,
                'Speaker': None
            }, ignore_index=True)
        return df

    @staticmethod
    def sentence_to_wordlist(raw: str):
        clean = re.sub("[^a-zA-Z]", " ", raw)
        words = [word for word in clean.split() if len(word) > 2]
        return words

    @classmethod
    def load_classifier(cls):
        if not cls.classifier:
            cls.classifier = pipeline("zero-shot-classification",
                                      model="facebook/bart-large-mnli")

    @classmethod
    def get_classes_scores(cls, utterance: str, labels: List[str], threshold: float = 0.5):
        cls.load_classifier()
        related_labels = []
        results = cls.classifier(utterance, labels, multi_label=True)
        for label, score in zip(results['labels'], results['scores']):
            if score > threshold:
                related_labels.append(label)
        return related_labels

    @staticmethod
    def load_stop_words() -> Set[str]:
        import spacy
        import nltk

        sp = spacy.load('en_core_web_sm')
        spacy_stopwords = sp.Defaults.stop_words
        en_stop = set(nltk.corpus.stopwords.words('english'))
        dirname = os.path.dirname(__file__)
        filename = 'stopwords.txt'
        with open(os.path.join(dirname, filename), 'r') as f:
            stop_words = set(f.read().strip().split('\n'))
            f.close()
        stop_words = set.union(en_stop, stop_words)
        stop_words = set.union(set(spacy_stopwords), stop_words)
        return stop_words

    @classmethod
    def find_backchannel(cls, sentences: List[str], type='default', model_name='all-MiniLM-L6-v2'):
        """
        For utterance, tag if it a backchannel or not.
        Supports two methods to do this, by default - it uses a dictionary of known backchannels to identify(fast, less precise.)
        also supports a slightly better/more precise way of identifying backchannels via language-model. (set type='nlp')

        Parameters
        ----------
        type: str, default or nlp
            type of back channel detection.

        model_name: str 
            Pass the any sentence transfromer model name which use similary when backchannel type is nlp

        Returns
        -------
        questions: pd.dataframe or pd.Series
            Returns the dataframe or series 
        """
        if type == 'default':
            backchannel = [cls._is_backchannel(sent) for sent in sentences]

        elif type == "nlp":
            backchannel = cls._nlp_backchannel(sentences, model_name)

        else:
            raise ValueError(
                "Please pass backchannel either as `default` or `nlp`")

        return backchannel

    @staticmethod
    def remove_punct(text: str):
        punct_list = re.compile('[%s]' % re.escape(string.punctuation))
        text = re.sub(punct_list, ' ', text)
        text = re.sub("  +", ' ', text)
        return text

    @classmethod
    def _is_backchannel(cls, text):
        is_back_ch = False
        clean_text = cls.remove_punct(text)

        if text in backchannel_consts or text.lower() in backchannel_consts:
            is_back_ch = True

        if clean_text in backchannel_consts or clean_text.lower() in backchannel_consts:
            is_back_ch = True

        return is_back_ch

    @classmethod
    def _nlp_backchannel(cls, utterances, model='all-MiniLM-L6-v2'):
        if isinstance(utterances, str):
            utterances = [utterances]

        return_list = [False]*len(utterances)
        back_ch_vect = cls.sentence_transformer.encode(backchannel_consts)
        back_ch_vect = [np.mean(back_ch_vect, axis=0)]
        utterance_vect = cls.sentence_transformer.encode(utterances)
        sim = cosine_similarity(back_ch_vect, utterance_vect)[0]
        for idx, i in enumerate(sim):
            if i >= 0.55 or abs(i-0.55) < 0.05:
                return_list[idx] = True
        return return_list

    @staticmethod
    def write_keywords(keywords: str, file_name: str):
        """
        Write the keywords in String on the file with name (file_name)
        """
        folder_name = 'results'
        with open(os.path.join(folder_name, file_name), 'w') as f:
            f.write(keywords)
            f.close()

    @staticmethod
    def check_version(file_path):
        previous_versions = os.listdir('results')
        previous_versions = [entity for entity in previous_versions if entity.startswith(
            file_path
        )]
        return len(previous_versions)

    @staticmethod
    def get_sections(text: str, num_words: int, permutation: bool = False):
        text = text.split()
        result = np.arange(0, len(text), num_words)
        if result[-1] != len(text):
            result = np.append(result, len(text))
        sections = []
        for index in range(len(result) - 1):
            sections.append(' '.join(text[result[index]:result[index + 1]]))
        if permutation:
            sections = np.random.permutation(np.array(sections))
        return sections

    @staticmethod
    def get_graph_backbone(G: nx.Graph, alpha: float = 0.3):
        """
            Get the back bone of graph using disparity filter
        """
        if G.number_of_edges() < 30:
            return G
        print(
            f'Number of nodes before applying backboning {G.number_of_nodes()}\n',
            f'Number of edges before applying backboning {G.number_of_edges()}',
        )
        nodes_attributes = dict(G.nodes(data=True))
        G = disparity_filter(G)
        G = nx.Graph([(u, v, d)
                      for u, v, d in G.edges(data=True) if d['alpha'] < alpha])
        nx.set_node_attributes(G, values=nodes_attributes)
        print(
            f'Number of nodes after applying backboning {G.number_of_nodes()}\n',
            f'Number of edges after applying backboning {G.number_of_edges()}',
        )
        return G

    @staticmethod
    def pretty_print_json(json_obj: dict):
        print(json.dumps(json_obj, indent=4, sort_keys=True))

    @staticmethod
    def rouge_evaulate(hypothese: List[str] or str, references: List[str] or str) -> dict:
        scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rouge3', 'rouge4', 'rougeL'])
        scores = scorer.score(hypothese, references)
        return scores

    @staticmethod
    def contains(inner_interval: Tuple[float], start_times: List[float], end_times: List[float]) -> bool:

        for start_time, end_time in zip(start_times, end_times):
            if start_time <= inner_interval[0] and inner_interval[1] <= end_time:
                return True
        return False


if __name__ == "__main__":
    print('Calling from the utils.py')
