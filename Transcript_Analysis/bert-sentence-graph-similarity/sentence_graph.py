from distutils import text_file
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch as th
import networkx as nx
from sklearn.preprocessing import normalize
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm


def get_utterances(
    path: str = '/Users/user/Documents/dialogue_summarization/transcriptsforkeyphraseextraction/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv'
) -> List[str]:
    df = pd.read_csv(path)
    return df['Utterance'].tolist()


def get_sentences(
    text: str,
    min_sent_length: int = 20
) -> List[str]:
    sentences = sent_tokenize(text)
    sentences = [sent for sent in sentences if len(
        sent.split()) > min_sent_length]
    return sentences


def get_sentence_embedding(sentences: List[str] or str) -> np.array:
    checkpoint = 'sentence-transformers/all-mpnet-base-v2'
    model = SentenceTransformer(checkpoint)
    embeddings = model.encode(sentences=sentences)
    return embeddings


def get_text_embedding(text: str):

    from transformers import BartTokenizer, BartModel

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartModel.from_pretrained(
        "facebook/bart-large", output_hidden_states=True)

    def get_chunks(text, max_len=800):
        tokenized_sents = text.split()
        length = len(tokenized_sents)
        tokenized_sents_chunks = [' '.join(tokenized_sents[i:min(
            length, i + max_len)]) for i in range(0, length, max_len)]
        return tokenized_sents_chunks

    def get_last_hidden_state(text):
        inputs = tokenizer(text, return_tensors='pt',
                           add_special_tokens=False, padding='max_length')
        outputs = model(**inputs)
        return outputs.last_hidden_state.squeeze().cpu().detach().numpy()

    chunks = get_chunks(text)
    last_hiddent_states = np.zeros((len(chunks), 1024, 1024))
    for i, chunk in tqdm(enumerate(chunks)):
        last_hiddent_states[i] = get_last_hidden_state(chunk)

    mean_hidden_state = np.mean(np.mean(last_hiddent_states, axis=1), axis=0)
    return mean_hidden_state


def get_similarity_matrix(embeddings: np.array, do_normalize: bool = True) -> np.array:
    embeddings_tensors = th.Tensor(embeddings)
    similarity_matrix = util.dot_score(
        embeddings_tensors, embeddings_tensors).numpy()
    if do_normalize:
        similarity_matrix = normalize(similarity_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix


def create_sentences_graph(similarity_matrix: np.array, embeddings: np.array) -> nx.DiGraph:
    g = nx.DiGraph()
    for node_i in range(similarity_matrix.shape[0]):
        for node_j in range(node_i + 1, similarity_matrix.shape[0]):
            g.add_edge(node_i, node_j,
                       weight=similarity_matrix[node_i, node_j])
    nodal_attributes = {
        k: embeddings[k] for k in range(embeddings.shape[0])
    }
    nx.set_node_attributes(g, nodal_attributes, 'embedding')
    return g


def get_tf_idf_keywords(sentences):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(min_df=3, stop_words='english')
    X = vectorizer.fit_transform(sentences)
    return X


def get_graph_centralities(g: nx.DiGraph): pass


def get_sentence_scores(g: nx.DiGraph, lambda_1: float = 1, lambda_2: float = 1) -> List[float]:
    sentence_scores = []
    for node_i in range(g.number_of_nodes()):
        score = 0
        in_edges = g.in_edges(node_i, data=True)
        for _, _, data in in_edges:
            score += data['weight'] * lambda_1
        out_edges = g.out_edges(node_i, data=True)
        for _, _, data in out_edges:
            score += data['weight'] * lambda_2
        sentence_scores.append(score)
    return sentence_scores


def get_top_sentences(sentence_scores: List[float], num_sentences: int = 5) -> List[int]:
    top_indices = np.argsort(sentence_scores)[::-1][:num_sentences]
    return sorted(top_indices)


def get_summary(text_file_path: str = None):
    if text_file_path != None:
        utterances = get_utterances(text_file_path)
    else:
        utterances = get_utterances()
    text = ' '.join(utterances)
    sentences = get_sentences(text)
    embeddings = get_sentence_embedding(sentences)
    similarity_matrix = get_similarity_matrix(embeddings)
    g = create_sentences_graph(
        similarity_matrix=similarity_matrix, embeddings=embeddings)
    sentence_scores = get_sentence_scores(g)
    sorted_indices = get_top_sentences(sentence_scores)

    summary = '\n'.join([sentences[index] for index in sorted_indices])
    return summary


if __name__ == "__main__":
    # get_summary()

    utterances = get_utterances()
    text = ' '.join(utterances)
    text = ' '.join(get_sentences(text))

    text_embedding = get_text_embedding(text)
    breakpoint()
