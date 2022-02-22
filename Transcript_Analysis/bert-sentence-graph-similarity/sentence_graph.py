from tracemalloc import stop
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch as th
import networkx as nx
from sklearn.preprocessing import normalize
from rouge_score import rouge_scorer


def get_text(
    path: str = '/Users/user/Documents/dialogue_summarization/transcriptsforkeyphraseextraction/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv'
) -> List[str]:
    df = pd.read_csv(path)
    return df['Utterance'].tolist()


def get_sentence_embedding(sentences: List[str]) -> np.array:
    checkpoint = 'sentence-transformers/all-mpnet-base-v2'
    model = SentenceTransformer(checkpoint)
    embeddings = model.encode(sentences=sentences)
    return embeddings


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


def get_max_flow(g: nx.DiGraph): raise NotImplementedError


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
    top_indices = np.argsort(sentence_scores)[:num_sentences]
    return sorted(top_indices)


reference_summary = 'Templates Brief Summary of the dialogues Voice activity detection Interscriber  Dynatrace User behavior Replay the session of user Swiss-german Demo Exhibition in September Increasing estimated transcription time Preparing for the industry day Transcript explorer Expensive invoice Someone to generate the summaries Unexpected different stuff in cloud computing exam Rule-based system for looking at particular segments of the meeting in more detail Detect main decisions in a meeting'

if __name__ == "__main__":
    results_df = pd.DataFrame()
    scorer = rouge_scorer.RougeScorer(
        ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    sentences = get_text()
    embeddings = get_sentence_embedding(sentences)
    similarity_matrix = get_similarity_matrix(embeddings)
    g = create_sentences_graph(
        similarity_matrix=similarity_matrix, embeddings=embeddings)
    values_for_lambda_1 = np.linspace(start=-1, stop=1, num=10)
    values_for_lambda_2 = np.linspace(start=-1, stop=1, num=10)
    for lambda_1 in values_for_lambda_1:
        for lambda_2 in values_for_lambda_2:
            sentence_scores = get_sentence_scores(
                g, lambda_1=lambda_1, lambda_2=lambda_2)
            sorted_indices = get_top_sentences(sentence_scores)
            summary = '. '.join([sentences[index] for index in sorted_indices])
            scores = scorer.score(reference_summary, summary)
            results_df = results_df.append({
                "lambda_1": lambda_1,
                "lambda_2": lambda_2,
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure,
            }, ignore_index=True)
    results_df.sort_values(by=['rouge1', 'rouge2', 'rougeL'], inplace=True)
    results_df.to_csv('results.csv', index=False)
