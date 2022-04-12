from distutils import text_file
from typing import List
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch as th
import networkx as nx
from sklearn.preprocessing import normalize
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

    def get_chunks(text, max_len=800):
        tokenized_sents = text.split()
        length = len(tokenized_sents)
        tokenized_sents_chunks = [' '.join(tokenized_sents[i:min(
            length, i + max_len)]) for i in range(0, length, max_len)]
        return tokenized_sents_chunks

    def get_last_hidden_state(text):
        inputs = tokenizer(text, return_tensors='pt',
                           add_special_tokens=False)
        outputs = model(**inputs)
        return outputs.last_hidden_state.squeeze().cpu().detach().numpy()

    chunks = get_chunks(text)
    last_hidden_states = np.zeros((len(chunks), 1024))
    for i, chunk in tqdm(enumerate(chunks), leave=False):
        last_hidden_states[i] = np.mean(get_last_hidden_state(chunk), axis=0)

    return np.mean(last_hidden_states, axis=0)


def get_similarity_matrix(embeddings: np.array, do_normalize: bool = True) -> np.array:
    embeddings_tensors = th.Tensor(embeddings)
    similarity_matrix = util.dot_score(
        embeddings_tensors, embeddings_tensors).numpy()
    if do_normalize:
        similarity_matrix = normalize(similarity_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    return similarity_matrix


def get_similarity_score(a, b):
    return util.dot_score(th.Tensor(a), th.Tensor(b)).numpy()


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


def get_summary_v1(text_file_path: str = None):
    """
    In this algorithm the sentence embeddings of the sentences are computed and then with the similarity between them as edges of a graph
    the most central nodes which are the central sentences are picked to generate the summary
    """
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


def get_summary_v2():
    """
    First the embedding of the whole text which consists of chunks are gathered and then all of them are fed through a AVG pooling layer.
    The same thing is done for all the sentences and then getting the similarity between sentences and the summary itself the most similar sentences are picked to be shown as the summary
    """
    from transformers import BartTokenizer, BartModel

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    model = BartModel.from_pretrained(
        "facebook/bart-large", output_hidden_states=True)

    text = ' '.join(get_utterances())
    sentences = get_sentences(text)
    text = ' '.join(sentences)

    all_embeddings = np.zeros((len(sentences) + 1, 1024))

    all_embeddings[0] = get_text_embedding(text)
    for i, sent in tqdm(enumerate(sentences), leave=False):
        all_embeddings[i + 1] = get_text_embedding(sent)

    from sklearn import preprocessing
    scaler = preprocessing.StandardScaler()

    X = scaler.fit_transform(all_embeddings)

    similarity_scores = []
    for embd in all_embeddings[1:]:
        similarity_scores.append(
            get_similarity_score(embd, all_embeddings[0])[0][0])

    similarity_scores_indices = np.argsort(similarity_scores)[::-1]
    return [sentences[i] for i in sorted(similarity_scores_indices[:10])]


if __name__ == "__main__":
    pass
