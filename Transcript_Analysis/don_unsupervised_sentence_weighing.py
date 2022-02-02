from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import spacy
import torch as th
from community.community_louvain import best_partition
from kneed.knee_locator import KneeLocator
from sklearn.cluster import KMeans


from Transcript_Analysis.utils.utils import *
from keybert import KeyBERT


class DialogueTranscript:

    def __init__(self, csv_file: str, source_dataframe: pd.DataFrame = None, source: str = 'interscriber', lang: str = 'en', keyword_pos: Tuple = ('NOUN',)):
        self.keyword_pos = keyword_pos
        self.lang = lang
        self.keyword_counts: Counter[str] = Counter()
        if source == 'interscriber':
            self.df: pd.DataFrame = pd.read_csv(
                csv_file) if csv_file != None else source_dataframe
        elif source == 'icsi':
            self.df: pd.DataFrame = pd.read_csv(
                csv_file, names=['Speaker', 'Utterance'], sep='\t') if csv_file != None else source_dataframe
        else:
            raise NotImplementedError
        self.nlp = spacy.load('en_core_web_sm')  # TODO lang-specific model
        self.df['parse'] = None
        self.sentences: List[str] = list()
        self.load_sentences()
        self.sentence_similarity_graph: Optional[nx.Graph] = None
        self.sentence_similarity: Optional[np.array] = None
        self.sentence_vectors: Optional[list] = list()
        self.sentence_weights: Optional[list] = list()
        self.sentences_communities: Optional[dict[Any, Any]] = None
        self.entailment_matrix: Optional[np.array] = None
        self.entailment_graph: Optional[nx.DiGraph] = None
        self.sentences_graph: Optional[nx.Graph] = None
        self.sentence_cluster_ids: Optional[np.array] = None
        # TODO switch to multilingual sentence transformer as base model?
        self.kw_model = None

    def load_sentences(self):
        self.sentences = []
        for i, turn in self.df.iterrows():
            self.df.at[i, 'parse'] = self.nlp(turn['Utterance'])
            for sent in self.df.at[i, 'parse'].sents:
                self.sentences.append(sent.text)

    def filter_backchannels(self):
        """
        Computes the backchannels of the utterances
        """
        back_channel_sentences_cnt = 0
        for i, turn in self.df.iterrows():
            proper_turn_sentences = []
            for sent in self.nlp(turn['Utterance']).sents:
                if not Utils._nlp_backchannel(sent.text)[0]:
                    proper_turn_sentences.append(sent.text)
                else:
                    back_channel_sentences_cnt += 1
            self.df.at[i, 'Utterance'] = ' '.join(proper_turn_sentences)
        self.load_sentences()
        print(
            f'Number of back channels in the sentences {back_channel_sentences_cnt}')

    def count_keywords(self, max_np_len: int = 5, min_count: int = 0):
        tok_counts, mwt_counts = Counter(), Counter()
        for i, turn in self.df.iterrows():
            for sent in turn['parse'].sents:
                np_aggregates = list()
                for tok in sent:
                    new_np_aggregates = list()
                    for np_aggregate in np_aggregates:
                        if len(np_aggregate) > max_np_len:
                            continue
                        np_aggregate.append(tok.lemma_)
                        new_np_aggregates.append(np_aggregate)
                        if tok.pos_ in self.keyword_pos:
                            if len(np_aggregate) > 1:
                                mwt_counts[tuple(np_aggregate)] += 1
                    if tok.pos_ in self.keyword_pos:
                        tok_counts[tok.lemma_] += 1
                        new_np_aggregates.append([tok.lemma_])
                    np_aggregates = new_np_aggregates
        """
        # This filters tokens that are part of multi-word terms with equal frequency.
        keyword_counts = Counter()
        for tok, cnt in tok_counts.items():
            if cnt >= min_count:
                mwts = Counter({mwt: c for mwt, c in mwt_counts.items() if tok in mwt})
                for mwt, c in mwts.most_common():
                    if c >= cnt:
                        break
                else:
                    keyword_counts[tok] = cnt
        """
        keyword_counts = tok_counts
        for mwt, cnt in mwt_counts.items():
            if cnt >= min_count:
                keyword_counts[' '.join(mwt)] = cnt
        self.keyword_counts = keyword_counts

    def print_sentences_length(self):
        num_sentences = 0
        for i, turn in self.df.iterrows():
            for _ in turn['parse'].sents:
                num_sentences += 1
        print(len(self.sentences), num_sentences)

    def weigh_sentences(self):
        """ Weigh sentences based on sum of count values of keywords """
        if not self.keyword_counts:
            self.count_keywords()
        self.df['sentence_weights'] = None
        for i, turn in self.df.iterrows():
            self.df.at[i, 'sentence_weights'] = list()
            for sent in turn['parse'].sents:
                keyword_lemmas = [
                    tok.lemma_ for tok in sent if tok.pos_ in self.keyword_pos]
                score = sum([self.keyword_counts[noun]
                            for noun in set(keyword_lemmas)])
                self.df.at[i, 'sentence_weights'].append(score)
                self.sentence_weights.append(score)

    def vectorize_sentences_with_keywords(self):
        """ Vectorize based on keyword occurrence """
        if not self.keyword_counts:
            self.count_keywords()
        self.df['sentence_vectors'] = None
        keyword_vocab_ix = {k: i for i, k in enumerate(
            self.keyword_counts.keys())}
        vec_size = len(keyword_vocab_ix)
        for i, turn in self.df.iterrows():
            self.df.at[i, 'sentence_vectors'] = list()
            for sent in turn['parse'].sents:
                sent_vec = np.zeros(vec_size, dtype=int)
                for tok in sent:
                    if tok.lemma_ in self.keyword_counts and tok.pos_ in self.keyword_pos:
                        sent_vec[keyword_vocab_ix[tok.lemma_]
                                 ] = self.keyword_counts[tok.lemma_]
                self.df.at[i, 'sentence_vectors'].append(sent_vec)
                self.sentence_vectors.append(sent_vec)

    def calculate_sentence_similarity(self):
        """ Sentence graph based on similarity and sentence weight
        We simply sum the scores of the shared keywords to determine sentence similarity """
        if 'sentence_vectors' not in self.df.columns:
            self.vectorize_sentences_with_keywords()
        if 'sentence_weights' not in self.df.columns:
            self.weigh_sentences()
        self.sentence_similarity = np.zeros(
            (len(self.sentences), len(self.sentences))
        )
        for i, sent_vec1 in enumerate(self.sentence_vectors):
            if np.sum(sent_vec1) == 0:  # No keywords
                continue
            for j, sent_vec2 in enumerate(self.sentence_vectors[i + 1:]):
                if np.sum(sent_vec2) == 0:
                    continue
                shared_keywords_ixs = np.where(sent_vec1 == sent_vec2)[0]
                shared_keywords_scores = np.take(
                    sent_vec1, shared_keywords_ixs)
                sim = np.sum(shared_keywords_scores)
                self.sentence_similarity[i][i + 1 + j] = sim
                self.sentence_similarity[i + 1 + j][i] = sim

    def generate_sentence_similarity_graph(self, sim_thresh: float = 0.75, max_only: bool = False):
        if not self.sentence_similarity:
            self.calculate_sentence_similarity()
        g = nx.Graph()
        for i, sim_vec in enumerate(self.sentence_similarity):
            if np.sum(sim_vec) == 0:  # no keywords in sent
                continue
            # -2 to not take sim with self
            nearest_sent_ix = np.argsort(sim_vec)[-2]
            highest_sim = sim_vec[nearest_sent_ix]
            if highest_sim == 0:  # shouldn't happen
                continue
            if highest_sim < sim_thresh or max_only:
                g.add_edge(
                    self.sentences[i], self.sentences[nearest_sent_ix], weight=highest_sim)
            else:
                for j, sim in enumerate(sim_vec[i+1:]):
                    if sim >= sim_thresh:
                        g.add_edge(
                            self.sentences[i], self.sentences[i+1+j], weight=sim)
        for i, sent in enumerate(self.sentences):  # add sentence weights to nodes
            if sent in g:
                g.nodes()[sent]['weight'] = self.sentence_weights[i]
        self.sentence_similarity_graph = g

    def cluster_sentences(self, num_clusters: int = None, ks: tuple = (2, 20)):
        if self.sentence_similarity is None:
            self.calculate_sentence_similarity()
        if not num_clusters:
            sse: dict[int, float] = dict()  # Sum of squared errors
            # Store fitted kmeans estimators
            kmeans_fits: dict[int, Any] = dict()
            for k in range(ks[0], ks[1] + 1):
                kmeans_fits[k] = KMeans(n_clusters=k, max_iter=1000).fit(
                    self.sentence_vectors)
                # Sum of squared distances of samples to their closest cluster center
                sse[k] = kmeans_fits[k].inertia_
            kl = KneeLocator(list(kmeans_fits.keys()), list(
                sse.values()), curve="convex", direction='decreasing')
            kmeans = kmeans_fits[kl.knee]
        else:
            kmeans = KMeans(n_clusters=num_clusters, max_iter=1000).fit(
                self.sentence_vectors)
        self.sentence_cluster_ids = kmeans.labels_

    def write_clustered_sentences_w_keywords(self):
        self.kw_model = KeyBERT()
        if self.sentence_cluster_ids is None:
            self.cluster_sentences()
        clustered_sentences = defaultdict(list)
        for cluster_id, sent in zip(self.sentence_cluster_ids, self.sentences):
            clustered_sentences[cluster_id].append(sent)

        for cluster_id, sents in clustered_sentences.items():
            cluster_weight = np.mean(
                [self.sentence_weights[self.sentences.index(sent)] for sent in sents])
            print('cluster weight', cluster_weight)
            print('cluster size', len(sents))
            keyword_counts = Counter([t.lemma_ for t in self.nlp(
                ' '.join(sents)) if t.pos_ in self.keyword_pos])
            print(keyword_counts.most_common(5))
            keywords = self.kw_model.extract_keywords(
                ' '.join(sents), keyphrase_ngram_range=(1, 3), top_n=5)
            print(keywords)
            print(sents)
            breakpoint()

    def keywords_per_sentence_cluster(self, num_keywords: int = 25):
        if self.sentence_cluster_ids is None:
            self.cluster_sentences()
        texts_per_cluster = defaultdict(str)
        for sent, cluster_id in zip(self.sentences, self.sentence_cluster_ids):
            texts_per_cluster[cluster_id] += ' ' + sent
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), stop_words='english')
        cluster_vecs = tfidf.fit_transform(
            list(texts_per_cluster.values())).toarray()
        word_list = tfidf.get_feature_names()
        for clust_id, vec in zip(texts_per_cluster.keys(), cluster_vecs):
            best_ixs = np.argsort(vec)
            best_ixs = np.flip(best_ixs)
            print(clust_id, np.take(word_list, best_ixs[:num_keywords]))

    def best_sentences_per_cluster(self, num_sent: int = 5):
        if self.sentence_weights is None:
            self.weigh_sentences()
        if self.sentence_cluster_ids is None:
            self.cluster_sentences()
        sent_weights = {sent: weight for sent, weight in zip(
            self.sentences, self.sentence_weights)}
        sents_per_cluster = defaultdict(list)
        for sent, cluster_id in zip(self.sentences, self.sentence_cluster_ids):
            sents_per_cluster[cluster_id].append((sent_weights[sent], sent))
        for cluster_id, sents in sents_per_cluster.items():
            print(cluster_id, len(sents))
            for sent in sorted(sents, reverse=True)[:num_sent]:
                print(sent)
            print()

    def load_sentence_weights_to_graph(self):
        if self.sentences_graph is None:
            self.construct_sentences_graph()
        if self.sentence_weights is None:
            self.weigh_sentences()
        sentence_weight_mapping = {
            self.sentences[i]: self.sentence_weights[i] for i in range(len(self.sentences))}
        nx.set_node_attributes(self.sentences_graph, sentence_weight_mapping,
                               'sentence_weight')

    def construct_sentences_graph(self):
        """
        Construct the graph of sentences based on the similarity they have
        """
        if self.sentence_similarity is None:
            self.calculate_sentence_similarity()
        self.sentences_graph = nx.Graph()
        for i in range(self.sentence_similarity.shape[0]):
            for j in range(i + 1, self.sentence_similarity.shape[1]):
                if self.sentence_similarity[i, j] != 0:
                    self.sentences_graph.add_edge(
                        self.sentences[i], self.sentences[j], weight=self.sentence_similarity[i, j])

    def cluster_sentences_by_louvain(self, G: nx.Graph):
        """
        Cluster the sentences with Louvain community detection algorithm based on the sentence similarities.
        """
        self.sentences_communities = best_partition(G)
        print(
            f'Number of partitions in the network {len(set(self.sentences_communities.values()))}')
        nx.set_node_attributes(G,
                               self.sentences_communities, 'community')

    @staticmethod
    def _load_numpy_object(file_name: str):
        if os.path.exists(file_name):
            with open(file_name, 'rb') as f:
                array = np.load(f)
                f.close()
            return array
        else:
            return None

    @staticmethod
    def _store_numpy_object(array: np.array, file_name: str):
        with open(file_name, 'wb') as f:
            np.save(f, array)
            f.close()

    def compute_entailment_matrix(self, threshold: float = 0.5, npy_file: str = 'entailment_matrix.npy'):
        """
        Computing the entailment matrix between pairs of sentences
        """
        self.entailment_matrix = DialogueTranscript._load_numpy_object(
            npy_file)
        if self.entailment_matrix:
            return
        # to check (smaller models): https://huggingface.co/typeform/distilbert-base-uncased-mnli
        # https://huggingface.co/valhalla/distilbart-mnli-12-9
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-base-mnli")
        model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-base-mnli")

        if self.sentence_similarity is None:
            self.calculate_sentence_similarity()

        num_sentences = len(self.sentences)
        entailment_matrix = np.zeros([num_sentences, num_sentences])
        G = nx.DiGraph()
        for i, sent_a in enumerate(self.sentences):
            if self.sentence_weights[i] == 0:
                continue
            for j, sent_b in enumerate(self.sentences):
                if i == j or self.sentence_weights[j] == 0 or self.sentence_similarity[i, j] < 2 \
                        or len(sent_a) < len(sent_b):
                    continue
                print(f'\r{i} {j}', end='')
                sentence = '[CLS]' + sent_a + '[SEP]' + sent_b + '[SEP]'
                inputs = tokenizer(
                    sentence, padding='max_length', return_tensors='pt')
                outputs = model(**inputs)
                logits = outputs.logits.detach().numpy()
                probs = th.nn.Softmax(dim=1)(th.tensor(logits))
                entailment_prob = probs[0][2].numpy()
                # if entailment_prob > threshold:
                if int(np.argmax(probs.numpy())) == 2:
                    entailment_matrix[i, j] = float(entailment_prob)
                    G.add_edge(sent_a, sent_b, weight=float(entailment_prob))
        self.entailment_matrix = entailment_matrix
        self.entailment_graph = G
        DialogueTranscript._store_numpy_object(
            self.entailment_matrix, npy_file)

    def remove_nodes_based_on_entailment_matrix_and_similarity_graph(self):
        if self.sentences_graph == None:
            self.construct_sentences_graph()
        if np.any(self.entailment_matrix) == None:
            self.simple_entailment()
        sentence2index = {sent: i for i, sent in enumerate(self.sentences)}

        edges = self.sentences_graph.edges(data=True)

        entailment_matrix = self.entailment_matrix

        nodes_to_remove = list()
        non_removable_nodes = list()
        for sent_a, sent_b, _ in edges:
            if entailment_matrix[sentence2index[sent_a], sentence2index[sent_b]] == 1 and sent_b not in non_removable_nodes:
                nodes_to_remove.append(sent_b)
                non_removable_nodes.append(sent_a)
        print(
            f'Number of nodes before applying removing sentences using entailment matrix {self.sentences_graph.number_of_nodes()}',
            f'Number of edges before applying removing sentences using entailment matrix {self.sentences_graph.number_of_edges()}',
        )
        self.sentences_graph.remove_nodes_from(nodes_to_remove)
        print(
            f'Number of nodes after applying removing sentences using entailment matrix {self.sentences_graph.number_of_nodes()}',
            f'Number of edges after applying removing sentences using entailment matrix {self.sentences_graph.number_of_edges()}',
        )

    def simple_entailment(self, threshold: float = 0.85):
        """ Simply regard entailment as the overlap of keywords.
        If sent 1 has all keywords of sent 2 and more, sent 1 entails sent 2."""
        if self.sentence_similarity is None:
            self.calculate_sentence_similarity()
        entailment_graph = nx.DiGraph()
        num_sentences = len(self.sentences)
        self.entailment_matrix = np.zeros([num_sentences, num_sentences])
        for i, sent_weight1 in enumerate(self.sentence_weights):
            for j, sent_weight2 in enumerate(self.sentence_weights[i + 1:]):
                real_j_ix = i + j + 1
                if self.sentence_similarity[i, real_j_ix] > 0:
                    # how much of the weight is explained by the similarity
                    entailment_score_sent1 = self.sentence_similarity[i,
                                                                      real_j_ix] / sent_weight1
                    entailment_score_sent2 = self.sentence_similarity[i,
                                                                      real_j_ix] / sent_weight2
                    if entailment_score_sent2 > threshold:
                        entailment_graph.add_edge(
                            self.sentences[i], self.sentences[real_j_ix], weight=entailment_score_sent1)
                        self.entailment_matrix[i, real_j_ix] = 1
                    if entailment_score_sent1 > threshold:
                        entailment_graph.add_edge(
                            self.sentences[real_j_ix], self.sentences[i], weight=entailment_score_sent2)
                        self.entailment_matrix[real_j_ix, i] = 1

    def write_dataframe_with_weight_community_html(self):
        self.kw_model = KeyBERT()
        html_output = ""
        num_communities = len(set(self.sentences_communities.values()))
        colors = np.random.rand(num_communities, 3)
        colors *= 256
        colors = colors.astype('int')
        normalized_sentence_weights = np.array(
            self.sentence_weights) / max(self.sentence_weights)
        sentence_weight_mapping = {
            self.sentences[i]: normalized_sentence_weights[i]
            for i in range(len(self.sentences))
        }
        keywords = []
        for community in set(self.sentences_communities.values()):
            community_sentences = '; '.join([
                sent for sent, sent_att in self.sentences_graph.nodes(data=True) if sent_att['community'] == community])
            keywords.append([word[0] for word in self.kw_model.extract_keywords(
                community_sentences, keyphrase_ngram_range=(1, 2), top_n=3)])

        html_legend = """
        <div class="legend">
        """
        for i in range(num_communities):
            html_legend += """<div style="margin: 5px 0;">\n"""
            html_legend += f"""
                    <div class="community" style="color: rgb{str(tuple(colors[i]))};">
                        community {i + 1}
                    </div>
            """
            html_legend += "<div>\n"
            for keyword in keywords[i]:
                html_legend += f"""<span style="color: rgb{str(tuple(colors[i]))}
                ">{keyword}, </span>"""
            html_legend += "</div>\n"
            html_legend += "</div>\n"
        html_legend += "</div>\n"
        graph_nodes_sentences = [sent for sent,
                                 _ in self.sentences_graph.nodes(data=True)]

        html_output += """
        <style>
            .legend {
                transition-property: opacity;
                transition-duration: .2s;
                width: 200px;
                display: flex;
                justify-content: center;
                align-content: center;
                padding: 23px;
                background: gray;
                position: fixed;
                right: 20px;
                top: 10%;
                opacity: 50%;
                border-radius: 8px;
                flex-direction: column;
            }

            .community::first-letter {
                font-size: 30px;
            }

            .community {
                font-size: 20px;
            }


            .legend:hover {
                opacity: 100%;
            }
        </style>
        """
        html_output += '<html>\n'
        html_output += html_legend
        html_output += '<table border=1>\n'
        for _, turn in self.df.iterrows():
            html_output += '<tr><td>' + str(turn['Speaker']) + '</td><td>'
            for sent in self.nlp(turn['Utterance']).sents:
                if sent.text not in graph_nodes_sentences:
                    font = 11
                else:
                    font = max(sentence_weight_mapping[sent.text] * 20, 16)
                if sent.text not in graph_nodes_sentences:
                    color = (169, 169, 169)
                else:
                    color = tuple(
                        colors[self.sentences_communities[sent.text]])
                html_output += f'<span style="font-size: {str(int(font))}; color: rgb{str(color)};">' + \
                    sent.text + ' </span>'
            html_output += '</td></tr>\n'
        html_output += '</table></html>'
        with open('weighted_sentences_per_speaker.html', 'w') as f:
            f.write(html_output)
            f.close()
        return html_output


def main(csv_file: str, source: str = 'interscriber', lang: str = 'EN'):
    dt = DialogueTranscript(csv_file=csv_file, source=source, lang=lang)
    dt.count_keywords()
    # dt.write_clustered_sentences_w_keywords()
    """
    dt.construct_sentences_graph()
    dt.cluster_sentences_by_louvain(G=dt.sentences_graph)
    dt.write_dataframe_with_weight_community_html()
    """
    # dt.count_noun_chunks()
    # dt.simple_entailment()
    # dt.generate_sentence_similarity_graph(max_only=False)
    # nx.write_graphml(dt.sentence_similarity_graph, 'sent_sim.graphml')
    # dt.construct_sentences_graph()
    # dt.compute_entailment_matrix()
    # nx.write_graphml(dt.entailment_graph, 'sent_entailment2.graphml')
    # dt.cluster_sentences()


if __name__ == '__main__':
    main(
        csv_file='../transcriptsforkeyphraseextraction/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv',
        source='interscriber'
        # csv_file = '../US Election Debates/us_election_2020_2nd_presidential_debate.csv',
        # source='interscriber'
    )
