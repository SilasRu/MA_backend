from collections import Counter, defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
import spacy
from community.community_louvain import best_partition
from kneed.knee_locator import KneeLocator
from sklearn.cluster import KMeans
from transcript_analyser.data_types.transcript import Transcript
from transcript_analyser.data_types.general import *

from transcript_analyser.utils.utils import *
from keybert import KeyBERT
from tqdm.auto import tqdm
from nltk.tokenize import sent_tokenize


class UnsupervisedSummarizer:

    def __init__(self, csv_file: str, source_dataframe: pd.DataFrame = None, lang: str = 'en', keyword_pos: Tuple = ('NOUN',)):
        self.keyword_pos = keyword_pos
        self.lang = lang
        self.keyword_counts: Counter[str] = None
        self.df: pd.DataFrame = pd.read_csv(
            csv_file) if csv_file != None else source_dataframe
        self.nlp = spacy.load('en_core_web_sm')  # TODO lang-specific model
        self.sentences: List[Any] = [self.nlp(sent) for sent in sent_tokenize(
            ' '.join(self.df['Utterance'].tolist()))]
        speakers_num_sentences = list(
            map(lambda x: len(sent_tokenize(x)), self.df['Utterance']))
        self.speakers = []
        for speaker, num in zip(self.df['Speaker'].tolist(), speakers_num_sentences):
            self.speakers.extend([speaker] * num)
        self.sentence_similarity_graph: Optional[nx.Graph] = None
        self.sentence_similarity: Optional[np.array] = None
        self.sentence_vectors: Optional[list] = None
        self.sentence_weights: Optional[list] = None
        self.sentences_communities: Optional[dict[Any, Any]] = None
        self.entailment_matrix: Optional[np.array] = None
        self.entailment_graph: Optional[nx.DiGraph] = None
        self.sentence_cluster_ids: Optional[np.array] = None
        # TODO switch to multilingual sentence transformer as base model?
        self.kw_model = None

    def __call__(
        self,
            output: Output_type,
            filter_backchannels: bool = True,
            remove_entailed_sentences: bool = True,
            get_graph_backbone: bool = True,
            do_cluster: bool = True,
            clustering_algorithm: str = 'louvain',
            per_cluster_results: bool = True
    ):
        """
        Get the summary of the meeting alongside the important words
        """

        if filter_backchannels:
            self.filter_backchannels_duplicates()
        self.count_keywords()

        if output == Output_type.WORD:
            return self.keyword_counts

        self.weigh_sentences()

        self.remove_unnecessary_sentences()
        self.vectorize_sentences_with_keywords()
        self.calculate_sentence_similarity()

        self.generate_sentence_similarity_graph()

        if remove_entailed_sentences:
            self.simple_entailment()
            self.remove_nodes_based_on_entailment_matrix_and_similarity_graph()

        if get_graph_backbone:
            self.sentence_similarity_graph = Utils.get_graph_backbone(
                self.sentence_similarity_graph)

        # TODO use the information gathered by the sentence similarity graph to the clustering algorithm input
        if do_cluster:
            if clustering_algorithm == 'louvain':
                self.cluster_sentences_by_louvain()
            elif clustering_algorithm == 'kmeans':
                self.cluster_sentences()

            else:
                raise Exception(
                    'the algorithm specified is not in the accepted ones!'
                )
            if per_cluster_results:
                output_bert = self.get_clustered_sentences_w_keywords()
                output_tfidf = self.get_keywords_per_sentence_cluster()
                output_best_per_cluster = self.best_sentences_per_cluster()

                return {
                    'output_bert': output_bert,
                    'output_tfidf': output_tfidf,
                    'output_best_per_cluster': output_best_per_cluster
                }

        if output == Output_type.SENTENCE:
            return dict(self.sentence_similarity_graph.nodes(data=True))

        # if output == Output_type.HTML:
            # return {'HTML': self.write_dataframe_with_weight_community_html()}

    def remove_unnecessary_sentences(self):
        if self.sentence_weights is None:
            self.weigh_sentences()

        unnecessary_sentences_indices = np.where(
            np.array(self.sentence_weights) == 0)[0]

        self.sentences = [self.sentences[i] for i in range(
            len(self.sentences)) if i not in unnecessary_sentences_indices]
        self.sentence_weights = [self.sentence_weights[i] for i in range(
            len(self.sentence_weights)) if i not in unnecessary_sentences_indices]
        self.speakers = [self.speakers[i] for i in range(
            len(self.speakers)) if i not in unnecessary_sentences_indices]

    def filter_backchannels_duplicates(self):
        """
        Computes the backchannels of the utterances
        """
        count_sentences_before_filtering = len(self.sentences)
        proper_sentence_indices = []
        for i, sent in tqdm(enumerate(self.sentences), leave=False):
            if not Utils._nlp_backchannel(sent.text)[0]:
                proper_sentence_indices.append(i)

        self.sentences = [self.sentences[i] for i in proper_sentence_indices]
        self.speakers = [self.speakers[i] for i in proper_sentence_indices]

        count_sentences_after_filtering = len(self.sentences)
        print(
            f'Number of back channels in the sentences {count_sentences_before_filtering - count_sentences_after_filtering}')

        sentences_set = set()
        keep_indices = []
        for i, sent in enumerate(self.sentences):
            if sent.text not in sentences_set:
                keep_indices.append(i)
                sentences_set.add(sent.text)
        print(
            f'Found {len(self.sentences) - len(keep_indices)} duplicate sentences')
        self.sentences = [self.sentences[i] for i in keep_indices]
        self.speakers = [self.speakers[i] for i in keep_indices]

    def count_keywords(self, max_np_len: int = 5, min_count: int = 0):
        self.keyword_counts = Counter()
        tok_counts, mwt_counts = Counter(), Counter()
        for sent in self.sentences:
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
                mwts = Counter(
                    {mwt: c for mwt, c in mwt_counts.items() if tok in mwt})
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

    def weigh_sentences(self):
        """ Weigh sentences based on sum of count values of keywords """
        if self.keyword_counts is None:
            self.count_keywords()
        self.sentence_weights = list()

        for sent in self.sentences:
            keyword_lemmas = [
                tok.lemma_ for tok in sent if tok.pos_ in self.keyword_pos]
            score = sum([self.keyword_counts[noun]
                        for noun in set(keyword_lemmas)])
            self.sentence_weights.append(score)

    def vectorize_sentences_with_keywords(self):
        """ Vectorize based on keyword occurrence """
        if self.keyword_counts is None:
            self.count_keywords()
        self.sentence_vectors = list()
        keyword_vocab_ix = {k: i for i, k in enumerate(
            self.keyword_counts.keys())}
        vec_size = len(keyword_vocab_ix)

        for sent in self.sentences:
            sent_vec = np.zeros(vec_size, dtype=int)
            for tok in sent:
                if tok.lemma_ in self.keyword_counts and tok.pos_ in self.keyword_pos:
                    sent_vec[keyword_vocab_ix[tok.lemma_]
                             ] = self.keyword_counts[tok.lemma_]
            self.sentence_vectors.append(sent_vec)

    def calculate_sentence_similarity(self):
        """ Sentence graph based on similarity and sentence weight
        We simply sum the scores of the shared keywords to determine sentence similarity """
        if self.sentence_vectors is None:
            self.vectorize_sentences_with_keywords()

        if self.sentence_weights is None:
            self.weigh_sentences()
        self.sentence_similarity = np.zeros(
            (len(self.sentences), len(self.sentences))
        )
        for i in range(len(self.sentence_vectors)):
            for j in range(i + 1, len(self.sentence_vectors)):
                sent_vec1 = self.sentence_vectors[i]
                sent_vec2 = self.sentence_vectors[j]
                shared_keywords_ixs = np.where(sent_vec1 == sent_vec2)[0]
                shared_keywords_scores = np.take(
                    sent_vec1, shared_keywords_ixs)
                sim = np.sum(shared_keywords_scores)
                self.sentence_similarity[i][j] = sim
                self.sentence_similarity[j][i] = sim

    def generate_sentence_similarity_graph(self, sim_thresh: float = 0.75, max_only: bool = False):
        if self.sentence_similarity is None:
            self.calculate_sentence_similarity()
        self.sentence_similarity_graph = nx.Graph()
        for i, sim_vec in enumerate(self.sentence_similarity):
            if np.sum(sim_vec) == 0:  # No similarity to others
                continue

            nearest_sent_ix = np.argsort(sim_vec)[-1]
            highest_sim = sim_vec[nearest_sent_ix]
            if highest_sim == 0:  # shouldn't happen
                continue
            if highest_sim >= sim_thresh or max_only:
                self.sentence_similarity_graph.add_edge(
                    self.sentences[i].text, self.sentences[nearest_sent_ix].text, weight=highest_sim)
            else:
                for j in range(i + 1, len(sim_vec)):
                    if sim_vec[j] >= sim_thresh:
                        self.sentence_similarity_graph.add_edge(
                            self.sentences[i].text, self.sentences[j].text, weight=sim_vec[j])
        for i, sent in enumerate(self.sentences):  # add sentence weights to nodes
            if sent.text in self.sentence_similarity_graph:
                self.sentence_similarity_graph.nodes(
                )[sent.text]['weight'] = self.sentence_weights[i]
                self.sentence_similarity_graph.nodes(
                )[sent.text]['sentence_id'] = i

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
        self.sentence_cluster_ids = [int(label) for label in kmeans.labels_]

    def get_clustered_sentences_w_keywords(self):
        self.kw_model = KeyBERT()
        if self.sentence_cluster_ids is None:
            self.cluster_sentences()
        clustered_sentences = defaultdict(list)
        for cluster_id, sent in zip(self.sentence_cluster_ids, self.sentences):
            clustered_sentences[cluster_id].append(sent.text)

        outputs = {}
        for cluster_id, sents in clustered_sentences.items():
            outputs[cluster_id] = {}
            cluster_weight = np.mean([self.sentence_weights[i] for i, sent in enumerate(
                self.sentences) if sent.text in sents])

            keyword_counts = Counter([t.lemma_ for t in self.nlp(
                ' '.join(sents)) if t.pos_ in self.keyword_pos])
            outputs[cluster_id]['common_keywords'] = [{
                'content': entity[0],
                'weight': entity[1]
            }
                for entity in keyword_counts.most_common(5)
            ]
            keywords = self.kw_model.extract_keywords(
                ' '.join(sents), keyphrase_ngram_range=(1, 3), top_n=5)
            outputs[cluster_id]['keybert'] = [{
                'content': entity[0],
                'weight': entity[1]
            } for entity in keywords]

        return outputs

    def get_keywords_per_sentence_cluster(self, num_keywords: int = 25):
        if self.sentence_cluster_ids is None:
            self.cluster_sentences()
        texts_per_cluster = defaultdict(str)
        for sent, cluster_id in zip(self.sentences, self.sentence_cluster_ids):
            texts_per_cluster[cluster_id] += ' ' + sent.text
        from sklearn.feature_extraction.text import TfidfVectorizer
        tfidf = TfidfVectorizer(
            sublinear_tf=True, ngram_range=(1, 2), stop_words='english')
        cluster_vecs = tfidf.fit_transform(
            list(texts_per_cluster.values())).toarray()
        word_list = tfidf.get_feature_names()
        outputs = {}
        for clust_id, vec in zip(texts_per_cluster.keys(), cluster_vecs):
            best_ixs = np.argsort(vec)
            best_ixs = np.flip(best_ixs)
            outputs[clust_id] = list(
                np.take(word_list, best_ixs[:num_keywords]))
        return outputs

    def best_sentences_per_cluster(self, num_sent: int = 5):
        outputs = {}
        if self.sentence_weights is None:
            self.weigh_sentences()
        if self.sentence_cluster_ids is None:
            self.cluster_sentences()
        sent_weights = {sent.text: weight for sent, weight in zip(
            self.sentences, self.sentence_weights)}
        sents_per_cluster = defaultdict(list)
        for sent, cluster_id in zip(self.sentences, self.sentence_cluster_ids):
            sents_per_cluster[cluster_id].append(
                (sent_weights[sent.text], sent.text))
        for cluster_id, sents in sents_per_cluster.items():
            outputs[cluster_id] = '\n'.join([tup[1] for tup in
                                             list(sorted(sents, reverse=True,
                                                         key=lambda x: x[0])[:num_sent])
                                             ])
        return outputs

    def cluster_sentences_by_louvain(self):
        """
        Cluster the sentences with Louvain community detection algorithm based on the sentence similarities.
        """
        self.sentences_communities = best_partition(
            self.sentence_similarity_graph)
        print(
            f'Number of partitions in the network {len(set(self.sentences_communities.values()))}')
        nx.set_node_attributes(self.sentence_similarity_graph,
                               self.sentences_communities, 'community')
        self.sentence_cluster_ids = []
        sentence_ids_in_graph = list(map(
            lambda x: x['sentence_id'], dict(self.sentence_similarity_graph.nodes(data=True)).values()))

        for i, sent in enumerate(self.sentences):
            if i not in sentence_ids_in_graph:
                self.sentence_cluster_ids.append(-1)
            else:
                self.sentence_cluster_ids.append(
                    self.sentences_communities[sent.text]
                )

    def remove_nodes_based_on_entailment_matrix_and_similarity_graph(self):
        if self.sentence_similarity_graph is None:
            self.generate_sentence_similarity_graph()
        if np.any(self.entailment_matrix) is None:
            self.simple_entailment()

        sentences_to_remove_ids = np.nonzero(self.entailment_matrix)[1]
        sentences_to_keep_ids = np.nonzero(self.entailment_matrix)[0]

        for i in range(len(sentences_to_keep_ids)):
            print(f'keep: {self.sentences[sentences_to_keep_ids[i]].text}')
            print(f'remove: {self.sentences[sentences_to_remove_ids[i]].text}')
            print(10 * '*')

        sentences_to_remove = [self.sentences[i].text
                               for i in set(sentences_to_remove_ids)]

        print(
            f'Number of nodes before applying removing sentences using entailment matrix {self.sentence_similarity_graph.number_of_nodes()}\n',
            f'Number of edges before applying removing sentences using entailment matrix {self.sentence_similarity_graph.number_of_edges()}',
        )
        self.sentence_similarity_graph.remove_nodes_from(sentences_to_remove)
        print(
            f'Number of nodes after applying removing sentences using entailment matrix {self.sentence_similarity_graph.number_of_nodes()}\n',
            f'Number of edges after applying removing sentences using entailment matrix {self.sentence_similarity_graph.number_of_edges()}',
        )

    def simple_entailment(self, threshold: float = 0.85):
        """ Simply regard entailment as the overlap of keywords.
        If sent 1 has all keywords of sent 2 and more, sent 1 entails sent 2."""
        if self.sentence_similarity is None:
            self.calculate_sentence_similarity()
        self.entailment_graph = nx.DiGraph()
        num_sentences = len(self.sentences)
        self.entailment_matrix = np.zeros([num_sentences, num_sentences])
        for i in range(len(self.sentence_weights)):
            for j in range(i + 1, len(self.sentence_weights)):
                sent_weight1 = self.sentence_weights[i]
                sent_weight2 = self.sentence_weights[j]
                similarity = self.sentence_similarity[i, j]
                if similarity > 0:
                    if sent_weight1 > sent_weight2:
                        if similarity / sent_weight2 > threshold:
                            self.entailment_graph.add_edge(i, j)
                            self.entailment_matrix[i, j] = 1
                    else:
                        if similarity / sent_weight1 > threshold:
                            self.entailment_graph.add_edge(j, i)
                            self.entailment_matrix[j, i] = 1

# TODO
# do something for the speaker part and align it with the self.sentences property

    # def write_dataframe_with_weight_community_html(self):
    #     self.kw_model = KeyBERT()
    #     html_output = ""
    #     num_communities = len(set(self.sentences_communities.values()))
    #     colors = np.random.rand(num_communities, 3)
    #     colors *= 256
    #     colors = colors.astype('int')
    #     normalized_sentence_weights = np.array(
    #         self.sentence_weights) / max(self.sentence_weights)
    #     sentence_weight_mapping = {
    #         self.sentences[i].text: normalized_sentence_weights[i]
    #         for i in range(len(self.sentences))
    #     }
    #     keywords = []
    #     for community in set(self.sentences_communities.values()):
    #         community_sentences = '; '.join([
    #             sent for sent, sent_att in self.sentence_similarity_graph.nodes(data=True) if sent_att['community'] == community])
    #         keywords.append([word[0] for word in self.kw_model.extract_keywords(
    #             community_sentences, keyphrase_ngram_range=(1, 2), top_n=3)])

    #     html_legend = """
    #     <div class="legend">
    #     """
    #     for i in range(num_communities):
    #         html_legend += """<div style="margin: 5px 0;">\n"""
    #         html_legend += f"""
    #                 <div class="community" style="color: rgb{str(tuple(colors[i]))};">
    #                     community {i + 1}
    #                 </div>
    #         """
    #         html_legend += "<div>\n"
    #         for keyword in keywords[i]:
    #             html_legend += f"""<span style="color: rgb{str(tuple(colors[i]))}
    #             ">{keyword}, </span>"""
    #         html_legend += "</div>\n"
    #         html_legend += "</div>\n"
    #     html_legend += "</div>\n"
    #     graph_nodes_sentences = [sent for sent,
    #                              _ in self.sentence_similarity_graph.nodes(data=True)]

    #     html_output += """
    #     <style>
    #         .legend {
    #             transition-property: opacity;
    #             transition-duration: .2s;
    #             width: 200px;
    #             display: flex;
    #             justify-content: center;
    #             align-content: center;
    #             padding: 23px;
    #             background: gray;
    #             position: fixed;
    #             right: 20px;
    #             top: 10%;
    #             opacity: 50%;
    #             border-radius: 8px;
    #             flex-direction: column;
    #         }

    #         .community::first-letter {
    #             font-size: 30px;
    #         }

    #         .community {
    #             font-size: 20px;
    #         }

    #         .legend:hover {
    #             opacity: 100%;
    #         }
    #     </style>
    #     """
    #     html_output += '<html>\n'
    #     html_output += html_legend
    #     html_output += '<table border=1>\n'
    #     for sent, speaker in zip(self.sentences, self.speakers):
    #         html_output += '<tr><td>' + str(speaker) + '</td><td>'
    #         for sent in self.nlp(turn['Utterance']).sents:
    #             if sent.text not in graph_nodes_sentences:
    #                 font = 11
    #             else:
    #                 font = max(sentence_weight_mapping[sent.text] * 20, 16)
    #             if sent.text not in graph_nodes_sentences:
    #                 color = (169, 169, 169)
    #             else:
    #                 color = tuple(
    #                     colors[self.sentences_communities[sent.text]])
    #             html_output += f'<span style="font-size: {str(int(font))}; color: rgb{str(color)};">' + \
    #                 sent.text + ' </span>'
    #         html_output += '</td></tr>\n'
    #     html_output += '</table></html>'
    #     with open('weighted_sentences_per_speaker.html', 'w') as f:
    #         f.write(html_output)
    #         f.close()
    #     return html_output


if __name__ == '__main__':
    df = pd.read_csv(
        '../../dialogue_summarization/transcriptsforkeyphraseextraction/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv')
    # with open('sample_data/sample_01.json') as f:
    #     data = json.load(f)
    #     f.close()
    # transcript = Transcript(data)
    # source_dataframe = transcript.df
    dt = UnsupervisedSummarizer(
        csv_file=None, source_dataframe=df)

    result = dt(
        output=Output_type.SENTENCE,
        clustering_algorithm='kmeans'
    )
    breakpoint()
