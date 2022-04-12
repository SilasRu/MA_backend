from typing import List
import networkx as nx
import spacy
from spacy.matcher import Matcher
import pandas as pd
from nltk.tokenize import sent_tokenize


class KnowledgeGraph:
    nlp = spacy.load('en_core_web_sm')

    def __init__(self, sentences: List[str]) -> None:
        self.sentences = sentences
        self.find_subj_obj()
        self.find_root_verbs()
        self.gather_graph_nodes_edges()
        self.write_graph()

    def write_graph(self):
        nx.write_gexf(
            self.G, '/Users/user/Documents/dialogue_summarization/transcript_analyser/utils/sample_knowledge_graph.gexf')

    def gather_graph_nodes_edges(self):
        # extract subject

        indices_to_keep = []

        for index, pair in enumerate(self.entity_pairs):
            if pair[0] != '' and pair[1] != '':
                indices_to_keep.append(index)

        self.entity_pairs = list(map(
            lambda index: self.entity_pairs[index], indices_to_keep))
        self.relations = list(map(
            lambda index: self.relations[index], indices_to_keep
        ))

        self.source = [i[0] for i in self.entity_pairs]

        # extract object
        self.target = [i[1] for i in self.entity_pairs]

        self.kg_df = pd.DataFrame(
            {'source': self.source, 'target': self.target, 'edge': self.relations})

        self.G = nx.from_pandas_edgelist(self.kg_df, "source", "target",
                                         edge_attr=True, create_using=nx.MultiDiGraph())

        # breakpoint()

    def find_root_verbs(self):
        self.relations = [self.get_relation(sent) for sent in self.sentences]

    def get_relation(self, sent):
        doc = self.nlp(sent)

        # Matcher class object
        matcher = Matcher(self.nlp.vocab)

        # define the pattern
        pattern = [{'DEP': 'ROOT'},
                   {'DEP': 'prep', 'OP': "?"},
                   {'DEP': 'agent', 'OP': "?"},
                   {'POS': 'ADJ', 'OP': "?"}]

        matcher.add("matching_1", None, pattern)

        matches = matcher(doc)
        k = len(matches) - 1

        span = doc[matches[k][1]:matches[k][2]]

        return(span.text)

    def find_subj_obj(self):
        self.entity_pairs = []
        for sent in self.sentences:
            self.entity_pairs.append(self.get_entities(sent))

    def get_entities(self, sent):
        # print(sent)
        # chunk 1
        ent1 = ""
        ent2 = ""

        prv_tok_dep = ""    # dependency tag of previous token in the sentence
        prv_tok_text = ""   # previous token in the sentence

        prefix = ""
        modifier = ""

        #############################################################

        for tok in self.nlp(sent):
            # chunk 2
            # if token is a punctuation mark then move on to the next token
            if tok.dep_ != "punct":
                # check: token is a compound word or not
                if tok.dep_ == "compound":
                    prefix = tok.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " " + tok.text

            # check: token is a modifier or not
            if tok.dep_.endswith("mod") == True:
                modifier = tok.text
                # if the previous word was also a 'compound' then add the current word to it
                if prv_tok_dep == "compound":
                    modifier = prv_tok_text + " " + tok.text

            # chunk 3
            if tok.dep_.find("subj") == True:
                ent1 = modifier + " " + prefix + " " + tok.text
                prefix = ""
                modifier = ""
                prv_tok_dep = ""
                prv_tok_text = ""

            # chunk 4
            if tok.dep_.find("obj") == True:
                ent2 = modifier + " " + prefix + " " + tok.text

            # chunk 5
            # update variables
            prv_tok_dep = tok.dep_
            prv_tok_text = tok.text
        #############################################################

        # print(ent1.strip(), ent2.strip())
        return [ent1.strip(), ent2.strip()]


if __name__ == "__main__":
    df = pd.read_csv(
        '/Users/user/Documents/dialogue_summarization/transcriptsforkeyphraseextraction/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv'
    )
    text = '. '.join(df[df['Speaker'] == "Mark"]['Utterance'].to_list())
    sentences = sent_tokenize(text)
    kg = KnowledgeGraph(sentences)
