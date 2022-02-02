"""
Experiment on recursively selecting the split that minimizes entropy
"""
import os.path
import re
import networkx as nx
import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm
from rake_nltk import Rake
from collections import Counter, defaultdict
from nltk.corpus import stopwords
from nltk import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import BartTokenizer, BartForConditionalGeneration

NGRAM_RANGE = (1, 4)


def get_text_per_speaker(df: pd.DataFrame) -> dict:
    speakers = list(set(df['Speaker']))
    return {speaker: ' '.join(df[df['Speaker'] == speaker]['Utterance'].tolist())
            for speaker in speakers}


def get_noun_chunks(text: str, nlp: spacy.language, lemmatize: bool = False, stopword_removal: bool = False) -> list:
    doc = nlp(text)
    ncs = list()

    if stopword_removal:
        stops = set(stopwords.words('english'))
        stops.update(['everything', 'something', 'everyone', 'someone'])

    for chunk in doc.noun_chunks:
        if any(tok for tok in chunk if tok.pos_ in {'PROPN', 'NOUN'}):

            if stopword_removal:
                chunk = [tok for tok in chunk if not tok.lemma_.lower() in stops]

            if lemmatize:
                ncs.append(' '.join([t.lemma_ for t in chunk]))
            else:
                ncs.append(' '.join([t.text for t in chunk]))
    return ncs


def keyphrases_per_speaker_tfidf(df: pd.DataFrame, nlp: spacy.language, num_keyphrases: int = 10) -> None:
    """
    either build vocab on full text and then fit separate tfidfs using vocab,
    or train tfidf on all text and vectorize separately
    """
    print('\nKeyphrases per speaker')

    text_per_speaker = get_text_per_speaker(df)
    speakers = text_per_speaker.keys()
    texts_per_speaker = text_per_speaker.values()

    noun_chunks_per_speaker = [' '.join(get_noun_chunks(text, nlp)) for text in texts_per_speaker]

    #vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=NGRAM_RANGE, stop_words='english')
    vectorizer = CountVectorizer(ngram_range=NGRAM_RANGE, stop_words='english')
    vecs_per_speaker = vectorizer.fit_transform(noun_chunks_per_speaker).toarray()
    features = vectorizer.get_feature_names()

    for speaker, vec, text in zip(speakers, vecs_per_speaker, texts_per_speaker):
        idxs = np.flip(np.argsort(vec))
        print(speaker)
        for idx in idxs[:num_keyphrases]:
            print(vec[idx], features[idx])


def frequent_noun_chunks(df: pd.DataFrame, nlp: spacy.language, num_keyphrases: int = 25,
                         stopword_filtering: bool = True, add_speakers_to_graph: bool = False,
                         selection_method: str = 'kwic') -> None:
    print('\nFrequent noun chunks')
    text = ' '.join(df['Utterance'])
    noun_chunks = get_noun_chunks(text, nlp)
    if stopword_filtering:
        stops = set(stopwords.words('english'))
        stops.update(['everything', 'something', 'everyone', 'someone'])
        filtered_ncs = [
            ' '.join([w for w in nc.split(' ') if w.lower() not in stops])
            for nc in noun_chunks
        ]
    else:
        filtered_ncs = noun_chunks
    nc_counts = Counter(filtered_ncs)
    print('Raw counts')
    print(nc_counts.most_common(100))
    nc_text = ' '.join(get_noun_chunks(text, nlp))

    vectorizer = CountVectorizer(ngram_range=(1, 7), stop_words='english')
    vec = vectorizer.fit_transform([nc_text]).toarray()[0]
    features = vectorizer.get_feature_names()

    print('Count ngrams in noun chunks')
    # TODO add speakers to the graph: add edge to nodes they are connected to
    text_per_speaker = get_text_per_speaker(df)
    nc_per_speaker = {speaker: Counter(get_noun_chunks(text_per_speaker[speaker], nlp, stopword_removal=True))
                      for speaker in text_per_speaker.keys()}

    g = nx.Graph()
    if add_speakers_to_graph:
        for speaker in nc_per_speaker:
            g.add_node(speaker, label='speaker')
    idxs = np.flip(np.argsort(vec))

    for idx in idxs[:num_keyphrases]:
        ngram = features[idx]
        children = [(nc, c) for nc, c in nc_counts.most_common() if ngram in nc]
        print(vec[idx], features[idx], '-', children)
        g.add_node(ngram, weight=vec[idx], label='ngram')

        if add_speakers_to_graph:
            for speaker, nc_counts_speaker in nc_per_speaker.items():
                if ngram in nc_counts_speaker:
                    g.add_edge(speaker, ngram, weight=nc_counts_speaker[ngram], label='speaker2ngram')

        for (child, weight) in children:
            if not child == ngram:
                g.add_node(child, weight=weight, label='ngram')
                g.add_edge(ngram, child, weight=1, label='ngram2ngram')

                """
                for speaker, nc_counts_speaker in nc_per_speaker.items():
                    if child in nc_counts_speaker:
                        g.add_edge(speaker, child, weight=nc_counts_speaker[child], label='speaker2ngram')
                """

    nx.write_graphml(g, 'nouns.graphml')

    def find_verbal_head(kw_tok, doc):
        if kw_tok.head.pos_ == 'VERB':
            return kw_tok.head
        else:
            return find_verbal_head(kw_tok.head, doc)

    print('\nPhrases per keyword')
    sents = sent_tokenize(text)  # TODO do this per speaker

    with open('keyword_phrases.html', 'w') as html_out_file:

        for idx in idxs[:num_keyphrases]:
            html_out_file.write('<table>\n')
            keyword = features[idx]
            sents_w_keyword = [sent for sent in sents if keyword.lower() in sent.lower()]
            print()
            print(keyword)
            print('-' * len(keyword))

            kwic_window_size = 12
            html_out_file.write('<tr><td></td><td><b>' + keyword + '</b></td><td></td></tr>')
            for sent in sents_w_keyword:

                if selection_method == 'kwic':
                    toks = word_tokenize(sent)
                    kw_tok_ixs = [i for i, tok in enumerate(toks) if tok == keyword]
                    for ix in kw_tok_ixs:
                        start_ix = ix - kwic_window_size if (ix - kwic_window_size) > 0 else 0
                        end_ix = ix + kwic_window_size + 1 if (ix + kwic_window_size + 1) < len(toks) else len(toks) + 1
                        phrase_str = ' '.join(toks[start_ix: end_ix])
                        html = '<tr><td style="text-align: right">' + ' '.join(toks[start_ix:ix]) + '</td><td style="background-color: pink; text-align: center"> ' + keyword + \
                               ' </td><td>' + ' '.join(toks[ix+1: end_ix]) + '</td></tr>'
                        print(html, file=html_out_file)
                        print(phrase_str)

                elif selection_method == 'verb_phrases':
                    # TODO
                    doc = nlp(sent)
                    kw_toks = [t for t in doc if t.lemma_ == keyword]
                    for kw_tok in kw_toks:
                        try:
                            head = find_verbal_head(kw_tok, doc)
                        except RecursionError:
                            head = kw_tok.head
                        siblings = [child for child in head.children if child.pos_ not in {'PUNCT', 'VERB', 'CCONJ'}]
                        if siblings:
                            end_tok_ix = siblings[-1].i + 1 if siblings[-1].i + 1 <= len(doc) else len(doc)
                            phrase = doc[siblings[0].i: end_tok_ix]
                            phrase_str = ' '.join([t.text for t in phrase])
                            print(phrase_str)

                else:
                    raise NotImplementedError

            html_out_file.write('<tr><td colspan=3>&nbsp;</td></tr>')
            html_out_file.write('</table>\n')


def tfidf_noun_chunks(df: pd.DataFrame, nlp: spacy.language, num_keyphrases: int = 25) -> None:
    print('\nTfidf noun chunks')
    texts = list(df['Utterance'])
    noun_chunks = [' '.join(get_noun_chunks(text, nlp)) for text in texts]
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=NGRAM_RANGE, stop_words='english')
    vectorizer.fit(noun_chunks)
    features = vectorizer.get_feature_names()
    vec = vectorizer.transform([' '.join(noun_chunks)]).toarray()[0]
    idxs = np.flip(np.argsort(vec))
    for idx in idxs[:num_keyphrases]:
        print(vec[idx], features[idx])


def tfidf_sentences(df: pd.DataFrame, num_keyphrases: int = 20) -> None:
    texts = ' '.join(df['Utterance'])
    sentences = sent_tokenize(texts)
    vectorizer = TfidfVectorizer(sublinear_tf=True, ngram_range=NGRAM_RANGE, stop_words='english')
    sent_vecs = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    vec = vectorizer.transform([texts]).toarray()[0]
    idxs = np.flip(np.argsort(vec))
    for idx in idxs[:num_keyphrases]:
        print(vec[idx], features[idx])

    """
    idxs = np.argsort(sent_vecs, axis=None)
    idxs = np.dstack(np.unravel_index(np.argsort(sent_vecs.ravel()), sent_vecs.shape))
    feats_idxs = np.flip(idxs[0][:, 1])
    for idx in feats_idxs[:num_keyphrases]:
        print(features[idx])
    """


def summary_per_speaker(df: pd.DataFrame) -> None:
    print('\nTextRank Summary per speaker')
    text_per_speaker = get_text_per_speaker(df)
    for speaker, text in text_per_speaker.items():
        sentences = sent_tokenize(text)
        if len(sentences) > 1:
            print(speaker)
            print(summarize(text, word_count=500))
        print()
    print('\nTextRank summary')
    text = ' '.join(df['Utterance'])
    print(summarize(text, word_count=500, split=True))


def rake_keyphrases(df: pd.DataFrame) -> None:
    print('\nRake keyphrases')
    texts = ' '.join(df['Utterance'])
    r = Rake()
    r.extract_keywords_from_text(texts)
    keyphrases = r.get_ranked_phrases_with_scores()
    for kp in keyphrases[:25]:
        print(kp)


def rake_keyphrases_per_speaker(df: pd.DataFrame) -> None:
    print('\nRake keyphrases per speaker')
    text_per_speaker = get_text_per_speaker(df)
    for speaker, text in text_per_speaker.items():
        r = Rake()
        r.extract_keywords_from_text(text)
        keyphrases = r.get_ranked_phrases_with_scores()
        print(speaker)
        for kp in keyphrases[:25]:
            print(kp)
        print()


def pmi_keyphrases(df: pd.DataFrame, window_size: int = 5, stride: int = 3) -> None:

    print('\nPMI keyphrases ngrams')
    texts = ' '.join(df['Utterance'])

    cv = CountVectorizer(ngram_range=(2, 4))#, stop_words='english')  #instead of stop words, make sure that there is a noun in the ngram?
    vec = cv.fit_transform([texts]).toarray()[0]
    max_pair_count = np.max(vec)
    toks = re.findall(r'(?u)\b\w\w+\b', texts.lower())  # use the same tokenization as sklearn
    tok_counts = Counter(toks)
    max_count = tok_counts.most_common(1)[0][1]
    tok_probs = Counter({w: c / max_count for w, c in tok_counts.items()})
    pmi_ngrams = Counter()

    for ngram, ix in cv.vocabulary_.items():
        if vec[ix] == 1:    # skip ngrams only seen once
            continue
        ngram_toks = ngram.split(' ')
        ngram_prob = vec[ix] / max_pair_count
        pmi = ngram_prob / np.prod([tok_probs[tok] for tok in ngram_toks])
        pmi_ngrams[ngram] = pmi

    for ngram_pmi in pmi_ngrams.most_common(25):
        print(ngram_pmi)

    tok_pair_counts = Counter()
    for ix in range(0, len(toks) - window_size, stride):
        window_toks = toks[ix: ix+window_size]
        for i, tok1 in enumerate(window_toks):
            for tok2 in window_toks[i+1:]:
                tok_pair_counts[(tok1, tok2)] += 1

    max_pair_count = tok_pair_counts.most_common(1)[0][1]
    tok_pair_probs = Counter({w: c/max_pair_count for w, c in tok_pair_counts.items()})

    pmi_pairs = Counter({tok_pair: tok_pair_prob / (tok_probs[tok_pair[0]] * tok_probs[tok_pair[1]])
                         for tok_pair, tok_pair_prob in tok_pair_probs.items()
                         if (tok_counts[tok_pair[0]] > 1 and tok_counts[tok_pair[1]] > 1)}
                        )

    print('\nPMI keyphrases from sliding window')
    for pmi_pair in pmi_pairs.most_common(25):
        print(pmi_pair)


def get_sections(text, num_words, permutation=False):
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


def get_bart_keywords(df: pd.DataFrame, summary_min_length: int = 300, summary_max_length: int = 300, permutation=False):
    print('\nBART summary')
    utterances = ' '.join(df['Utterance'])
    model = BartForConditionalGeneration.from_pretrained("philschmid/bart-large-cnn-samsum")
    tokenizer = BartTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")
    result = ''
    sections_to_process = get_sections(utterances, 900 - summary_max_length, permutation=permutation)
    for section in tqdm(sections_to_process, leave=False):
        text = result + section
        inputs = tokenizer([text], max_length=1024, return_tensors='pt', truncation=True)
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=summary_min_length,
                                     max_length=summary_max_length, early_stopping=True)
        result = tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return result


def dep_collocates_of_keywords(df: pd.DataFrame, nlp, keywords: list) -> None:
    """Find collocates of keywords based on the dependency parse"""
    print('\nDependency collocates')
    text = '\n'.join(df['Utterance'])
    sents = sent_tokenize(text)

    for kw in keywords:
        collocates = list()

        sents_kw = [s for s in sents if kw.lower() in s.lower()]
        for sent in sents_kw:

            doc = nlp(sent)
            kw_toks = [t for t in doc if t.lemma_ == kw]

            for kw_tok in kw_toks:

                collocates.extend([tok for tok in kw_tok.children if tok.pos_ in {'NOUN', 'ADJ'}])  # team meeting, difficult meeting

                if kw_tok.head.pos_ == 'VERB':  # summarize the meeting
                    collocates.append(kw_tok.head)
                elif kw_tok.head.pos_ == 'ADP':  # summary of the meeting
                    if kw_tok.head.head.pos_ == 'NOUN':  # meeting summaries
                        collocates.append(kw_tok.head.head)
                elif kw_tok.head.pos_ == 'NOUN':
                    collocates.append(kw_tok.head)

        print()
        print(kw, collocates)
        for collocate in collocates:
            rel_sents = [sent for sent in sents if kw.lower() in sent.lower() and collocate.text.lower() in sent.lower()]
            print(kw, collocate)
            print(rel_sents)


def cluster_key_words(df: pd.DataFrame, key_words:list) -> None:
    print('\nKeyword clustering')
    import json
    from scipy.spatial.distance import pdist, squareform
    wemb = np.load('/home/don/resources/fastText_MUSE/wiki.multi.en.vec_data.npy')
    wemb_vocab = json.load(open('/home/don/resources/fastText_MUSE/wiki.multi.en.vec_vocab.json'))
    text = ' '.join(df['Utterance'])
    toks = list(set(word_tokenize(text.lower())))
    stops = set(stopwords.words('english'))
    toks = [tok for tok in toks if tok not in stops and tok in wemb_vocab]
    tok_ixs = [wemb_vocab[tok] for tok in toks]
    tok_vecs = np.take(wemb, np.asarray(tok_ixs), axis=0)
    tok_similarity = 1 - squareform(pdist(tok_vecs, metric='cosine'))
    num_syns = 10
    for kw in key_words:
        kw_sim_vec = tok_similarity[toks.index(kw)]
        sorted_ixs = np.flip(np.argsort(kw_sim_vec))
        sim_toks = np.take(toks, sorted_ixs)
        sims = np.take(kw_sim_vec, sorted_ixs)
        print(kw, list(zip(sims[1:num_syns], sim_toks[1:num_syns])))  # first one is kw itself


def sentence_clustering(df: pd.DataFrame, calc_sim: bool = False, sim_thresh: float = 0.75) -> None:
    text = ' '.join(df['Utterance'])
    sents = sent_tokenize(text)
    if not os.path.exists('sent_sim.npy') or calc_sim:
        from sentence_transformers import SentenceTransformer
        from scipy.spatial.distance import pdist, squareform
        print('loading sentence encoder')
        sent_enc = SentenceTransformer('stsb-xlm-r-multilingual')
        print('encoding sentences')
        sent_vecs = sent_enc.encode(sents)
        print('calculating sentence similarity')
        sent_sim = 1 - squareform(pdist(sent_vecs, metric='cosine'))
        np.save('sent_sim.npy', sent_sim)
    else:
        sent_sim = np.load('sent_sim.npy')

    g = nx.Graph()
    for i, sim_vec in enumerate(sent_sim): # only iterate through upper right triangle
        if np.max(sim_vec) < sim_thresh:
            g.add_edge(sents[i], sents[np.argmax[sim_vec]], weight=np.max(sim_vec))
        else:
            for j, sim in enumerate(sim_vec[i+1:]):
                if sim >= sim_thresh:
                    g.add_edge(sents[i], sents[i+1+j], weight=sim)
    # TODO add sentence weights
    
    nx.write_graphml(g, 'sent_graph.graphml')
    for cc in nx.connected_components(g):
        print(cc)


def sentence_ranking_by_similarity(df: pd.DataFrame, sent_sim_file: str = 'sent_sim.py') -> None:
    text = ' '.join(df['Utterance'])
    sents = sent_tokenize(text)
    sent_sim = np.load('sent_sim.npy')
    weighted_sents = Counter({
        sent: np.sum(sent_sim[i]) for i, sent in enumerate(sents)
    })
    for w_sent in weighted_sents.most_common(10):
        print(w_sent)
    breakpoint()


def sentence_clustering_tfidf(df: pd.DataFrame, calc_sim: bool = True, sim_thresh: float = 0.4) -> None:
    # Results are inferior to sentence transformer
    text = ' '.join(df['Utterance'])
    sents = sent_tokenize(text)
    from scipy.spatial.distance import pdist, squareform
    sent_vecs = TfidfVectorizer().fit_transform(sents).toarray()
    print('calculating sentence similarity')
    sent_sim = 1 - squareform(pdist(sent_vecs, metric='cosine'))
    g = nx.Graph()
    for i, sim_vec in enumerate(sent_sim): # only iterate through upper right triangle
        for j, sim in enumerate(sim_vec[i+1:]):
            if sim >= sim_thresh:
                g.add_edge(sents[i], sents[i+1+j])
    nx.write_graphml(g, 'sent_graph.graphml')
    for cc in nx.connected_components(g):
        print(cc)


def sentence_uniqueness(df: pd.DataFrame) -> None:
    print('\nSentences ranked by uniqeness')
    text = ' '.join(df['Utterance'])
    sents = sent_tokenize(text)
    if not os.path.exists('sent_sim.npy'):
        sentence_clustering(df)
    sent_dists = 1 - np.load('sent_sim.npy')
    dists = list()
    for i, dist_vec in enumerate(sent_dists):
        if i > len(sents)/2:
            break
        dists.append((np.min(dist_vec[i+1:]), sents[i]))
    dists.sort(reverse=True)
    for sent in dists[:25]:
        print(sent)
    breakpoint()


def get_collocation_counts_from_sentences(df: pd.DataFrame, nlp, distance_decay: bool = False, valid_pos: set = None) -> tuple:
    text = ' '.join(df['Utterance'])
    doc = nlp(text)
    tok_lemmas = [t.lemma_ for t in doc]
    tok_counts = Counter(tok_lemmas)
    tok_pair_counts = Counter()
    if not valid_pos:
        valid_pos = {'NOUN'}
    for sent in doc.sents:
        for i, tok1 in enumerate(sent):  # TODO only regard tokens with certain POS?
            if tok1.pos_ in valid_pos:
                dist = 0
                for j, tok2 in enumerate(sent[i + 1:], 1):
                    if tok2.pos_ in valid_pos:
                        dist += 1
                        if not tok2.lemma_ == tok1.lemma_:
                            if distance_decay:
                                tok_pair_counts[(tok1.lemma_, tok2.lemma_)] += 1 / dist
                            else:
                                tok_pair_counts[(tok1.lemma_, tok2.lemma_)] += 1
    return tok_counts, tok_pair_counts


def get_pmi_from_sentences(df: pd.DataFrame, nlp) -> Counter:
    tok_counts, tok_pair_counts = get_collocation_counts_from_sentences(df, nlp, distance_decay=False)
    max_count = tok_counts.most_common(1)[0][1]
    tok_probs = Counter({w: c / max_count for w, c in tok_counts.items()})
    max_pair_count = tok_pair_counts.most_common(1)[0][1]
    tok_pair_probs = Counter({w: c / max_pair_count for w, c in tok_pair_counts.items()})
    pmi_pairs = Counter({tok_pair: tok_pair_prob / (tok_probs[tok_pair[0]] + tok_probs[tok_pair[1]])
                         for tok_pair, tok_pair_prob in tok_pair_probs.items()
                         if (tok_counts[tok_pair[0]] > 1 and tok_counts[tok_pair[1]] > 1)}
                        )
    g = nx.DiGraph()
    for tok_pair, pmi in pmi_pairs.items():
        g.add_edge(tok_pair[0], tok_pair[1], weight=pmi)
    nx.write_graphml(g, 'pmi_graph.graphml')
    return pmi_pairs


def get_word_frequencies(df: pd.DataFrame, nlp, pos_selection: set = None) -> Counter:
    if not pos_selection:
        pos_selection = {'NOUN'}
    text = ' '.join(df['Utterance'])
    doc = nlp(text)
    filtered_doc = [tok.lemma_ for tok in doc if tok.pos_ in pos_selection]
    return Counter(filtered_doc)


def rank_sentences_per_speaker(df: pd.DataFrame, nlp, keywords: Counter) -> None:
    df['nlp'] = None
    sent_scores_per_speaker = defaultdict(Counter)
    for i, turn in df.iterrows():
        doc = nlp(turn['Utterance'])
        df.at[i, 'nlp'] = doc
        for sent in doc.sents:
            noun_lemmas = [tok.lemma_ for tok in sent if tok.pos_ == 'NOUN']
            score = sum([keywords[noun] for noun in set(noun_lemmas)])
            sent_scores_per_speaker[turn['Speaker']][sent.text] = score

    norm_sents_scores_per_speaker =defaultdict(Counter)
    for speaker, sents in sent_scores_per_speaker.items():
        max_score = max(sents.values())
        for sent, score in sents.items():
            norm_sents_scores_per_speaker[speaker][sent] = score / max_score

    with open('weighted_sentences_per_speaker.html', 'w') as f:
        f.write('<html>\n<table border=1>\n')
        for _, turn in df.iterrows():
            f.write('<tr><td>' + turn['Speaker'] + '</td><td>')
            for sent in turn['nlp'] .sents:
                font_size = max([11, 16 * norm_sents_scores_per_speaker[turn['Speaker']][sent.text]])
                """
                transparency = max([0.2, norm_sents_scores_per_speaker[turn['Speaker']][sent.text]])
                f.write('<span style="color:  rgba(0, 0, 0, ' + str(transparency) + ')">' + sent.text + ' </span>')
                """
                if font_size > 11:
                    f.write('<span style="font-size:' + str(int(font_size)) + '; font-weight: bold;">' + sent.text + ' </span>')
                else:
                    f.write('<span style="font-size:' + str(int(font_size)) + '">' + sent.text + ' </span>')

            f.write('</td></tr>\n')
        f.write('</table></html>')


def rank_sentences_w_keywords(df: pd.DataFrame, nlp, keywords: Counter) -> Counter:
    text = ' '.join(df['Utterance'])
    sents = sent_tokenize(text)
    weighted_sents = Counter()
    weighted_sents_per_keyword = defaultdict(Counter)
    lemmata_per_sent = dict()

    for sent in sents:
        doc = nlp(sent)
        noun_lemmas = [tok.lemma_ for tok in doc if tok.pos_ == 'NOUN']
        lemmata_per_sent[sent] = set(noun_lemmas)
        # score = sum([keywords[noun] for noun in noun_lemmas])
        score = sum([keywords[noun] for noun in set(noun_lemmas)])  # use set(noun_lemmas) to dampen the length effect of the sentence on the score
        weighted_sents[sent] = score
        for keyword, _ in keywords.most_common():
            if keyword in sent:
                weighted_sents_per_keyword[keyword][sent] = score

    print('\n25 highest ranked sentences based on keyword weights sum')
    for s, c in weighted_sents.most_common(25):
        print()
        keyword_weights = [(tok, keywords[tok]) for tok in lemmata_per_sent[s]]
        keyword_weights.sort(reverse=True, key=lambda x: x[1])
        print(s, c)
        print(keyword_weights)

    print('\n10 highest ranked sentences per keyword')
    for keyword, _ in keywords.most_common(10):
        print('\n' + keyword)
        for sent, score in weighted_sents_per_keyword[keyword].most_common(5):
            print(sent, score)

    return weighted_sents


def normalize_vals(counts: Counter, get_norm_val = max) -> Counter:
    max_val = get_norm_val(counts.values())
    return Counter({k: v / max_val for k, v in counts.most_common()})


def reweigh_word_frequencies_with_wordnet(keywords: Counter) -> Counter:
    """Take avg distance to root as indicator of specificity"""
    #  https://www.nltk.org/api/nltk.corpus.reader.wordnet.html#nltk.corpus.reader.wordnet.Synset.hypernym_paths
    from nltk.corpus import wordnet as wn

    avg_path_lenght_to_root = Counter()
    for keyword, score in keywords.most_common():
        synsets = wn.synsets(keyword, pos='n')  # TODO change pos restrictions if we allow verbs etc.
        if synsets:
            path_lengths_to_root = list()
            for synset in synsets:
                path_length_to_root = np.mean([len(path_to_root) for path_to_root in synset.hypernym_paths()])
                path_lengths_to_root.append(path_length_to_root)
            avg_path_lenght_to_root[keyword] = np.mean(path_lengths_to_root)
        else:
            avg_path_lenght_to_root[keyword] = 0  # TODO hm, what to do with stuff that isn't in wordnet?

    keywords_norm = normalize_vals(keywords)
    avg_path_lenght_to_root_norm = normalize_vals(avg_path_lenght_to_root)
    re_weighted_keywords = Counter(
        {k: keywords_norm[k] * avg_path_lenght_to_root_norm[k] for k in keywords.keys()}
    )
    return re_weighted_keywords


def reweigh_word_frequencies_with_chisquare(keywords: Counter, df: pd.DataFrame, nlp,
                                            normalize: bool = True) -> Counter:
    from scipy.stats import chisquare
    tok_counts, tok_pair_counts = get_collocation_counts_from_sentences(df, nlp, distance_decay=False)
    chi2_collocation_counts = Counter()
    re_weighted_keywords = Counter()
    for keyword, cnt in keywords.most_common():
        colloc_counts = [v for k, v in tok_pair_counts.most_common() if keyword in k]
        if len(colloc_counts) < len(keywords):
            colloc_counts.extend(np.zeros(len(keywords) - len(colloc_counts)).tolist())
        chi2_collocation_counts[keyword] = chisquare(colloc_counts).statistic
        re_weighted_keywords[keyword] = tok_counts[keyword] * chi2_collocation_counts[keyword]

    if normalize:
        tok_counts_norm = normalize_vals(tok_counts)
        chi2_collocation_counts_norm = normalize_vals(chi2_collocation_counts)
        re_weighted_keywords = Counter(
            {k: tok_counts_norm[k] * chi2_collocation_counts_norm[k] for k in keywords.keys()}
        )

    return re_weighted_keywords


def reweigh_word_frequencies_with_avg_pmi(keywords: Counter, df: pd.DataFrame, nlp,
                                            normalize: bool = True) -> Counter:
    pmi_pairs = get_pmi_from_sentences(df, nlp)
    if normalize:
        pmi_pairs = normalize_vals(pmi_pairs)
    avg_pmi_per_keyword = Counter()
    for keyword, cnt in keywords.most_common():
        pmi_scores = [v for k, v in pmi_pairs.most_common() if keyword in k]
        avg_pmi_per_keyword[keyword] = np.mean(pmi_scores)

    re_weighted_keywords = Counter(
        {k: keywords[k] * avg_pmi_per_keyword[k] for k in keywords.keys()}
    )

    return re_weighted_keywords


def write_sentence_weighted_transcript(df: pd.DataFrame, weighted_sentences: Counter, nlp) -> None:
    max_weight = max(weighted_sentences.values())
    font_scaling = 20 / max_weight
    with open('weighted_sentences.html', 'w') as f:
        f.write('<html>\n<table border=2>')
        for _, turn in df.iterrows():
            f.write('<tr><td>' + turn['Speaker'] + '</td><td>')
            for sent in nlp(turn['Utterance']).sents:
                font_size = max([weighted_sentences[sent.text] * font_scaling, 11])
                if font_size > 11:
                    f.write('<span style="font-size: ' + str(int(font_size)) + '; font-weight: bold">' + sent.text + ' </span>')
                else:
                    f.write('<span style="font-size: ' + str(int(font_size)) + '">' + sent.text + ' </span>')
            f.write('</tr>\n')
        f.write('</table>\n</html>')


def write_keyword_weighted_transcript(df, keywords, nlp) -> None:
    max_weight = max(keywords.values())
    font_scaling = 20 / max_weight
    with open('weighted_words.html', 'w') as f:
        f.write('<html>\n<table border=2>')
        for _, turn in df.iterrows():
            f.write('<tr><td>' + turn['Speaker'] + '</td><td>')
            for tok in nlp(turn['Utterance']):
                font_size = max([keywords[tok.text] * font_scaling, 11])
                if font_size > 11:
                    f.write('<span style="font-size: ' + str(int(font_size)) + '; font-weight: bold">' + tok.text + ' </span>')
                else:
                    f.write('<span style="font-size: ' + str(int(font_size)) + '">' + tok.text + ' </span>')
            f.write('</tr>\n')
        f.write('</table>\n</html>')


def main(csv_file: str, source: str = 'interscriber') -> None:
    if source == 'interscriber':
        df = pd.read_csv(csv_file)  # for Interscriber
    elif source == 'icsi':
        df = pd.read_csv(csv_file, names=['Speaker', 'Utterance'], sep='\t')
    else:
        raise NotImplementedError

    nlp = spacy.load('en_core_web_sm')
    #nlp = spacy.load('de_core_news_sm')

    """
    Approach to sentence weighting:
    1. get keywords by frequency (use pos tagging and only get verbs and nouns?)
    1.5. reweighing: weigh a word by the diversity of collocates it has: the more collocates, the less specific
    1.5 reweighing: calculate PMI or similarity/associativity with a keyword to its collocates? Use avg word embedding similarity between (content) words in a sentences?
    2. score each sentence by the number of keywords they contain (weight keywords by rank or similar)
    """

    #sentence_ranking_by_similarity(df)
    #dep_collocates_of_keywords(df, nlp, ['meeting', 'docker', 'voice', 'summary'])
    # summary_per_speaker(df)

    # get_pmi_from_sentences(df, nlp)
    keywords = get_word_frequencies(df, nlp)
    keywords_norm = normalize_vals(keywords)
    rank_sentences_per_speaker(df, nlp, keywords_norm)
    # write_keyword_weighted_transcript(df, keywords_norm, nlp)
    #re_weigthed_keywords_pmi = reweigh_word_frequencies_with_avg_pmi(keywords, df, nlp)
    #re_weigthed_keywords_chi = reweigh_word_frequencies_with_chisquare(keywords, df, nlp)
    #re_weigthed_keywords_wn = reweigh_word_frequencies_with_wordnet(keywords)
    weighted_sentences = rank_sentences_w_keywords(df, nlp, keywords)
    breakpoint()
    write_sentence_weighted_transcript(df, weighted_sentences, nlp)

    # cluster_key_words(df, ['meeting', 'summary', 'docker', 'transcript', 'vista', 'microsoft'])
    # frequent_noun_chunks(df, nlp, add_speakers_to_graph=False, selection_method='verb_phrases')
    # frequent_noun_chunks(df, nlp, add_speakers_to_graph=False, selection_method='kwic')

    # sentence_clustering(df, calc_sim=True)
    # sentence_uniqueness(df)
    # sentence_clustering_tfidf(df)
    # summary_per_speaker(df)
    # pmi_keyphrases(df)
    # rake_keyphrases(df)
    # rake_keyphrases_per_speaker(df)
    # tfidf_noun_chunks(df, nlp)
    # keyphrases_per_speaker_tfidf(df, nlp)
    # tfidf_sentences(df)
    # kws = get_bart_keywords(df)

    """
    # Start in the middle, calc entropy per split, move split, observe change in entropy
    text = ' '.join(df['Utterance'])
    sentences = sent_tokenize(text)
    """


if __name__ == '__main__':
    csv_file = '/home/don/projects/interscriber/data/Interviews/2021 07 12 Interscriber Meeting with Transcript/2021-07-12 14.35.38 Interscriber Wrapup.m4a.csv'
    #csv_file = '/home/don/projects/interscriber/data/PA meeting 3.11_Don_audio.mp4.csv'
    # csv_file = '/home/don/projects/interscriber/data/Interviews/US Election Debates/us_election_2020_2nd_presidential_debate.csv'
    # csv_file = '/home/don/projects/interscriber/dialogue_summarization/datasets/icsi/data/cleaned/valid/Bed011.txt'
    main(csv_file, source='interscriber')