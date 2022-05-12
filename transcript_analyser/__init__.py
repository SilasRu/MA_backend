import os
from transcript_analyser.data_types.transcript import *
from transcript_analyser.extractive.extractive import Extractive
from transcript_analyser.abstractive.abstractive import Abstractive
from transcript_analyser.utils.utils import Utils
from transcript_analyser.custom_unsupervised_summarizer import UnsupervisedSummarizer
import nltk
import spacy

print('**** Attempting to load the package ... **** ')
try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords', quiet=True)

try:
    nlp = spacy.load("en_core_web_sm")
except:
    os.system('python -m spacy download en_core_web_sm')
