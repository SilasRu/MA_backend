import os
from transcript_analyser.data_types.transcript import *
from transcript_analyser.extractive.extractive import Extractive
from transcript_analyser.abstractive.abstractive import Abstractive
from transcript_analyser.utils.utils import Utils
from transcript_analyser.custom_unsupervised_summarizer import UnsupervisedSummarizer
import nltk

print('**** Attempting to load the package ... **** ')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

os.system('python -m spacy download en_core_web_sm')
