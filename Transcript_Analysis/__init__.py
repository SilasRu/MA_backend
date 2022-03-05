import os
from Transcript_Analysis.data_types.Transcript import *
from Transcript_Analysis.extractive.Extractive import Extractive
from Transcript_Analysis.abstractive.Abstractive import Abstractive
from Transcript_Analysis.utils.utils import Utils
from Transcript_Analysis.custom_unsupervised_summarizer import Unsupervised_Summarizer
import nltk

print('**** Attempting to load the package ... **** ')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

os.system('python -m spacy download en_core_web_sm')
