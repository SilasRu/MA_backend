import os

from matplotlib.pyplot import quiver
from Transcript_Analysis.data_types.Transcript import *
import nltk


nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

os.system('python -m spacy download en_core_web_sm')
