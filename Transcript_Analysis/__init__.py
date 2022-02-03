import os
from Transcript_Analysis.utils.Autocomplete import Meeting_Autocomplete
import nltk


nltk.download('punkt')
nltk.download('stopwords')
os.system('python -m spacy download en_core_web_sm')
