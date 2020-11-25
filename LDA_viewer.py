import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from matplotlib import pyplot as plt
import pyLDAvis
from nltk import word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer
stemming = SnowballStemmer("english")
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import stopwords
stops = set(stopwords.words("english"))
from joblib import dump, load

def identify_tokens(text):
    tokens = word_tokenize(text)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def do_stemming(inlist):
    return [stemming.stem(word) for word in inlist]

def remove_stops(inlist):
    return [w for w in inlist if not w in stops]

def do_join(inlist):
    return " ".join(inlist)

root_dir = "D:\Programming\Python\DonorsChoose\data\DonorsChoose/"
model_dir = "D:\Programming\Python\DonorsChoose\model/"

df = pd.read_csv(root_dir + "Projects.csv")

texts = df["Project Essay"].sample(n = 500000, random_state= 42)

# apply lda to approximate picture
texts= texts.apply(identify_tokens)
texts= texts.apply(do_stemming)
texts= texts.apply(remove_stops)
texts= texts.apply(do_join)

vectorizer = load(model_dir + 'vectorizer_lda.pkl')
texts_vec = vectorizer.transform(texts)

log_likelihood = []
perplexity = []

lda = load(model_dir + 'lda_model_6.pkl')

pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(lda, texts_vec, vectorizer, mds='tsne')
panel