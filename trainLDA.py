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
    tokens = word_tokenize(str(text))
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

vectorizer = CountVectorizer()
texts_vec = vectorizer.fit_transform(texts)
dump(vectorizer, model_dir + 'vectorizer_lda_V2.pkl')

log_likelihood = []
perplexity = []

for x in np.arange(7, 11):
    print(x)
    lda = LatentDirichletAllocation(n_components=x, random_state=42)
    lda.fit(texts_vec)
    dump(lda, model_dir + 'lda_model_{}_V2.pkl'.format(x))
    log_l = lda.score(texts_vec)
    print("Log Likelihood (n_comp: {}): {}".format(x, log_l))
    log_likelihood.append(log_l)
    perplex = lda.perplexity(texts_vec)
    print("Perplexity (n_comp: {}): {}".format(x, perplex))
    perplexity.append(perplex)

plt.plot(log_likelihood)
plt.show()

plt.plot(perplexity)
plt.show()

#lda =  LatentDirichletAllocation(n_components=15, random_state=42)
#if os.path.exists(aux_dir + "X_picture.npy"):
#    X_picture =np.load(aux_dir + "X_picture.npy")
#else:
#    X_picture = lda.fit_transform(X_vec)
#    np.save(aux_dir + "X_picture.npy", X_picture)
#print("picture")

#if show_lda_panel: #only possible if using jupyter
#    pyLDAvis.enable_notebook()
#    panel = pyLDAvis.sklearn.prepare(lda, X_vec, vectorizer, mds='tsne')
#    panel