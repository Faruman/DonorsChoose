import pandas as pd
import numpy as np
import sqlalchemy
from os import listdir
import unicodedata
import json
import math
import csv

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

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

def normalize_string(text: str) -> str:
    return unicodedata.normalize('NFD', str(text)).encode('latin-1', 'ignore').decode('latin-1')

def convert_bool(text: str) -> bool:
    if text.lower() == "yes":
        return True
    else:
        return False

sentiment_analyzer = SentimentIntensityAnalyzer()
def get_polarity(text: str, analyzer) -> list:
    polScores_dict = analyzer.polarity_scores(text)
    return [polScores_dict["pos"], polScores_dict["neu"], polScores_dict["neg"]]

def preprocess_data(df, column_dict, dtype_dict):
    for column in list(df.columns):
        if column_dict[column] == "int":
            df[column] = pd.to_numeric(df[column], downcast= "integer", errors='coerce')
        elif column_dict[column] == "float":
            df[column] = pd.to_numeric(df[column], downcast="float", errors='coerce')
        elif column_dict[column] == "date":
            df[column] = pd.to_datetime(df[column])
        elif column_dict[column] == "bool":
            df[column] = df[column].apply(convert_bool)
        else:
            df[column] = df[column].apply(normalize_string)

        if column == "Project Essay":
            polarity = df[column].apply(lambda x: sentiment_analyzer.polarity_scores(x)["compound"])
            #eventually implement lda to find topics
            #eventually implement liwc scoring
            df["sentiment"] = polarity
            df["sentiment"].astype(float)

            # apply lda to approximate picture
            texts = df[column].apply(identify_tokens)
            texts = texts.apply(do_stemming)
            texts = texts.apply(remove_stops)
            texts = texts.apply(do_join)

            vectorizer = load(model_dir + 'vectorizer_lda.pkl')
            texts_vec = vectorizer.transform(texts)

            lda = load(model_dir + 'lda_model_9.pkl')
            clusters = lda.transform(texts_vec)
            clusters = pd.DataFrame(clusters.transpose()).idxmax()
            df["cluster"] = clusters.apply(lambda x: cluster_9_dict[x]).values
            df["cluster"].astype(str)

        df[column].astype(dtype_dict[column_dict[column]])
    return df

dtype_dict = {
    "str": str,
    "bool": bool,
    "float": float,
    "int": int,
    "date": np.datetime64
}

cluster_12_dict = {
    0: "study supplies",
    1: "tech equipment I",
    2: "stem support",
    3: "books and literature",
    4: "furniture",
    5: "math and new technologies",
    6: "plants and environment",
    7: "tech equipment II",
    8: "languages I",
    9: "languages II",
    10: "support for disabled",
    11: "music and art"
}

cluster_9_dict = {
    0: "tech equipment",
    1: "study supplies",
    2: "tech-enabled learning",
    3: "stem support",
    4: "books and literature",
    5: "furniture",
    6: "sport and health",
    7: "languages",
    8: "music and art"
}



# get all files from DonorsChoose Table and add them to sql
root_dir = "D:\Programming\Python\DonorsChoose\data\DonorsChoose/"
model_dir = "D:\Programming\Python\DonorsChoose\model/"

with open(root_dir + "DonorsDtype.json") as f:
    column_dict = json.load(f)

project_df = pd.read_csv(root_dir + "Projects.csv").sample(n=100000, random_state=24)
project_df = preprocess_data(project_df, column_dict, dtype_dict)
print("Projects")

project_ids = project_df['Project ID'].values

resource_df = pd.read_csv(root_dir + "Resources.csv")
resource_df = resource_df.loc[resource_df['Project ID'].isin(project_ids)]
resource_df = preprocess_data(resource_df, column_dict, dtype_dict)
print("Resources")

donation_df = pd.read_csv(root_dir + "Donations.csv")
donation_df = donation_df.loc[donation_df['Project ID'].isin(project_ids)]
donation_df = preprocess_data(donation_df, column_dict, dtype_dict)
print("Donations")

donor_ids = donation_df['Donor ID'].values

donor_df = pd.read_csv(root_dir + "Donors.csv")
donor_df = donor_df.loc[donor_df['Donor ID'].isin(donor_ids)]
donor_df = preprocess_data(donor_df, column_dict, dtype_dict)
print("Donors")

school_ids = project_df['School ID'].values
teacher_ids = project_df['Teacher ID'].values

school_df = pd.read_csv(root_dir + "Schools.csv")
school_df = school_df.loc[school_df['School ID'].isin(school_ids)]
school_df = preprocess_data(school_df, column_dict, dtype_dict)
print("Schools")

teacher_df = pd.read_csv(root_dir + "Teachers.csv")
teacher_df = teacher_df.loc[teacher_df['Teacher ID'].isin(teacher_ids)]
teacher_df = preprocess_data(teacher_df, column_dict, dtype_dict)
print("Teachers")

donation_df.to_csv(root_dir + "/sample/donation_sample_V2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
project_df.to_csv(root_dir + "/sample/project_sample_V2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
resource_df.to_csv(root_dir + "/sample/resource_sample_V2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
donor_df.to_csv(root_dir + "/sample/donor_sample_V2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
school_df.to_csv(root_dir + "/sample/school_sample_V2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)
teacher_df.to_csv(root_dir + "/sample/teacher_sample_V2.csv", index=False, quoting=csv.QUOTE_NONNUMERIC)


