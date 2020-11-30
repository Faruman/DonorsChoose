import pandas as pd
import numpy as np
import sqlalchemy
from os import listdir
import unicodedata
import json
import math

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def normalize_string(text: str) -> str:
    return unicodedata.normalize('NFD', str(text)).encode('latin-1', 'ignore')

def convert_bool(text: str) -> bool:
    if text.lower() == "yes":
        return True
    else:
        return False

sentiment_analyzer = SentimentIntensityAnalyzer()
def get_polarity(text: str, analyzer) -> list:
    polScores_dict = analyzer.polarity_scores(text.decode('latin-1'))
    return [polScores_dict["pos"], polScores_dict["neu"], polScores_dict["neg"]]

dtype_dict = {
    "str": str,
    "bool": bool,
    "float": float,
    "int": int,
    "date": np.datetime64
}

engine = sqlalchemy.create_engine('mysql://administrator:ARLN*??yDT-FsE-FgFXQxN4RmxRde3f3@datadriventransformationdb.cg5tribykeng.us-east-1.rds.amazonaws.com/donorschoose')

# get all files from DonorsChoose Table and add them to sql
root_dir = "D:\Programming\Python\DonorsChoose\data\DonorsChoose/"
with open(root_dir + "DonorsDtype.json") as f:
    column_dict = json.load(f)

for file in [filename for filename in listdir(root_dir) if filename.endswith(".csv")]:
    print(file)

    df = pd.read_csv(root_dir + file)
    df = df.iloc[:10000, :]

    column_dtypes = [dtype_dict[x] for x in [column_dict[x] for x in list(df.columns)]]
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
            textAnalysis_colNames = ['Project_Essay_Sentiment_positive', 'Project_Essay_Sentiment_neutral', 'Project_Essay_Sentiment_negative']
            polarity = np.column_stack(df[column].apply(get_polarity, analyzer= sentiment_analyzer).values)
            for i, colName in enumerate(textAnalysis_colNames):
                df[colName] = polarity[i]
            #eventually implement lda to find topics
            #eventually implement liwc scoring

        df[column].astype(dtype_dict[column_dict[column]])

    df.to_sql(file[:-4], con=engine, index=False, if_exists='replace', method='multi')

    #chunk_size = 100000
    #error_chunks = []
    #df_array = np.array_split(df, int(df.shape[0]/chunk_size))
    #for i, df in enumerate(df_array):
    #    if i == 0:
    #        df.to_sql(file[:-4], con=engine, index=False, if_exists='replace', method='multi')
    #    else:
    #        try:
    #            df.to_sql(file[:-4], con=engine, index=False, if_exists='append', method='multi')
    #        except:
    #            error_chunks.append(i * chunk_size)

    #print(error_chunks)


