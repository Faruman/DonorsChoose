from dask import dataframe as dd
from dask.distributed import Client
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from os import listdir
import unicodedata
import json
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def normalize_string(text: str) -> str:
    return unicodedata.normalize('NFD', str(text)).encode('latin-1', 'ignore')

def convert_bool(text: str) -> bool:
    if text.lower() == "yes":
        return True
    else:
        return False

sentiment_analyzer = SentimentIntensityAnalyzer()
def get_polarity(row, analyzer) -> tuple:
    polScores_dict = analyzer.polarity_scores(row["Project Essay"])
    return (polScores_dict["pos"], polScores_dict["neu"], polScores_dict["neg"])

dtype_dict = {
    "str": str,
    "bool": bool,
    "float": float,
    "int": int,
    "date": np.datetime64
}

client = Client(processes=False, threads_per_worker=4)

sql_uri = 'mysql://administrator:ARLN*??yDT-FsE-FgFXQxN4RmxRde3f3@datadriventransformationdb.cg5tribykeng.us-east-1.rds.amazonaws.com/donorschoose'

root_dir = "D:\Programming\Python\DonorsChoose\data\DonorsChoose/"


with open(root_dir + "DonorsDtype.json") as f:
    column_dict = json.load(f)

for file in [filename for filename in listdir(root_dir) if filename.endswith(".csv")]:
    print(file)

    columns = pd.read_csv(root_dir + file, nrows=0).columns
    column_dtypes = [dtype_dict[x] for x in [column_dict[x] for x in list(columns)]]

    #dtype_dummy_dict = dict(zip(columns, ['object']*len(columns)))
    #df = dd.read_csv(root_dir + file, dtype=dtype_dummy_dict, engine="python", encoding="utf-8", error_bad_lines=False)
    #df = dd.read_csv(root_dir + file, dtype=dtype_dummy_dict, delimiter=',', encoding='utf-8', header=0, quoting=csv.QUOTE_NONE, error_bad_lines=False)

    df = pd.read_csv(root_dir + file)
    df = dd.from_pandas(df, npartitions=4)

    for column in columns:
        if column_dict[column] in ["int", "float"]:
            df[column] = dd.to_numeric(df[column], errors='coerce')
        elif column_dict[column] == "date":
            df[column] = dd.to_datetime(df[column])
        elif column_dict[column] == "bool":
            df[column] = df[column].apply(convert_bool, meta=(column, 'bool'))
        else:
            df[column] = df[column].apply(normalize_string, meta=(column, 'object'))

        df[column].astype(dtype_dict[column_dict[column]])

        if column == "Project Essay":
            res = df.apply(get_polarity, analyzer=sentiment_analyzer, axis=1, result_type='expand', meta={'Project_Essay_Sentiment_positive': 'float64', 'Project_Essay_Sentiment_neutral': 'float64', 'Project_Essay_Sentiment_negative': 'float64'})
            full = df.merge(res, left_index=True, right_index=True)

            # eventually implement lda to find topics
            # eventually implement liwc scoring

    dd.DataFrame.to_sql(df, name= file[:-4], uri=sql_uri, index=False, if_exists='replace', method="multi", chunksize=1000, parallel=True)

