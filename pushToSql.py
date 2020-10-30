import pandas as pd
import numpy as np
import sqlalchemy
from os import listdir
import unicodedata

def normalize_string(text: str) -> str:
    return unicodedata.normalize('NFD', str(text)).encode('latin-1', 'ignore')

engine = sqlalchemy.create_engine('mysql://administrator:ARLN*??yDT-FsE-FgFXQxN4RmxRde3f3@datadriventransformationdb.cg5tribykeng.us-east-1.rds.amazonaws.com/donorschoose')

# get all files from DonorsChoose Table and add them to sql
root_dir = "D:\Programming\Python\DonorsChoose\data\DonorsChoose/"
for file in listdir(root_dir):
    df = pd.read_csv(root_dir + file)

    #for debugging
    #df = df.iloc[:100, :]

    date_cols = [s for s in df.columns if "date" in s.lower()]
    for date_col in date_cols:
        df[date_col] = pd.to_datetime(df[date_col])
    int_cols = [s for s in df.columns if "zip" in s.lower()]
    for int_col in int_cols:
        df[int_col] = df[int_col].astype(np.float)
    df.loc[:, df.dtypes == object] = df.loc[:, df.dtypes == object].applymap(normalize_string)
    df.to_sql(file[:-4], con=engine, index=False, if_exists='replace')