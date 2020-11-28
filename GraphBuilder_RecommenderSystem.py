import os

import pandas as pd

import logging
from tqdm import tqdm
tqdm.pandas()


class GraphBuilder():
    def __init__(self):
        self.data = pd.DataFrame()

    def load_from_dataframe(self, df: pd.DataFrame):
        self.data= df

    def load_from_parquet(self, path_to_parquet):
        self.data = pd.read_parquet(path_to_parquet)

    def preprocess_data(self):
        def apply_func(group):
            print(group)
        self.data.groupby('Donor ID').apply(apply_func)
        self.user_profile = self.data.groupby('Donor ID').agg({'Donation Amount': ['min', 'max', 'mean', 'median'],
                                                          'cost': ['min', 'max', 'mean', 'median'],
                                                          'Project Subject Category Tree': lambda x: ", ".join(x),
                                                          'Project ID': lambda x: ",".join(x),
                                                          'School Metro Type': lambda x: ",".join(x),
                                                          'Project Title': lambda x: ",".join(x),
                                                          'area_context_cluster': lambda x: ",".join(x),
                                                          'School Percentage Free Lunch': 'mean',
                                                          'Project Grade Level Category': lambda x: ",".join(x),
                                                          'Project Type': 'count'}
                                                         ).reset_index().rename(columns={'Project Type': "Projects Funded"})

        #TODO: parallelize apply