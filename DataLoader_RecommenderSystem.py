import os

import pandas as pd
import numpy as np

import settings

import gc

import fasttext

from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

import logging
from tqdm import tqdm
tqdm.pandas()


class DataLoader():
    def __init__(self):
        self.data = pd.DataFrame()
        self.interactions = pd.DataFrame()
        self.donors = list()
        self.donations = list()
        self.schools = list()
        self.projects = list()
        self.external = list()
        self.settings = settings

    def load_from_file(self, donors_path: str, donations_path: str, schools_path: str, projects_path: str, external_path: str):
        logging.info("DataLoader - Load data from csvs")
        ## load the datasets
        donors = pd.read_csv(donors_path)
        donors['Donor Zip'] = pd.to_numeric(donors['Donor Zip'], errors= "coerce")
        donors = donors.dropna(subset=["Donor Zip"])
        donors['Donor Zip'] = donors['Donor Zip'].astype(int).astype(str)
        self.donors = list(donors.columns)
        donations = pd.read_csv(donations_path)
        self.donations = list(donations.columns)
        schools = pd.read_csv(schools_path)
        self.schools = list(schools.columns)
        projects = pd.read_csv(projects_path)
        self.projects = list(projects.columns)
        ## Merge donations and donors
        temp1 = pd.merge(donations, donors, on='Donor ID', how='inner')
        ## Merge projects and schools
        temp2 = pd.merge(projects, schools, on='School ID', how='inner')
        ## Merge to master data frame
        self.data = pd.merge(temp1, temp2, on='Project ID', how='inner')

        ## integrate the external data
        external = pd.read_csv(external_path).dropna()
        external['id'] = external['id'].astype(int).astype(str)
        external = external.groupby('id').mean().reset_index()
        self.external = list(external.columns)

        self.data = pd.merge(self.data, external, left_on='Donor Zip', right_on='id', how="left")
        self.data = self.data.dropna(subset=["Donor Zip"])
        #TODO: Add additional dataframes
        
        del temp1, temp2, donors, donations, schools, projects, external
        gc.collect()

    def do_preprocessing(self, filter: dict= None):
        logging.info("DataLoader - Do preprocessing")
        ## Create some additional features in projects data
        self.data['cost'] = self.data['Project Cost'].apply(lambda x: float(str(x).replace("$", "").replace(",", "")))
        self.data['Posted Date'] = pd.to_datetime(self.data['Project Posted Date'])
        self.data['Posted Year'] = self.data['Posted Date'].dt.year
        self.data['Posted Month'] = self.data['Posted Date'].dt.month
        if filter:
            for key in filter.keys():
                self.data = self.data.loc[self.data[key].isin(filter[key])]
        self.data['Project Essay'] = self.data['Project Essay'].fillna(" ")
        self.data['Project Type'] = self.data['Project Type'].fillna("Teacher Led")
        self.data['Project Subject Category Tree'] = self.data['Project Subject Category Tree'].fillna(" ")
        
        #TODO: Add additional dataframes to data data
        #TODO: Add length of project to dataData


    def create_embeddings(self, embedding_type= settings.EMBEDDING_TYPE):
        logging.info("DataLoader - Create embeddings")
        if embedding_type == "fasttext":
            embeddingModel = fasttext.load_model(self.settings.EMBEDDING_FILE_FASTTEXT)
            def generate_doc_vectors(s):
                words = str(s).lower().split()
                words = [w for w in words if w.isalpha()]
                M = []
                for w in words:
                    M.append(embeddingModel.get_word_vector(w))
                v = np.array(M).sum(axis=0)
                if type(v) != np.ndarray:
                    return np.zeros(300)
                return v / np.sqrt((v ** 2).sum())

            temp = self.data[["Project ID", "Project Essay"]]
            temp = temp.groupby("Project ID").first().reset_index()
            temp["Project Essay Embedding"] = temp['Project Essay'].progress_apply(generate_doc_vectors)
            temp = temp.drop('Project Essay', axis= 1)
            self.data = pd.merge(self.data, temp, on="Project ID", how= "left")
            # TODO parallelize this step for higher speed pandarallel

        else:
           Exception("Embedding needs to be defined, adjust settings.py")

    def create_clustering(self, search_optimal_size:bool = False, clustering_type= settings.CLUSTERING_TYPE):
        logging.info("DataLoader - Create clusters")
        if 'area_context_cluster' in self.external:
            features = list(set(self.external) - set(['id', 'area_context_cluster']))
        else:
            features = list(set(self.external) - set(['id']))
        if clustering_type == "KMeans":
            if search_optimal_size == True:
                inretia = []
                min_clust = 2
                max_clust = 11
                for i in range(min_clust, max_clust):
                    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
                    kmeans.fit(self.data[features].fillna(0))
                    inretia.append(kmeans.inertia_)
                plt.plot(range(min_clust, max_clust), inretia)
                plt.title('Finding the Optimal Number of Clusters')
                plt.xlabel('Number of clusters')
                plt.xlabel('Kmeans Inretia')
                plt.show()

            kmeans = KMeans(n_clusters= settings.KMEANS_CLUSTERS, init='k-means++', max_iter=300, n_init=10, random_state=0)
            self.data['area_context_cluster'] = kmeans.fit_predict(self.data[features].fillna(0))
            if not 'area_context_cluster' in self.external:
                self.external.append('area_context_cluster')
            self.data['area_context_cluster'] = self.data['area_context_cluster'].astype(str)

        else:
            Exception("Cluster needs to be defined, adjust settings.py")

    def create_interactions(self):
        self.interactions = self.data[['Project ID', 'Donor ID', 'Donation Amount']]
        unique_donors = pd.DataFrame(self.interactions['Donor ID'].unique(), columns=['Donor ID']).reset_index().rename(columns={'index': 'user_id'})
        unique_projs = pd.DataFrame(self.interactions['Project ID'].unique(), columns=['Project ID']).reset_index().rename(columns={'index': 'proj_id'})
        self.interactions = self.interactions.merge(unique_donors, how="left", on="Donor ID").merge(unique_projs, how="left", on="Project ID")

    def return_master_data(self):
        return self.data

    def save_master_data(self, folder_path: str):
        self.data.to_hdq(os.path.join(folder_path + 'master_data.parquet.gzip'), compression='gzip')

    def return_interactions_data(self):
        return self.interactions

    def save_interactions_data(self, folder_path: str):
        self.interactions.to_hdq(os.path.join(folder_path + 'interactions_data.parquet.gzip'), compression='gzip')



