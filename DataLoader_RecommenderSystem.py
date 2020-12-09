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

import random

from transformers import DistilBertTokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

from sklearn.preprocessing import LabelEncoder

from pickle import dump

class DataLoader():
    def __init__(self, seed= 42):
        self.data = pd.DataFrame()
        self.interactions = pd.DataFrame()
        self.negativeInteractions = pd.DataFrame()
        self.donors = list()
        self.donations = list()
        self.schools = list()
        self.projects = list()
        self.external = list()
        self.settings = settings
        self.embedding_type = "none"
        self.seed = 42

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
        self.data = self.data.drop(["id"], axis= 1)
        #TODO: Add additional dataframes
        
        del temp1, temp2, donors, donations, schools, projects, external
        gc.collect()

    def do_preprocessing(self, filter: dict= None):
        logging.info("DataLoader - Do preprocessing")
        ## Create some additional features in projects data
        self.data['Project Cost'] = self.data['Project Cost'].apply(lambda x: float(str(x).replace("$", "").replace(",", "")))
        self.data['Posted Date'] = pd.to_datetime(self.data['Project Posted Date'])
        self.data['Posted Year'] = self.data['Posted Date'].dt.year.astype("int")
        self.data['Posted Month'] = self.data['Posted Date'].dt.month.astype("int")
        if filter:
            for key in filter.keys():
                self.data = self.data.loc[self.data[key].isin(filter[key])]
        self.data['Project Essay'] = self.data['Project Essay'].fillna(" ")
        self.data['Project Type'] = self.data['Project Type'].fillna("Teacher Led")
        self.data['Project Subject Category Tree'] = self.data['Project Subject Category Tree'].fillna(" ")
        self.data['Teacher Project Posted Sequence'] = self.data['Teacher Project Posted Sequence'].astype("int")
        #TODO: Drop all irrelevant columns
        self.data = self.data.drop(["Donation ID", "Donation Included Optional Donation", "Donation Received Date", 'Teacher ID', 'Project Posted Date', 'School City', 'School Name', 'School ID', 'Project Expiration Date', 'Posted Date', 'Project Current Status'], axis=1)
        #TODO: Add additional dataframes to data data
        #TODO: Add length of project to dataData
        #remove nan
        self.data = self.data.dropna(axis=0)

        ##self.data = self.data.sample(10000)

    def create_embeddings(self, embedding_type= settings.EMBEDDING_TYPE, embedding_columns= ["Project Essay"]) :
        self.embedding_type = embedding_type
        logging.info("DataLoader - Create embeddings")

        for embedding_column in embedding_columns:
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

                temp = self.data[["Project ID", embedding_column]]
                temp = temp.groupby("Project ID").first().reset_index()
                temp[embedding_column + " Embedding"] = temp[embedding_column].progress_apply(generate_doc_vectors)
                temp = temp.drop(embedding_column, axis= 1)
                self.data = pd.merge(self.data, temp, on="Project ID", how= "left")
                # TODO parallelize this step for higher speed pandarallel

            elif embedding_type == "DistilBERT":
                temp = self.data[["Project ID", embedding_column]]
                temp = temp.groupby("Project ID").first().reset_index()
                temp[embedding_column + " Embedding"] = temp[embedding_column].progress_apply(lambda x: tokenizer(x.lower())['input_ids'])
                temp = temp.drop(embedding_column, axis=1)
                self.data = pd.merge(self.data, temp, on="Project ID", how="left")

            else:
               Exception("Embedding needs to be defined, adjust settings.py")

        self.data = self.data.drop(["Project Essay", "Project Short Description", 'Project Title', 'Project Need Statement'], axis=1)

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
            self.data['area_context_cluster'] = self.data['area_context_cluster']

        else:
            Exception("Cluster needs to be defined, adjust settings.py")

    def create_interaction_terms(self, projects_back):
        #display projects a donor has donated for
        def get_previous_projects(row, projects_back):
            previousProjects = self.data[(self.data["Donor Cart Sequence"] < row["Donor Cart Sequence"]) & (self.data["Donor ID"] == row["Donor ID"])].drop(['Donor ID', 'Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip', 'Population', 'Population Density', 'Housing Units', 'Median Home Value', 'Land Area', 'Water Area', 'Occupied Housing Units', 'Median Household Income'], axis= 1)
            if not previousProjects.empty:
                previousProjects = previousProjects.sort_values("Donor Cart Sequence", ascending= False)
                if previousProjects.shape[0] > projects_back:
                    previousProjects = previousProjects.iloc[:projects_back, :]
                row["previous Projects"] = previousProjects
            else:
                row["previous Projects"] = pd.DataFrame()
            return row
        self.data = self.data[self.data["Donor Cart Sequence"] > 1].progress_apply(get_previous_projects, axis=1, args=(projects_back, ))
        self.data = self.data.drop(["Donor Cart Sequence"], axis= 1)

    def quantify(self, exclude, encoder_path):
        for column in list(set(self.data.columns) - set(exclude)):
            if (not self.data[column].dtype in [np.float, np.int]) and (not "Embedding" in column):
                encoder = LabelEncoder()
                self.data[column] = encoder.fit_transform(self.data[column].astype(str))
                dump(encoder, open(os.path.join(encoder_path, 'LabelEncoder_{}.pkl'.format(column)), 'wb'))


    def create_interaction_terms2(self, projects_back):
        #display projects a donor has donated for
        all_interactions = self.data.drop(['Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip', 'Population', 'Population Density', 'Housing Units', 'Median Home Value', 'Land Area', 'Water Area', 'Occupied Housing Units', 'Median Household Income', 'area_context_cluster'], axis= 1).sort_values(by=['Donor Cart Sequence'], ascending=True).groupby("Donor ID")
        all_interactions = all_interactions.progress_apply(lambda x: pd.Series({"previous projects cart sequence": x["Donor Cart Sequence"].values, "previous projects": np.array(x.drop(["Donor ID", "Project ID"], axis=1).values)}))
        all_interactions = all_interactions.reset_index()

        self.data = self.data.merge(all_interactions, on="Donor ID", how="left")

        def get_correct_length(row, projects_back):
            mask = row["previous projects cart sequence"] < row["Donor Cart Sequence"]
            if sum(mask) > projects_back:
                i = int(np.where(row["previous projects cart sequence"] == row["Donor Cart Sequence"])[0][0])
                mask[:(i-projects_back)] = False
            row["previous projects"] = row["previous projects"][mask]
            if np.all(row["previous projects"]==0):
                row["previous projects"] = None
            return row

        self.data = self.data.progress_apply(get_correct_length, args=(projects_back,), axis=1)
        self.data = self.data.dropna(subset=["previous projects"])
        self.data = self.data.drop(["previous projects cart sequence", "Donor Cart Sequence"], axis=1)

    def create_negative_samples(self, percentage):
        logging.info("DataLoader - Create negative samples")
        self.negativeSamples = pd.DataFrame(columns= self.data.columns)
        n_negative_sample = int(self.data.shape[0] * percentage)
        max_n_negative_samples_per_iteration = int(n_negative_sample/30)

        while n_negative_sample > 0:
            if n_negative_sample > max_n_negative_samples_per_iteration:
                negative_samples = pd.DataFrame(columns= ['Project ID', 'Donor ID', 'Donation Amount'], index=range(0, max_n_negative_samples_per_iteration))
                negative_samples['Project ID'] = random.choices(list(self.data["Project ID"].unique().tolist()), k=max_n_negative_samples_per_iteration)
                negative_samples['Donor ID'] = random.choices(list(self.data["Donor ID"].unique().tolist()), k=max_n_negative_samples_per_iteration)
            else:
                negative_samples = pd.DataFrame(columns=['Project ID', 'Donor ID', 'Donation Amount'], index=range(0, n_negative_sample))
                negative_samples['Project ID'] = random.choices(list(self.data["Project ID"].unique().tolist()), k=n_negative_sample)
                negative_samples['Donor ID'] = random.choices(list(self.data["Donor ID"].unique().tolist()), k=n_negative_sample)
            negative_samples = pd.merge(negative_samples, self.data.loc[:, ['Project ID', 'Donor ID']], on=['Project ID', 'Donor ID'], how="outer", indicator=True, suffixes=("", "_y")).query('_merge=="left_only"')
            negative_samples = negative_samples.drop(['_merge'], axis=1)
            negative_samples['Donation Amount'] = 0
            projects = self.data.sample(self.data.shape[0]).loc[:, ['Project ID', 'Teacher Project Posted Sequence', 'Project Type',
                                           'Project Subject Category Tree', 'Project Subject Subcategory Tree', 'Project Grade Level Category',
                                           'Project Resource Category', 'Project Cost', 'Project Fully Funded Date', 'School Metro Type', 'School Percentage Free Lunch', 'School State',
                                           'School Zip', 'School County', 'School District', 'Posted Year', 'Posted Month', 'Project Essay Embedding']].groupby(['Project ID']).first()
            donors = self.data.sample(self.data.shape[0]).loc[:, ['Donor ID', 'Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip',
                                       'Population', 'Population Density', 'Housing Units', 'Median Home Value', 'Land Area', 'Water Area',
                                       'Occupied Housing Units', 'Median Household Income', 'area_context_cluster', 'previous projects']].groupby(['Donor ID']).first()
            negative_samples = negative_samples.merge(projects, how="left", on="Project ID").merge(donors, how="left", on="Donor ID")
            self.negativeSamples = self.negativeSamples.append(negative_samples)
            n_negative_sample -= negative_samples.shape[0]

    def filter_samples(self, min_number_of_donations: int):
            samples_filter = self.data["Donor ID"].value_counts().loc[self.data["Donor ID"].value_counts() > min_number_of_donations].reset_index()
            self.data = self.data.loc[self.data["Donor ID"].isin(samples_filter["index"])]

    def create_interactions(self):
        self.interactions = self.data[['Project ID', 'Donor ID', 'Donation Amount']]
        unique_donors = pd.DataFrame(self.interactions['Donor ID'].unique(), columns=['Donor ID']).reset_index().rename(columns={'index': 'user_id'})
        unique_projs = pd.DataFrame(self.interactions['Project ID'].unique(), columns=['Project ID']).reset_index().rename(columns={'index': 'proj_id'})
        self.interactions = self.interactions.merge(unique_donors, how="left", on="Donor ID").merge(unique_projs, how="left", on="Project ID")
        self.donors_dict = dict(zip(unique_donors['user_id'], unique_donors['Donor ID']))
        self.projs_dict = dict(zip(unique_projs['proj_id'], unique_projs['Project ID']))

    def filter_interactions(self, min_number_of_donations: int):
        interactions_filter = self.interactions["user_id"].value_counts().loc[self.interactions["user_id"].value_counts() > min_number_of_donations].reset_index()
        self.interactions = self.interactions.loc[self.interactions["user_id"].isin(interactions_filter["index"])]

    def create_negative_interactions(self, percentage: float):
        logging.info("DataLoader - Create negative samples")
        self.negativeInteractions = pd.DataFrame(columns= ['Project ID', 'Donor ID', 'Donation Amount', 'user_id', 'proj_id'])
        n_negative_sample = int(self.interactions.shape[0] * percentage)

        while n_negative_sample > 0:
            negative_samples = self.interactions.sample(n_negative_sample, replace=True)
            negative_samples['proj_id'] = random.choices(list(self.projs_dict.keys()), k=n_negative_sample)
            negative_samples = pd.merge(negative_samples, self.interactions, on=['proj_id', 'user_id'], how="outer", indicator=True, suffixes=("", "_y")).query('_merge=="left_only"')
            negative_samples = negative_samples.drop(['Project ID_y', 'Donor ID_y', 'Donation Amount_y', '_merge'], axis=1)
            negative_samples['Donation Amount'] = 0
            negative_samples['Project ID'] = negative_samples['proj_id'].apply(lambda x: self.projs_dict[x])
            self.negativeInteractions = self.negativeInteractions.append(negative_samples)
            n_negative_sample -= negative_samples.shape[0]

        #less efficient way
        #interactions_group = self.interactions.groupby('user_id')['proj_id'].apply(list)
        #interactions_group = dict(zip(list(interactions_group.index), list(interactions_group)))
        #sample_set = set(self.projs_dict.keys())

        #def create_negative_sample(row, interactions_group, samples):
        #    positive_samples = interactions_group[row['user_id']]
        #    available_samples = samples - set(positive_samples)
        #    proj_id = random.sample(available_samples, 1)[0]
        #    row['Project ID'] = self.projs_dict[proj_id]
        #    row['proj_id'] = proj_id
        #    row['Donation Amount'] = 0
        #    return row

        #self.interactions.sample(n_negative_sample, replace=True).progress_apply(create_negative_sample, args=(interactions_group,sample_set), axis=1)

    def return_master_data(self):
        return self.data.append(self.negativeSamples)

    def save_master_data(self, data_path: str):
        self.data.append(self.negativeSamples).to_pickle(data_path, compression='gzip')

    def return_interactions_data(self):
        return self.interactions.append(self.negativeInteractions)

    def save_interactions_data(self, data_path: str):
        self.interactions.append(self.negativeInteractions).to_pickle(data_path, compression='gzip')

#masterdata_path = os.path.join("./data/" + 'master_data_{}_{}.pkl.gz'.format("fasttext", "KMeans"))

#dataloader = DataLoader()
#dataloader.load_from_file(donations_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donations.csv",
#                                        donors_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donors.csv",
#                                        projects_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\projects.csv",
#                                        schools_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\schools.csv",
#                                        external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
#dataloader.do_preprocessing()
#dataloader.filter_samples(1)
#dataloader.create_embeddings("fasttext")
#dataloader.create_clustering(clustering_type="KMeans")
#dataloader.quantify(["Donor ID", "Project ID"], "D:\Programming\Python\DonorsChoose\model\labelEncoder")
#dataloader.create_interaction_terms2(5)
#dataloader.create_negative_samples(1)

#dataloader.save_master_data(data_path=masterdata_path)


