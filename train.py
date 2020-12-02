import os
import wandb

import pandas as pd

from DataLoader_RecommenderSystem import DataLoader
from base_RecommenderSystem import baseRecommender as BaselineRecommender

import json

import gc


with open(r"config.json") as f:
    args = json.load(f)
wandb.init(project="DonorsChoose", config=args)
args = wandb.config

def main():
    if args["use_sample"] == False:
        masterdata_path = os.path.join("./data/" + 'master_data_{}_{}.pkl.gz'.format(args["embedding"], args["clustering"]))
        interactionsdata_path = os.path.join("./data/" + "interactions_data_mp{}.pkl.gz".format(args["interactions_minProjectsperUser"]))
    else:
        masterdata_path = os.path.join("./data/sample/" + 'master_data_{}_{}.pkl.gz'.format(args["embedding"], args["clustering"]))
        interactionsdata_path = os.path.join("./data/sample/" + "interactions_data_mp{}.pkl.gz".format(args["interactions_minProjectsperUser"]))

    if not (os.path.isfile(masterdata_path) & os.path.isfile(interactionsdata_path)):
        dataloader = DataLoader()
        if args["use_sample"] == True:
            dataloader.load_from_file(donations_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donation_sample_V2.csv",
                                        donors_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donor_sample_V2.csv",
                                        projects_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\project_sample_V2.csv",
                                        schools_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\school_sample_V2.csv",
                                        external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
        else:
            dataloader.load_from_file(donations_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donations.csv",
                                        donors_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donors.csv",
                                        projects_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\projects.csv",
                                        schools_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\schools.csv",
                                        external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
        dataloader.do_preprocessing()
        if args["embedding"]:
            dataloader.create_embeddings(args["embedding"])

        if args["clustering"]:
            dataloader.create_clustering(clustering_type=args["clustering"])

        data = dataloader.return_master_data()

        dataloader.create_interactions()
        dataloader.filter_interactions(args["interactions_minProjectsperUser"]-1)
        dataloader.create_negative_interactions(1)

        interactions = dataloader.return_interactions_data()

        dataloader.save_master_data(data_path= masterdata_path)
        dataloader.save_interactions_data(data_path= interactionsdata_path)

    else:
        data = pd.read_pickle(masterdata_path)
        interactions = pd.read_pickle(interactionsdata_path)

    max_donorid = interactions['user_id'].drop_duplicates().max() + 1
    max_projid = interactions['proj_id'].drop_duplicates().max() + 1

    if args["model"] == "base":
        def scoring(amount: float):
            if amount > 0:
                return 1
            else:
                return 0

        baseline_recommender = BaselineRecommender(max_donorid, max_projid, args["embedding_dim"], args["linear_dim"], device='cuda:0', learning_rate=args["learning_rate"])
        baseline_recommender.load_data(interactions)
        baseline_recommender.generate_objective(scoring)
        baseline_recommender.generate_dataLoader(batch_size=args["batch_size"])
        baseline_recommender.train_model(args["num_train_epochs"], "./model/")
        #baseline_recommender.load_model('./model/baseLineRecommender.pt')
        prediction_df = baseline_recommender.evaluate_model()


if __name__ == "__main__":
    main()