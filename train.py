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
    if not (os.path.isfile("./data/interactions_data.pkl.gz") & os.path.isfile("./data/master_data.pkl.gz")):
        dataloader = DataLoader()
        #dataloader.load_from_file(donations_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donations.csv",
        #                          donors_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donors.csv",
        #                          projects_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\projects.csv",
        #                          schools_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\schools.csv",
        #                          external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
        dataloader.load_from_file(donations_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donation_sample_V2.csv",
                                  donors_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donor_sample_V2.csv",
                                  projects_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\project_sample_V2.csv",
                                  schools_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\school_sample_V2.csv",
                                  external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
        dataloader.do_preprocessing()
        #dataloader.do_preprocessing(filter={'Project Current Status': ['Fully Funded']})
        dataloader.create_embeddings(args["embedding"])
        dataloader.create_clustering()

        data = dataloader.return_master_data()
        dataloader.save_master_data(folder_path='./data/')

        dataloader.create_interactions()
        dataloader.filter_interactions(2)
        dataloader.create_negative_interactions(1)

        interactions = dataloader.return_interactions_data()
        dataloader.save_interactions_data(folder_path= './data/')

    else:
        data = pd.read_pickle("./data/master_data_{}.pkl.gz".format(args["embedding"]))
        interactions = pd.read_pickle("./data/interactions_data.pkl.gz")

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