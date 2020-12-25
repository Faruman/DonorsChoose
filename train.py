import os
import wandb

import pandas as pd

from DataLoader_RecommenderSystem import DataLoader
from base_RecommenderSystem import baseRecommender as BaselineRecommender
from advanced_RecommenderSystem import advancedRecommender as AdvancedRecommender

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
            dataloader.load_from_file(donations_path=args["data_path"] + r"\DonorsChoose\sample\donation_sample_V2.csv",
                                        donors_path=args["data_path"] + r"\DonorsChoose\sample\donor_sample_V2.csv",
                                        projects_path=args["data_path"] + r"\DonorsChoose\sample\project_sample_V2.csv",
                                        schools_path=args["data_path"] + r"\DonorsChoose\sample\school_sample_V2.csv",
                                        external_path=args["data_path"] + r"\EconomicIndicators\ZipCodes_AreaContext.csv")
        else:
            dataloader.load_from_file(donations_path=args["data_path"] + r"\DonorsChoose\donations.csv",
                                        donors_path=args["data_path"] + r"\DonorsChoose\donors.csv",
                                        projects_path=args["data_path"] + r"\DonorsChoose\projects.csv",
                                        schools_path=args["data_path"] + r"\DonorsChoose\schools.csv",
                                        external_path=args["data_path"] + r"\EconomicIndicators\ZipCodes_AreaContext.csv")
        dataloader.do_preprocessing()
        dataloader.filter_samples(1)
        if args["embedding"]:
            dataloader.create_embeddings(args["embedding"])

        if args["clustering"]:
            dataloader.create_clustering(clustering_type=args["clustering"])

        dataloader.quantify(["Donor ID", "Project ID"], args["model_path"] + "\labelEncoder", args["model_path"] + "\labelNormalizer", ['Teacher Project Posted Sequence', 'Project Cost', 'School Percentage Free Lunch', 'Population', 'Population Density','Housing Units', 'Median Home Value', 'Land Area', 'Water Area','Occupied Housing Units', 'Median Household Income'])
        dataloader.create_interaction_terms2(5)
        dataloader.create_negative_samples(1)

        dataloader.create_interactions()
        dataloader.filter_interactions(args["interactions_minProjectsperUser"]-1)

        interactions = dataloader.return_interactions_data()

        dataloader.save_master_data(data_path= masterdata_path)
        dataloader.save_interactions_data(data_path= interactionsdata_path)

    else:
        data = pd.read_pickle(masterdata_path)
        interactions = pd.read_pickle(interactionsdata_path)

    max_donorid = interactions['user_id'].drop_duplicates().max() + 1
    max_projid = interactions['proj_id'].drop_duplicates().max() + 1

    def scoring(amount: float):
        if amount > 0:
            return 1
        else:
            return 0

    if args["model"] == "base":
        baseline_recommender = BaselineRecommender(max_donorid, max_projid, args["base_embedding_dim"], args["base_linear_dim"], device='cuda:0', learning_rate=args["learning_rate"])
        baseline_recommender.load_data(interactions)
        baseline_recommender.generate_objective(scoring)
        baseline_recommender.generate_dataLoader(batch_size=args["batch_size"])
        baseline_recommender.train_model(args["num_train_epochs"], "./model/")
        prediction_df = baseline_recommender.evaluate_model()


    elif args["model"] == "advanced":
        project_columns =    ['Teacher Project Posted Sequence', 'Project Type',
                           'Project Subject Category Tree', 'Project Subject Subcategory Tree', 'Project Grade Level Category',
                           'Project Resource Category', 'Project Cost', 'School Metro Type', 'School Percentage Free Lunch', 'School State',
                           'School Zip', 'School County', 'School District', 'Posted Year', 'Posted Month', 'Project Essay Embedding']
        donor_columns =  ['Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip',
                           'Population', 'Population Density', 'Housing Units', 'Median Home Value', 'Land Area', 'Water Area',
                           'Occupied Housing Units', 'Median Household Income', 'area_context_cluster']
        project_history_column = ['previous projects']
        n_project_history = 318

        advanced_recommender =  AdvancedRecommender(donor_columns= donor_columns, donor_linear=args["advanced_donor_linear"], project_columns=project_columns, n_project_columns= len(project_columns)+299, project_linear=args["advanced_project_linear"], project_history_column=project_history_column, n_project_history=n_project_history, project_history_lstm_hidden=args["advanced_project_history_lstm_hidden"], linear1_dim=args["advanced_linear1_dim"], linear2_dim=args["advanced_linear2_dim"], device='cuda:0', learning_rate=args["learning_rate"])
        advanced_recommender.load_data(data)
        advanced_recommender.generate_objective(scoring)
        advanced_recommender.generate_dataLoader(batch_size=args["batch_size"])
        advanced_recommender.train_model(args["num_train_epochs"], args["model_path"])
        prediction_df = advanced_recommender.evaluate_model()

    print(prediction_df)
    #prediction_df.to_excel("evaluation_df.xlsx")

if __name__ == "__main__":
    main()