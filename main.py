import os

import pandas as pd

from DataLoader_RecommenderSystem import DataLoader
from base_RecommenderSystem import baseRecommender as BaselineRecommender

import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_auc_score

import gc


if not (os.path.isfile("./data/interactions_data.pkl") & os.path.isfile("./data/master_data.pkl")):
    dataloader = DataLoader()
    dataloader.load_from_file(donations_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donations.csv",
                              donors_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\donors.csv",
                              projects_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\projects.csv",
                              schools_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\schools.csv",
                              external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
    #dataloader.load_from_file(donations_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donation_sample_V2.csv",
    #                          donors_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donor_sample_V2.csv",
    #                          projects_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\project_sample_V2.csv",
    #                          schools_path=r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\school_sample_V2.csv",
    #                          external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
    dataloader.do_preprocessing(filter={'Project Current Status': ['Fully Funded']})
    #dataloader.create_embeddings()
    #dataloader.create_clustering()

    #data = dataloader.return_master_data()
    #dataloader.save_master_data(folder_path='./data/')

    dataloader.create_interactions()
    dataloader.create_negative_interactions(1)

    interactions = dataloader.return_interactions_data()
    #dataloader.save_interactions_data(folder_path= './data/')

else:
    data = pd.read_pickle("./data/master_data.pkl")
    interactions = pd.read_pickle("./data/interactions_data.pkl")


max_donorid = interactions['user_id'].drop_duplicates().max() + 1
max_projid = interactions['proj_id'].drop_duplicates().max() + 1

baseline_recommender = BaselineRecommender(max_donorid, max_projid, 20, device='cuda:0', learning_rate=1e-5)

def scoring(amount: float):
    if amount > 0:
        return 1
    else:
        return 0

baseline_recommender.load_data(interactions)
baseline_recommender.generate_objective(scoring)
baseline_recommender.generate_dataLoader(batch_size=1024)
baseline_recommender.train_model(2, "./model/")
#baseline_recommender.load_model('./model/baseLineRecommender.pt')
prediction_df = baseline_recommender.evaluate_model()

#plot the results
#plot precision recall plot
def plot_auc(label, score, title):
    precision, recall, thresholds = precision_recall_curve(label, score)
    plt.figure(figsize=(15, 5))
    plt.grid()
    plt.plot(thresholds, precision[1:], color='r', label='Precision')
    plt.plot(thresholds, recall[1:], color='b', label='Recall')
    plt.gca().invert_xaxis()
    plt.legend(loc='lower right')

    plt.xlabel('Threshold (0.00 - 1.00)')
    plt.ylabel('Precision / Recall')
    _ = plt.title(title)
    plt.savefig('./plots/prec_recall_curve_baselineModel.png')
    plt.show()

rocauc_score =  roc_auc_score(prediction_df['target'], prediction_df['prob'])
plot_auc(prediction_df['target'], prediction_df['prob'], "Baseline recommender sample - ROC AUC: {}".format(roc_auc_score))
