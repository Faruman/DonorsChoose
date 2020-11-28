from DataLoader_RecommenderSystem import DataLoader
from GraphBuilder_RecommenderSystem import GraphBuilder

import gc


dataloader = DataLoader()
dataloader.load_from_file(donations_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donation_sample_V2.csv",
                          donors_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\donor_sample_V2.csv",
                          projects_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\project_sample_V2.csv",
                          schools_path= r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\school_sample_V2.csv",
                          external_path=r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\ZipCodes_AreaContext.csv")
dataloader.do_preprocessing(filter={'Project Current Status': ['Fully Funded']})
dataloader.create_embeddings()
dataloader.create_clustering()

data = dataloader.return_master_data()

del dataloader
gc.collect

graphbuilder = GraphBuilder()
graphbuilder.load_from_dataframe(data)
graphbuilder.preprocess_data()