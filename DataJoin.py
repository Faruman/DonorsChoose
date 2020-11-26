import os
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

projects = pd.read_csv(r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\project_sample_V2.csv")
projects = projects.dropna(subset=['Project Need Statement'])
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(projects['Project Need Statement'].values)]

#projects = projects.sample(5, random_state=42)
model_path = "D:\Programming\Python\DonorsChoose\model\d2v.model"
if not os.path.isfile(model_path):
    model = Doc2Vec(size=10, alpha=0.025, min_alpha=0.00025, min_count=1, dm =1)
    model.build_vocab(tagged_data)
    for epoch in range(100):
        print('iteration {0}'.format(epoch))
        model.train(tagged_data, total_examples=model.corpus_count, epochs=model.iter)
        # decrease the learning rate
        model.alpha -= 0.0002
        # fix the learning rate, no decay
        model.min_alpha = model.alpha
    model.save("D:\Programming\Python\DonorsChoose\model\d2v.model")
    print("Model Saved")
else:
    model = Doc2Vec.load(model_path)

def create_embeddig(row):
    infer = model.infer_vector(word_tokenize(row['Project Need Statement'].lower()))
    for i, dim in enumerate(infer):
        row['Project Need Statement Dim {}'.format(i+1)] = dim
    return row
projects = projects.apply(create_embeddig, axis=1)

schools = pd.read_csv(r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\school_sample_V2.csv")
teachers = pd.read_csv(r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\teacher_sample_V2.csv")

economicIndicators = pd.read_excel(r"D:\Programming\Python\DonorsChoose\data\EconomicIndicators\United States.xlsx")
economicIndicators = economicIndicators.loc[economicIndicators["Year"] == 2018].iloc[:, :-2]

df = pd.merge(projects, schools, on= "School ID", how="left")
df = pd.merge(df, teachers, on= "Teacher ID", how="left")
df = pd.merge(df, economicIndicators, left_on="School State", right_on="State", how="left")

df.to_excel(r"D:\Programming\Python\DonorsChoose\data\DonorsChoose\sample\projectWithAdditionalInfos_sample.xlsx", engine='xlsxwriter')