import os

import pandas as pd
import numpy as np

import settings

import gc

import logging
from tqdm import tqdm
tqdm.pandas()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_sequence

import wandb

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

#build custom data loader to use chunked files
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, donors, projects, history, objective, max_history_len=5):
        self.donors = donors
        self.projects = projects
        self.history = history
        self.objective = objective
        self.max_history_len = max_history_len

    def __len__(self):
        return self.donors.shape[0]

    def __getitem__(self, index):
        donor = self.donors.iloc[index, :].values
        donor = torch.from_numpy(donor.astype(float)).type(torch.float32)
        project = self.projects.iloc[index, :].values
        project = torch.from_numpy(np.hstack(project).astype(float)).type(torch.float32)
        history = self.history.iloc[index, :].values
        history = np.apply_along_axis(np.hstack, 1, history[0])
        history = np.pad(history, [(self.max_history_len - history.shape[0], 0), (0, 0)], mode='constant')
        history = torch.from_numpy(history).type(torch.float32)
        objective = torch.from_numpy(np.asarray(self.objective.iloc[index])).type(torch.float32)
        return (donor, project, history), objective

class Recommender(nn.Module):
    def __init__(self, n_donor_columns, donor_linear, n_donor_linear, n_project_columns, project_linear, n_project_linear, n_project_history, project_history_lstm_hidden, n_project_history_lstm, linear1_dim, n_linear1, linear2_dim, n_linear2):
        super(Recommender, self).__init__()

        self.linear_donor_lst = []
        for i in range (0,n_donor_linear):
            if i == 0:
                self.linear_donor_lst.append(nn.Linear(n_donor_columns, donor_linear))
            else:
                self.linear_donor_lst.append(nn.Linear(donor_linear, donor_linear))

        self.linear_project_lst = []
        for i in range(0, n_project_linear):
            if i == 0:
                self.linear_project_lst.append(nn.Linear(n_project_columns, project_linear))
            else:
                self.linear_project_lst.append(nn.Linear(project_linear, project_linear))

        self.lstm_project_history = nn.LSTM(input_size= n_project_history, hidden_size=project_history_lstm_hidden, batch_first=True, num_layers=n_project_history_lstm)

        self.linear1_lst = []
        for i in range(0, n_linear1):
            if i == 0:
                self.linear1_lst.append(nn.Linear(donor_linear+project_linear+project_history_lstm_hidden, linear1_dim))
            else:
                self.linear1_lst.append(nn.Linear(linear1_dim, linear1_dim))

        self.linear2_lst = []
        for i in range(0, n_linear1):
            if i == 0:
                self.linear2_lst.append(nn.Linear(linear1_dim, linear2_dim))
            else:
                self.linear2_lst.append(nn.Linear(linear2_dim, linear2_dim))

        self.linear_final = nn.Linear(linear2_dim, 1)

    def forward(self, donor, project, project_history):
        for i, linear_donor in enumerate(self.linear_donor_lst):
            if i == 0:
                donor_embedding = F.relu(linear_donor(donor))
            else:
                donor_embedding = F.relu(linear_donor(donor_embedding))
        for i, linear_project in enumerate(self.linear_project_lst):
            if i == 0:
                project_embedding = F.relu(linear_project(project))
            else:
                project_embedding = F.relu(linear_project(project_embedding))
        history_embedding, (history_hn, history_cn) = self.lstm_project_history(project_history)
        input_vecs = torch.cat((donor_embedding, project_embedding, history_hn[-1]), dim=1)
        for i, linear1 in enumerate(self.linear1_lst):
            if i == 0:
                hidden = F.relu(linear1(input_vecs))
            else:
                hidden = F.relu(linear1(hidden))
        for i, linear2 in enumerate(self.linear2_lst):
                hidden = F.relu(linear2(hidden))
        y = torch.sigmoid(self.linear4(hidden))
        return y

class advancedRecommender():
    def __init__(self, donor_columns: list, donor_linear:int, n_donor_linear:int, project_columns:list, n_project_columns:int, project_linear:int, n_project_linear:int, project_history_column:list, n_project_history:int, project_history_lstm_hidden:int, n_project_history_lstm:int, linear1_dim:int, n_linear1: int, linear2_dim:int, n_linear2:int, device:str= "cpu", learning_rate: float= 1e-4):
        self.device = device

        self.donor_columns = donor_columns
        self.n_donor_columns = len(donor_columns)
        self.donor_linear = donor_linear
        self.n_donor_linear = n_donor_linear

        self.project_columns = project_columns
        self.n_project_columns = n_project_columns
        self.project_linear = project_linear
        self.n_project_linear = n_project_linear

        self.project_history_column = project_history_column
        self.n_project_history = n_project_history
        self.project_history_lstm_hidden = project_history_lstm_hidden
        self.n_project_history_lstm = n_project_history_lstm

        self.linear1_dim = linear1_dim
        self.n_linear1 = n_linear1
        self.linear2_dim = linear2_dim
        self.n_linear2 = n_linear2

        self.train = pd.DataFrame()
        self.val = pd.DataFrame()
        self.test = pd.DataFrame()
        # initialize model
        self.model = Recommender(self.n_donor_columns, self.donor_linear, self.n_donor_linear, self.n_project_columns, self.project_linear, self.n_project_linear, self.n_project_history, self.project_history_lstm_hidden, self.n_project_history_lstm, self.linear1_dim, self.n_linear1, self.linear2_dim, self.n_linear2)
        self.model = self.model.to(device)
        # setup optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(device)


    def load_data(self, df: pd.DataFrame, train_portion:float= 0.7):
        self.train, temp = train_test_split(df, test_size=1-train_portion, shuffle=True)
        self.test, self.val = train_test_split(temp, test_size=0.5, shuffle=True)

    def generate_objective(self, scoring_fct):
        self.train['score'] = self.train['Donation Amount'].apply(scoring_fct)
        self.val['score'] = self.val['Donation Amount'].apply(scoring_fct)
        self.test['score'] = self.test['Donation Amount'].apply(scoring_fct)

    def generate_dataLoader(self, batch_size:int = 256):
        self.train = CustomDataset(self.train.loc[:, self.donor_columns], self.train.loc[:, self.project_columns], self.train.loc[:, self.project_history_column], self.train.loc[:, 'score'])
        self.val = CustomDataset(self.val.loc[:, self.donor_columns], self.val.loc[:, self.project_columns], self.val.loc[:, self.project_history_column], self.val.loc[:, 'score'])
        self.test = CustomDataset(self.test.loc[:, self.donor_columns], self.test.loc[:, self.project_columns], self.test.loc[:, self.project_history_column], self.test.loc[:, 'score'])

        self.train_dataLoader = DataLoader(self.train, batch_size=batch_size)
        self.val_dataLoader = DataLoader(self.val, batch_size=batch_size)
        self.test_dataLoader = DataLoader(self.test, batch_size=batch_size)

    def train_model(self, num_epochs:int, model_dir:str):

        for epoch in range(num_epochs):
            self.model.train()
            print("Epoch {}".format(epoch))
            print("Train")
            for batch_idx, (data, target) in enumerate(self.train_dataLoader):
                donors, projects, history = data

                donors = donors.to(self.device)
                projects = projects.to(self.device)
                history = history.to(self.device)
                target = target.unsqueeze(1).to(self.device)

                self.optimizer.zero_grad()
                output = self.model(donors, projects, history)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 50 == 0:
                    pred_threshold = 0.5
                    pred = (output.detach().cpu() > pred_threshold).numpy().astype(float)
                    train_accuracy = accuracy_score(target.detach().cpu().numpy(), pred)
                    train_recall = recall_score(target.detach().cpu().numpy().astype(int), pred.astype(int))
                    train_precision = precision_score(target.detach().cpu().numpy().astype(int), pred.astype(int))
                    train_rocauc_score = roc_auc_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())
                    wandb.log({'train_loss': loss.item(), 'train_accuracy': train_accuracy, 'train_recall': train_recall, 'train_precision': train_precision, 'train_rocauc_score': train_rocauc_score})
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * self.train_dataLoader.batch_size, len(self.train_dataLoader) * self.train_dataLoader.batch_size, 100. * (batch_idx + 1) / len(self.train_dataLoader), loss.item()))

            print("Test")
            prediction_df = self.evaluate_model(self.val_dataLoader)
            val_accuracy = accuracy_score(prediction_df['target'], prediction_df['pred'])
            val_recall = recall_score(prediction_df['target'], prediction_df['pred'])
            val_precision = precision_score(prediction_df['target'], prediction_df['pred'])
            val_rocauc_score = roc_auc_score(prediction_df['target'], prediction_df['prob'])
            wandb.log({'val_accuracy': val_accuracy, 'val_recall': val_recall, 'val_precision': val_precision, 'val_rocauc_score': val_rocauc_score})

        # save the model
        prediction_df = self.evaluate_model(self.test_dataLoader)
        test_accuracy = accuracy_score(prediction_df['target'], prediction_df['pred'])
        test_recall = recall_score(prediction_df['target'], prediction_df['pred'])
        test_precision = precision_score(prediction_df['target'], prediction_df['pred'])
        test_rocauc_score = roc_auc_score(prediction_df['target'], prediction_df['prob'])
        wandb.log({'test_accuracy': test_accuracy, 'test_recall': test_recall, 'test_precision': test_precision, 'test_rocauc_score': test_rocauc_score})
        #torch.save(self.model.state_dict(), model_dir + "advancedRecommender.pt")

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))

    def evaluate_model(self, dataLoader= None, pred_threshold: float= 0.5):
        if not dataLoader:
           dataLoader = self.test_dataLoader
        self.model.eval()
        evaluation_df = pd.DataFrame(columns=['target', 'pred', 'prob'])

        for batch_idx, (data, target) in enumerate(dataLoader):
            donors, projects, history = data

            donors = donors.to(self.device)
            projects = projects.to(self.device)
            history = history.to(self.device)
            target = target.to(self.device)

            output = self.model(donors, projects, history)

            pred = (output>pred_threshold).float()

            evaluation_df = evaluation_df.append(pd.DataFrame({'target': target.cpu().detach().numpy().flatten(), 'pred': pred.cpu().detach().numpy().flatten(), 'prob': output.cpu().detach().numpy().flatten()}))

            if (batch_idx + 1) % 50 == 0:
                print('Evaluation: {}/{} ({:.0f}%)'.format((batch_idx + 1) * self.train_dataLoader.batch_size, len(self.test_dataLoader) * self.test_dataLoader.batch_size, 100. * (batch_idx + 1) / len(self.test_dataLoader)))

        return evaluation_df



#data = pd.read_pickle("D:\Programming\Python\DonorsChoose\data\sample\master_data_fasttext_KMeans.pkl.gz")
#data = data.dropna(axis=0)

#def scoring(amount: float):
#    if amount > 0:
#        return 1
#    else:
#        return 0

#project_columns =    ['Teacher Project Posted Sequence', 'Project Type',
#                   'Project Subject Category Tree', 'Project Subject Subcategory Tree', 'Project Grade Level Category',
#                   'Project Resource Category', 'Project Cost', 'Project Fully Funded Date', 'School Metro Type', 'School Percentage Free Lunch', 'School State',
#                   'School Zip', 'School County', 'School District', 'Posted Year', 'Posted Month', 'Project Essay Embedding']
#donor_columns =  ['Donor City', 'Donor State', 'Donor Is Teacher', 'Donor Zip',
#                   'Population', 'Population Density', 'Housing Units', 'Median Home Value', 'Land Area', 'Water Area',
#                   'Occupied Housing Units', 'Median Household Income', 'area_context_cluster']
#project_history_column = ['previous projects']
#n_project_history = 318

#advanced_recommender =  AdvancedRecommender(donor_columns= donor_columns, donor_linear=32, project_columns=project_columns, n_project_columns= len(project_columns)+299, project_linear=32, project_history_column=project_history_column, n_project_history=n_project_history, project_history_lstm_hidden=32, linear1_dim=256, linear2_dim=256, device='cuda:0', learning_rate=0.0001)
#advanced_recommender.load_data(data)
#advanced_recommender.generate_objective(scoring)
#advanced_recommender.generate_dataLoader(batch_size=64)
#advanced_recommender.train_model(2, "./model/")
#prediction_df = advanced_recommender.evaluate_model()