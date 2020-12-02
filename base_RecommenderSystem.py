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

import wandb

from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

class Recommender(nn.Module):
    def __init__(self, n_donorid, n_projectid, embedding_size, linear_dim):
        super(Recommender, self).__init__()
        self.n_donorid = n_donorid
        self.n_projectid = n_projectid
        self.embedding_size = embedding_size
        self.EmbeddingDonor = nn.Embedding(num_embeddings=n_donorid, embedding_dim=embedding_size)
        self.EmbeddingProject = nn.Embedding(num_embeddings=n_projectid, embedding_dim=embedding_size)
        self.linear1 = nn.Linear(embedding_size*2, linear_dim)
        self.linear2 = nn.Linear(linear_dim, 1)

    def forward(self, input):
        donor_embedding = self.EmbeddingDonor(input[:, 0])
        project_embedding = self.EmbeddingProject(input[:, 1])
        input_vecs = torch.cat((donor_embedding, project_embedding), dim=1)
        y = self.linear2(F.relu(self.linear1(input_vecs)))
        return y

class baseRecommender():
    def __init__(self, max_userid: int, max_movieid: int, embedding_size: int, linear_dim: int, device:str= "cpu", learning_rate: float= 1e-4):
        self.device = device
        self.max_userid = max_userid
        self.max_movieid = max_movieid
        self.embedding_size = embedding_size
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        # initialize model
        self.model = Recommender(max_userid, max_movieid, embedding_size, linear_dim)
        self.model = self.model.to(device)
        # setup optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(device)


    def load_data(self, df: pd.DataFrame, train_portion:float= 0.7):
        df["user_id"] = df["user_id"].astype(int)
        df["proj_id"] = df["proj_id"].astype(int)
        self.train, temp = train_test_split(df, test_size=1-train_portion, shuffle=True)
        self.test, self.val = train_test_split(temp, test_size=0.5, shuffle=True)

    def generate_objective(self, scoring_fct):
        self.train['score'] = self.train['Donation Amount'].apply(scoring_fct)
        self.val['score'] = self.val['Donation Amount'].apply(scoring_fct)
        self.test['score'] = self.test['Donation Amount'].apply(scoring_fct)

    def generate_dataLoader(self, batch_size:int = 256):
        self.train_dataLoader = DataLoader(torch.LongTensor(self.train[['user_id', 'proj_id', 'score']].values), batch_size=batch_size)
        self.val_dataLoader = DataLoader(torch.LongTensor(self.val[['user_id', 'proj_id', 'score']].values), batch_size=batch_size)
        self.test_dataLoader = DataLoader(torch.LongTensor(self.test[['user_id', 'proj_id', 'score']].values), batch_size=batch_size)

    def train_model(self, num_epochs:int, model_dir:str):
        self.model.train()

        for epoch in range(num_epochs):
            print("Epoch {}".format(epoch))
            print("Train")
            for batch_idx, data in enumerate(self.train_dataLoader):
                target = data[:, -1].type(torch.FloatTensor).unsqueeze(1)
                data = data[:, :-1]

                data = data.to(self.device)
                target = target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 50 == 0:
                    pred_threshold = 0.5
                    pred = (output.detach().cpu() > pred_threshold).float()
                    train_accuracy = accuracy_score(target.detach().cpu().numpy(), pred)
                    train_recall = recall_score(target.detach().cpu().numpy(), pred)
                    train_precision = precision_score(target.detach().cpu().numpy(), pred)
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
        torch.save(self.model.state_dict(), model_dir + "baseLineRecommender.pt")

    def load_model(self, model_path: str):
        self.model.load_state_dict(torch.load(model_path))

    def evaluate_model(self, dataLoader= None, pred_threshold: float= 0.5):
        if not dataLoader:
           dataLoader = self.test_dataLoader
        self.model.eval()
        evaluation_df = pd.DataFrame(columns=['target', 'pred', 'prob'])

        for batch_idx, data in enumerate(dataLoader):
            target = data[:, -1].type(torch.FloatTensor).unsqueeze(1)
            data = data[:, :-1]

            data = data.to(self.device)
            target = target.to(self.device)

            output = self.model(data)

            pred = (output>pred_threshold).float()

            evaluation_df = evaluation_df.append(pd.DataFrame({'target': target.cpu().detach().numpy().flatten(), 'pred': pred.cpu().detach().numpy().flatten(), 'prob': output.cpu().detach().numpy().flatten()}))

            if (batch_idx + 1) % 50 == 0:
                print('Evaluation: {}/{} ({:.0f}%)'.format((batch_idx + 1) * self.train_dataLoader.batch_size, len(self.test_dataLoader) * self.test_dataLoader.batch_size, 100. * (batch_idx + 1) / len(self.test_dataLoader)))

        return evaluation_df

