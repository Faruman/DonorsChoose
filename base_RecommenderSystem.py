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

from sklearn.model_selection import train_test_split

class Recommender(nn.Module):
    def __init__(self, n_donorid, n_projectid, embedding_size):
        super(Recommender, self).__init__()
        self.n_donorid = n_donorid
        self.n_projectid = n_projectid
        self.embedding_size = embedding_size
        self.EmbeddingDonor = nn.Embedding(num_embeddings=embedding_size, embedding_dim=n_donorid)
        self.EmbeddingProject = nn.Embedding(num_embeddings=embedding_size, embedding_dim=n_projectid)
        self.linear = nn.Sequential(
            nn.Linear(embedding_size, 128),
            F.relu,
            nn.Linear(128, 1)
        )

    def forward(self, input):
        donor_embedding = self.EmbeddingDonor(input)
        project_embedding = self.EmbeddingProject(input)
        donor_vecs = torch.reshape(donor_embedding, (self.embedding_size))
        project_vecs = torch.reshape(project_embedding, (self.embedding_size))
        input_vecs = torch.cat((donor_vecs, project_vecs), dim=1)
        y = self.linear(input_vecs)
        return y

class baseRecommender():
    def __init__(self, max_userid: int, max_movieid: int, embedding_size: int, device:str= "cpu"):
        self.device = device
        self.max_userid = max_userid
        self.max_movieid = max_movieid
        self.embedding_size = embedding_size
        self.train = pd.DataFrame()
        self.test = pd.DataFrame()
        # initialize model
        self.model = Recommender(max_userid, max_movieid, embedding_size)
        self.model = self.model.to(device)
        # setup optimizer
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=1e-4)
        self.criterion = nn.MSELoss()
        self.criterion = self.criterion.to(device)


    def provide_data(self, df: pd.DataFrame, train_portion:float= 0.8):
        self.train = df.sample(frac= train_portion, random_state= 42)
        self.test = df.sample(frac= 1- train_portion, random_state=42)

    def generate_objective(self, scoring_fct):
        self.train['score'] = self.train['Donation Amount'].apply(scoring_fct)
        self.test['score'] = self.test['Donation Amount'].apply(scoring_fct)

    def generate_dataLoader(self, batch_size:int = 64):
        self.train_dataLoader = DataLoader((self.train[['user_id', 'proj_id']], self.train['score']), batch_size=batch_size)
        self.test_dataLoader = DataLoader(self.test, batch_size=batch_size)

    def train_model(self, num_epochs:int, model_dir:str):
        self.model.train()

        for epoch in range(num_epochs):
            for batch_idx, (data, target) in enumerate(self.train_dataLoader):
                data = data.to(self.device)
                target = target.flatten().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if (batch_idx + 1) % 100 == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, (batch_idx + 1) * len(data), len(self.train_dataLoader),100. * (batch_idx + 1) / len(self.train_dataLoader), loss.item()))

        # save the model
        torch.save(self.model.state_dict(), model_dir + "simpleNet_4L_2.pt")

    def evaluate_model(self):
        self.model.eval()
        evaluation_df = pd.DataFrame()

        for batch_idx, (data, target) in enumerate(self.test_dataLoader)
            data = data.to(self.device)
            target = target.flatten().to(self.device)

            output = self.model(data)

            pred = output.data.max(1, keepdim=True)[1]
            correct = pred.eq(target.data.view_as(pred)).cpu().detach().sum()

            if (batch_idx + 1) % 100 == 0:
                print('Evaluation: {}/{} ({:.0f}%)'.format((batch_idx + 1) * len(data), len(self.train_dataLoader), 100. * (batch_idx + 1) / len(self.train_dataLoader)))


