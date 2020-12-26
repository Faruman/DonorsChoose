![Picture: Start Slide](https://github.com/Faruman/DonorsChoose/blob/master/imgs/FirstSlide.png?raw=true)

For buidling the recommender system the data from the [Data Science for Good challenge](https://www.kaggle.com/donorschoose/io) on kaggle was used. Here the challenge is described as follows:

> In the second Kaggle Data Science for Good challenge, DonorsChoose.org, in partnership with Google.org, is inviting the community to help them pair up donors to the classroom requests that will most motivate them to make an additional gift. To support this challenge, DonorsChoose.org has supplied anonymized data on donor giving from the past five years. The winning methods will be implemented in DonorsChoose.org email marketing campaigns.

Before creating the model a short evaluation of the use case was done and can be seen below:

![Picture: Use Case](https://github.com/Faruman/DonorsChoose/blob/master/imgs/UseCase.png?raw=true)

After evaluating the plausability of the business perspective we can dive into the data science part of the project. For this the following processing flow was created:

![Picture: Implementation](https://github.com/Faruman/DonorsChoose/blob/master/imgs/Implementation.png?raw=true)

Before we start with the preprocessing we need to set up the file structure for our data. This should look as follows:

```
data
 ├── DonorsChoose
 ├── EconomicIndicators
 └── fasttext
```

Next the data for training the recommender system needs to be downloaded from the [kaggle page](https://www.kaggle.com/donorschoose/io/download) and should be stored in the DonorsChoose folder we just created. Furthermore for creating the embeddings, the [pretrained fasttext vectors](https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz) needs to be downloaded into the fasttext folder.

If you want to run the model on a reduced dataset first, create the following folder:

```
data
 └── DonorsChoose
    └── sample
```

Than run the script pushToCsv_short.py from the preprocessing folder.

As can be seen the first step of the process requires us to enrich our data. For this information about the economic data of different zip codes from [United States ZIP Codes](unitedstateszipcodes.org) was used. To do this run the script CreateDonorLocalData.py, as your ip will be blocked after doing too many requests this script uses NordVPN to switch it during the process.

All the other steps of the preprocessing will be done by running the main script and the appropriate options can be set in the config.json.

Furthermore, the configuration also allows you to select the model to be run and to set the configuration for each model. Currently, the following two models are implemented:

![Picture: Architecture](https://github.com/Faruman/DonorsChoose/blob/master/imgs/Architecture.png?raw=true)

While the base model just creates embeddings based on the cooccurance of models and projects, the adavanced model tries to incoorporate more of the information given for projects and donors.

### Base Model

```Python
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
        y = torch.sigmoid(self.linear2(F.relu(self.linear1(input_vecs))))
        return y
```

Next a wandb sweep was done to find the optimal parameter combination. This sweeps can be seen bellow:

[Picture: Parameter Sweep Base Model](https://github.com/Faruman/DonorsChoose/blob/master/imgs/SweepBaseRecommenderSystem.png?raw=true)

From the XX runs the best parameter combination was:

```

```

Achieving the following scores:

```

```

### Advanced Model

```Python
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
```

TODO: write about configuration and performance

Now that you know the models you can just run the training process for the model specified in the configuration, by executing:

```
python train.py
```

For evaluating different configurations of the models [wandb](https://www.wandb.com/) sweeps can be used, the configurations for running these for the base and the advanced model can be found in the base_sweep.yaml and the advanced_sweep.yaml.



