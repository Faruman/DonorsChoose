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
TODO: write about configuration and performance

### Advanced Model
TODO: write about configuration and performance

Now that you know the models you can just run the training process for the model specified in the configuration, by executing:

```
python train.py
```

For evaluating different configurations of the models [wandb](https://www.wandb.com/) sweeps can be used, the configurations for running these for the base and the advanced model can be found in the base_sweep.yaml and the advanced_sweep.yaml.



