![Picture: Start Slide](https://github.com/Faruman/DonorsChoose/blob/master/imgs/FirstSlide.png?raw=true)

For buidling the recommender system the data from the [Data Science for Good challenge](https://www.kaggle.com/donorschoose/io) on kaggle was used. Here the challenge is described as follows:

> In the second Kaggle Data Science for Good challenge, DonorsChoose.org, in partnership with Google.org, is inviting the community to help them pair up donors to the classroom requests that will most motivate them to make an additional gift. To support this challenge, DonorsChoose.org has supplied anonymized data on donor giving from the past five years. The winning methods will be implemented in DonorsChoose.org email marketing campaigns.

Before creating the model a short evaluation of the use case was done and can be seen below:

![Picture: Use Case](https://github.com/Faruman/DonorsChoose/blob/master/imgs/UseCase.png?raw=true)

After evaluating the plausability of the business perspective we can dive into the data science part of the project. For this the following processing flow was created:

![Picture: Implementation](https://github.com/Faruman/DonorsChoose/blob/master/imgs/Implementation.png?raw=true)

As can be seen the first step of the process requires us to enrich our data. For this information about the economic data of different zip codes from [United States ZIP Codes](unitedstateszipcodes.org) was used.



Here also the data for training the recommender system can be downloaded and should be stored in a folder called data.

