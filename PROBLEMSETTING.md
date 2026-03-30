# aml2026-group-13
This is the Project repository for the UZH Module "Advanced Machine Learning".

Possible Projects as baselines: https://github.com/Bhavik-Jikadara/fake-news-detections
- uses the '''https://www.kaggle.com/datasets/bhavikjikadara/fake-news-detection''' as Dataset
    - Kaggle has many different Datasets for Fake News detection/classification, we choose this one in particular, because it has a usability score of 10/10 and it seems to be well maintained.
    - Fake News: This Data spans the years 2015-2018 in News (39%) and Politics (29%) and Others (32%), a total of 23.5k entries.
    - True News: This Data spans the years 2016-2017 in PoliticsNews (53%) and worldnews (47%), a total of 21.4k entries.


### Formal Problem Setting: (Draft)

We define the task of fake news detection as a supervised binary classification problem on natural language sequences. In this setting, we are given a news article represented as a sequence of tokens. Our objective is to learn a mapping function that takes both the sequence and a vector of hand-crafted linguistic features as input to predict a specific label. The output label represents the veracity of the article, where a value of one denotes fake news, representing fabricated or misleading content, and a value of zero denotes real news, representing verified factual reporting. The hand-crafted feature vector specifically captures external markers such as sentiment, readability, and lexical diversity extracted from the text.

Formally, let $\mathcal{V}$ be a vocabulary. Given a news article represented as a token sequence

$$\mathbf{x} = (x_1, x_2, \ldots, x_T),\ x_t \in \mathcal{V}$$

and a vector capturing external markers

$$\mathbf{z} = \phi(\mathbf{x}) \in \mathbb{R}^d$$

we learn a mapping

$$f_{\boldsymbol{\theta}} : (\mathbf{x}, \mathbf{z}) \rightarrow \hat{y} \in \{0, 1\}$$

where $y = 1$ denotes fake news and $y = 0$ denotes real news.

#### Goal:

Our goal is to find the optimal model parameters that minimize the empirical risk on the training data to approximate the true risk. We will evaluate the model's performance using a stratified split consisting of seventy percent for training, fifteen percent for validation, and fifteen percent for final testing. To measure the model’s ability to separate classes effectively, especially if the dataset is imbalanced, we will use the Area Under the Receiver Operating Characteristic curve (AUROC) as our primary metric. Additionally, we will monitor the F1-Score to ensure a high-quality balance between precision and recall.

#### Proposed Approach: Hybrid Transformer-Linguistic Fusion

We propose a hybrid architecture that combines deep contextual embeddings with explicit linguistic markers to identify patterns common in misinformation. Standard transformer fine-tuning relies solely on the latent representation of the classification token. However, fake news often utilizes specific emotional triggers and simplified language structures to increase virality. Our approach modifies the standard architecture by creating two distinct processing branches.

The first branch is the contextual branch, where we use DistilBERT to extract a seven hundred and sixty-eight dimensional contextual vector from the article. The second branch is the linguistic branch, where we calculate an auxiliary feature vector containing sentiment polarity to capture the emotional charge of sensationalist news and the Gunning Fog Index to measure the readability and complexity of the text. Instead of feeding the transformer output directly to the classification layer, we concatenate the contextual and linguistic vectors into a single hybrid representation. This fused vector is then passed through a multi-layer perceptron with dropout to produce the final classification.

#### Evaluation Protocoll:
We will benchmark this hybrid model against two specific baselines: a vanilla DistilBERT model to quantify the performance gain provided by our linguistic feature injection, and a Random Forest classifier using TF-IDF features to establish a traditional statistical baseline. Finally, we will tune hyperparameters including the learning rate, batch size, and dropout probability on the validation set to ensure that the hand-crafted features effectively complement the transformer’s learned representations.
