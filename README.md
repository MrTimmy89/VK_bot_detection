# VK_bot_detection

## The goal
The goal was to analyze different approaches to social network bot detection problem.


## Methodology
The analyzed approaches were:
0. Clusterization with visual analysis (EDA).
1. Random Forest Classification
2. Classic CatBoost classification.
3. Graph neural networks based on one- and two-hop account graphs.
4. GraphSAGE.
5. attri2vec.

## Metrics
The metrics measured were f1 and accuracy.

## The dataset
The dataset provided by <a href="https://data.vk.company/">Big Data Academy</a> professors contained the list of VK account IDs, approved to be automatically managed. On the other hand, a list of non-bot accounts was labeled manually. The lists were relatively same-sized. The lists were used for further parsing of the profiles with the help of python "vk" library. For some GNN methods graph augmentations, described in <a href="https://arxiv.org/abs/2103.00111">arxiv article</a> were used.
