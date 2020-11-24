# CFANE

implementation of paper **Unsupervised Attributed Network Embedding via Cross Fusion** (WSDM 2021)



## Requirements

* python >= 3.6
* pytorch >= 1.4
* network >= 1.11



## Usage

For default parameters, use following command

> python train.py

If you want to use other parameter setting and dataset, you can use following command for optional arguments description.

> python train.py -h



## Repository contents

| file           | description                                                  |
| -------------- | ------------------------------------------------------------ |
| train.py       | The main training code                                       |
| model.py       | The implementation of CFANE                                  |
| aggregators.py | Aggregators for information propagation                      |
| utils.py       | Data loading                                                 |
| node2vec.py    | Generating random walk contexts. Refer to: https://github.com/aditya-grover/node2vec |



## Datasets

We provide Cora dataset and partitions of its ego-network as example of data format.

Our ego-network partition refers to https://github.com/google-research/google-research/tree/master/graph_embedding/persona

