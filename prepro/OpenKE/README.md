# OpenKE-PyTorch

An Open-source Framework for Knowledge Embedding implemented with PyTorch.

More information is available on our website 
[http://openke.thunlp.org/](http://openke.thunlp.org/)


## Overview

This is an Efficient implementation based on PyTorch for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs. OpenKE composes 4 repositories:

OpenKE-PyTorch: the project based on PyTorch, which provides the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/OpenKE/tree/OpenKE-Tensorflow1.0"> OpenKE-Tensorflow1.0</a>: OpenKE implemented with TensorFlow, also providing the optimized and stable framework for knowledge graph embedding models.

<a href="https://github.com/thunlp/TensorFlow-TransX"> TensorFlow-TransX</a>: light and simple version of OpenKE based on TensorFlow, including TransE, TransH, TransR and TransD. 

<a href="https://github.com/thunlp/Fast-TransX"> Fast-TransX</a>: efficient lightweight C++ inferences for TransE and its extended models utilizing the framework of OpenKE, including TransH, TransR, TransD, TranSparse and PTransE. 




## Models

OpenKE (PyTorch): 

*  DistMult, ComplEx, Analogy
*  TransE, TransH, TransR, TransD
*	 RotatE

We welcome any issues and requests for model implementation and bug fix.

## Experimental Settings

For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. The scores of those corrupted triplets are first computed by the models and then sorted by the order. Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). The mean rank (MRR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.

Because some corrupted triplets may be in the training set and validation set. In this case, those corrupted triplets may be ranked above the test triplet, but this should not be counted as an error because both triplets are true. Hence, we remove those corrupted triplets appearing in the training, validation or test set, which ensures the corrupted triplets are not in the dataset. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. The mean rank (MRR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.

More details of the above-mentioned settings can be found from the papers [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf), [ComplEx](http://proceedings.mlr.press/v48/trouillon16.pdf).

For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. Hence, we also provide the experimental setting named "[type constraint](https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf)" to corrupt entities with some limited entity sets determining by their relations.




## Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/)

2. Clone the OpenKE-PyTorch branch:
```bash
git clone -b OpenKE-PyTorch https://github.com/thunlp/OpenKE --depth 1
cd OpenKE
cd openke
```
3. Compile C++ files => release/Base.so
```bash
bash make.sh
```	
4. Quick start (example)
```bash
cd ../
cp examples/train_transe_FB15K237.py ./
python train_transe_FB15K237.py
```	
## Data

* For training, datasets contain three files:

  train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** .
  **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**

  entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.

  relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.

  

