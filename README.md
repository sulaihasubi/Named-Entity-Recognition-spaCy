Named Entity Recognition (NER) with spaCy
========================
[![Built with spaCy](https://img.shields.io/badge/made%20with%20❤%20and-spaCy-09a3d5.svg)](https://spacy.io)

The notebooks should work with any of the Python versions listed below:
========================

[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-368/)
[![Python](https://img.shields.io/badge/python-3.7-blue.svg)](https://www.python.org/downloads/release/python-378/)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-386/)
[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-389/)

## ✍🏻 Table of Contents
- [⌛ Introduction](#-introduction)
- [🖋️ Label the Data](#-label-the-data)
- [🖥 Train the Model](#--train-the-model)


## ⌛ Introduction

The purpose of this notebook is to demonstrate the entire process of name-entity recognition(NER) from start to the end with Spacy. This notebook also explore pattern matching as an alternative to NER when there is a known small set of fixed values.

This will be a complete end-to-end demonstration of the entire process, including both labelling and model training.

In this notebook, we train a model to detect entities related to oil/petrol from this [public dataset](https://www.kaggle.com/mitusha/email-dataset) which contains a list of emails related to the oil industry. This is an over simplification because we want more generic entities, but it shows how pattern matching is a better alternative than NER in this case. To summarise, we will extract oil-related elements from email messages.

Below are the process perform in this notebook:

Read the emails data set which has an email per line.
- Label the emails with the OIL entity using [Doccano](https://doccano.github.io/doccano/tutorial/) labeling tool. This is a manual process.
- Save the labels in a text file as JSONL.
- Use spaCy Neural Network model to train a new statistical model.
- Save the model.
- Create a Spacy NLP pipeline and use the new model to detect oil entities never seen before.
- Finally, use pattern matching instead of a deep learning model to compare both method.

## 🖋️ Label the Data
First Step: Level the data using open source platform **Doccano**.

Follow [Doccano](https://doccano.github.io/doccano/tutorial/) instructions to install and open Doccano.

If you use Linux/Mac, I recommend using the docker image:
```
docker pull doccano/doccano
docker container create --name doccano -e "ADMIN_USERNAME=admin" -e "ADMIN_EMAIL=admin@example.com" -e "ADMIN_PASSWORD=password" -p 8000:8000 doccano/doccano
docker container start doccano
```

For Windows, just use pip:
```
pip install doccano
doccano
```

Go to http://127.0.0.1:8000/

Next, label the data using Doccano. Find entities which talk about oil, petrol, petroleum, etc and label them with the tag **OIL**.

Export the result as **JSONL(Text label)** format.

## 🖥  Train the Model
Use Deep Learning (NN) with a 0.3 dropout rate to avoid overfitting.

The idea is to use a Neural Network with numerous layers and a large number of neurons. We present them text that has already been classified, so the answer is already known. We'll run a lot of iterations, and on each one, we'll calculate the error using a Loss Function, which will modify the weight of the neurons, causing them to fire. The weight of the network will be modified over time in order to eliminate improper learning patterns and solve the problem.

To avoid overfitting, which means the model "memorises" the training data and does not perform well with new data, we remove specific neurons at random on each iteration. This makes it easier for the model to generalise.

Do not forgot to install spaCy first:
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
