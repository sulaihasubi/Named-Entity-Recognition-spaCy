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
- [📖 Problem Statements](#-problem-statements)
- [📊 About the Dataset](#-about-the-dataset)
- [🧮 Algorithm](#-algorithm)
- [🖥 The Flow](#-the-flow)
- [📊 Statistic Card](#-statistic-card)
- [🤖 Create Machine Learning Model (Auto ML) & Analysed the Results - Training Models](#-create-machine-learning-model-auto-ml--analysed-the-results---training-models)

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
