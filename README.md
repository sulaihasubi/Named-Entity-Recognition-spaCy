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
- [📊 Label the Data](%-label-the-data)
- [🖥 Train & Test the Model](#--train-the-model)


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


## 📊 Label the Data

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

## 🖥  Train & Test the Model
Use Deep Learning (NN) with a 0.3 dropout rate to avoid overfitting.

The idea is to use a Neural Network with numerous layers and a large number of neurons. We present them text that has already been classified, so the answer is already known. We'll run a lot of iterations, and on each one, we'll calculate the error using a Loss Function, which will modify the weight of the neurons, causing them to fire. The weight of the network will be modified over time in order to eliminate improper learning patterns and solve the problem.

To avoid overfitting, which means the model "memorises" the training data and does not perform well with new data, we remove specific neurons at random on each iteration. This makes it easier for the model to generalise.

Do not forgot to install spaCy first:
```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
```
### Result of the Train Model
```
{'ner': 114.18649069852805}
{'ner': 30.69304090604681}
{'ner': 2.103905114710004}
{'ner': 3.85198211228822}
{'ner': 1.9485266517025366}
{'ner': 0.004584697754935514}
{'ner': 0.0006568707259644221}
{'ner': 0.25235692383820013}
{'ner': 1.399154501044839}
{'ner': 0.005002993828410104}
{'ner': 0.0007396650207534202}
{'ner': 3.399979473944603e-07}
{'ner': 8.389657480840494e-07}
{'ner': 1.4286800035740475e-09}
{'ner': 4.3158335529742466e-07}
{'ner': 1.0808071583551658e-06}
{'ner': 3.095908195536521e-08}
{'ner': 2.361087045811699e-06}
{'ner': 7.632661516871893e-05}
{'ner': 9.230668441407683e-10}
{'ner': 2.430684610377712e-09}
{'ner': 2.0937108988823943e-06}
{'ner': 3.121169423770219e-08}
{'ner': 0.0005652823139269539}
{'ner': 0.003151026469285797}
{'ner': 8.318149025712086e-08}
{'ner': 1.707827043826918e-05}
{'ner': 0.0013131838653112446}
{'ner': 0.0024066308325335723}
{'ner': 3.7774335260993454e-05}
{'ner': 0.4828146020814521}
{'ner': 0.004459282312749806}
{'ner': 4.888743675530538e-07}
{'ner': 1.8908408503194669e-06}
{'ner': 3.453178471484433e-07}
{'ner': 2.7894427019818005e-07}
{'ner': 5.044446233919177e-07}
{'ner': 3.048672749029026e-08}
```
You should see the error decreasing as iterations go by, note that some times it may increase due to the dropout setting and ready to save this model

#### Test the Model
Let's test the model. For this we use displacy which will display the entities in the text.

```
from spacy import displacy
example = "service postings marathon petroleum co said it reduced the contract price it will pay for all grades of service oil one dlr a barrel effective today the decrease brings marathon s posted price for both west texas intermediate and west texas sour to dlrs a bbl the south louisiana sweet grade of service was reduced to dlrs a bbl the company last changed its service postings on jan reuter"

doc = nlp(example)
displacy.render(doc, style='ent')
```
![Alt Text](https://github.com/sulaihasubi/Named-Entity-Recognition-spaCy/blob/main/result/output.png)


## 📖 Conclusion
We have shown how to label data with Doccano and create a custom model with Spacy.

